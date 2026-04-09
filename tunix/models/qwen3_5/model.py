# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Qwen3.5 model (hybrid full-attention + Gated Delta Net linear-attention).

Architecture overview
---------------------
Qwen3.5 is a hybrid transformer where most decoder layers use standard
multi-head attention (Qwen3-style GQA), but every ``full_attention_interval``-th
layer (default every 4th) is replaced by a *Gated Delta Net* linear-attention
layer.  The latter maintains a conv1d state and a recurrent (SSM-like) state
alongside the standard KV cache.

Key differences from Qwen3
--------------------------
* ``RMSNorm`` formula: ``x * rsqrt(mean(x²)+ε) * (1 + w)`` with w=0 init.
* ``Attention``: gated output ``out * sigmoid(gate)``, partial RoPE.
* ``GatedDeltaNet`` layer: causal conv1d + chunk/recurrent delta rule.
* Extended cache: per-layer ``conv_state`` and ``recurrent_state`` in addition
  to the usual ``k``/``v``/``end_index``.
"""

from __future__ import annotations

import dataclasses
import enum
from typing import Tuple

import flax
from flax import nnx
import jax
from jax import numpy as jnp
from jax.experimental.pallas.ops.tpu.splash_attention import (
    splash_attention_kernel as splash,
)
from jax.experimental.pallas.ops.tpu.splash_attention import (
    splash_attention_mask as mask_lib,
)
from jax.experimental.shard_map import shard_map
from jax.interpreters import pxla
import jax.sharding as shd
from jax.sharding import PartitionSpec as P
import jaxtyping
from tunix.generate.mappings import BackendMappingMixin
from tunix.models.qwen3_5.gated_delta_rule import (
    chunk_gated_delta_rule,
    recurrent_gated_delta_rule,
)
from tunix.utils import compat
from tunix.utils import env_utils

env_utils.setup_sharding_environment()

K_MASK = -2.3819763e38

LayerCache = dict[str, jaxtyping.Array]
Cache = dict[str, LayerCache]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class RematConfig(enum.Enum):
  NONE = enum.auto()
  BLOCK = enum.auto()


@dataclasses.dataclass(slots=True, frozen=True)
class ShardingConfig:
  """Sharding configuration for Qwen3.5."""

  emb_vd: Tuple[str | None, ...]
  emb_dv: Tuple[str | None, ...]
  q_weight_dnh: Tuple[str | None, ...]
  kv_weight_dnh: Tuple[str | None, ...]
  o_weight_nhd: Tuple[str | None, ...]
  ffw_weight_df: Tuple[str | None, ...]
  ffw_weight_fd: Tuple[str | None, ...]
  rms_norm_weight: Tuple[str | None, ...]
  act_btd: Tuple[str | None, ...]
  act_btf: Tuple[str | None, ...]
  act_btnh: Tuple[str | None, ...]
  # linear-attention weights keep embedding dim unsharded (replicated)
  linear_weight_dd: Tuple[str | None, ...]

  @staticmethod
  def get_default_sharding(
      is_sampling: bool = False, enable_sp: bool = False
  ) -> 'ShardingConfig':
    fsdp = 'fsdp' if not is_sampling else None
    sp = 'sp' if (not is_sampling and enable_sp) else None
    fsdp = (fsdp, sp) if fsdp and sp else fsdp
    return ShardingConfig(
        emb_vd=('tp', fsdp),
        emb_dv=(fsdp, 'tp'),
        q_weight_dnh=(fsdp, 'tp', None),
        kv_weight_dnh=(fsdp, 'tp', None),
        o_weight_nhd=('tp', None, fsdp),
        ffw_weight_df=(fsdp, 'tp'),
        ffw_weight_fd=('tp', fsdp),
        rms_norm_weight=('tp',),
        act_btd=('fsdp', sp, None if is_sampling else 'tp'),
        act_btf=('fsdp', sp, 'tp'),
        act_btnh=('fsdp', sp, 'tp', None),
        linear_weight_dd=(fsdp, None),
    )


@dataclasses.dataclass(slots=True)
class ModelConfig:
  """Configuration for Qwen3.5."""

  num_layers: int
  vocab_size: int
  embed_dim: int
  hidden_dim: int
  num_heads: int
  head_dim: int
  num_kv_heads: int
  rope_theta: float
  norm_eps: float
  # partial RoPE: fraction of head_dim to rotate
  partial_rotary_factor: float = 1.0
  # layer types: list of 'full_attention' | 'linear_attention'
  # If None, inferred from full_attention_interval.
  layer_types: list[str] | None = None
  full_attention_interval: int = 4
  # Linear-attention head/dim config
  linear_num_key_heads: int = 16
  linear_num_value_heads: int = 16
  linear_key_head_dim: int = 64
  linear_value_head_dim: int = 64
  linear_conv_kernel_dim: int = 4
  # standard options
  use_tied_embedding: bool = False
  shd_config: ShardingConfig = dataclasses.field(
      default_factory=ShardingConfig.get_default_sharding
  )
  remat_config: RematConfig = RematConfig.NONE
  dtype: jnp.dtype = jnp.float32
  param_dtype: jnp.dtype = jnp.float32
  use_flash_attention: bool = False
  flash_attention_block_size: int = 1024

  def __post_init__(self):
    if self.layer_types is None:
      interval = self.full_attention_interval
      self.layer_types = [
          'linear_attention' if (i + 1) % interval else 'full_attention'
          for i in range(self.num_layers)
      ]

  # ------------------------------------------------------------------
  # Standard Qwen3.5 model configs
  # ------------------------------------------------------------------

  @classmethod
  def qwen3_5_0p8b(cls):
    """Matches Qwen/Qwen3.5-0.8B HF checkpoint (text tower of the VLM).

    Source: text_config in config.json:
      hidden_size=1024, num_hidden_layers=24, num_attention_heads=8,
      head_dim=256, num_key_value_heads=2, intermediate_size=3584,
      vocab_size=248320, partial_rotary_factor=0.25, rope_theta=10_000_000,
      layer_types: linear*3 + full every 4th (24 layers total).
    """
    layer_types = [
        'linear_attention', 'linear_attention', 'linear_attention', 'full_attention',
        'linear_attention', 'linear_attention', 'linear_attention', 'full_attention',
        'linear_attention', 'linear_attention', 'linear_attention', 'full_attention',
        'linear_attention', 'linear_attention', 'linear_attention', 'full_attention',
        'linear_attention', 'linear_attention', 'linear_attention', 'full_attention',
        'linear_attention', 'linear_attention', 'linear_attention', 'full_attention',
    ]
    return cls(
        num_layers=24,
        vocab_size=248320,
        embed_dim=1024,
        hidden_dim=3584,
        num_heads=8,
        head_dim=256,
        num_kv_heads=2,
        norm_eps=1e-6,
        rope_theta=10_000_000,
        partial_rotary_factor=0.25,
        layer_types=layer_types,
        linear_num_key_heads=16,
        linear_num_value_heads=16,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
        use_tied_embedding=True,
    )

  @classmethod
  def qwen3_5_1p5b(cls):
    return cls(
        num_layers=28,
        vocab_size=151936,
        embed_dim=1536,
        hidden_dim=8960,
        num_heads=12,
        head_dim=128,
        num_kv_heads=8,
        norm_eps=1e-6,
        rope_theta=1_000_000,
        partial_rotary_factor=0.625,
        linear_num_key_heads=8,
        linear_num_value_heads=8,
        linear_key_head_dim=64,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
        use_tied_embedding=True,
    )

  @classmethod
  def qwen3_5_7b(cls):
    return cls(
        num_layers=28,
        vocab_size=151936,
        embed_dim=4096,
        hidden_dim=22016,
        num_heads=32,
        head_dim=128,
        num_kv_heads=8,
        norm_eps=1e-6,
        rope_theta=1_000_000,
        partial_rotary_factor=0.625,
        linear_num_key_heads=16,
        linear_num_value_heads=16,
        linear_key_head_dim=64,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
    )

  @classmethod
  def qwen3_5_14b(cls):
    return cls(
        num_layers=40,
        vocab_size=151936,
        embed_dim=5120,
        hidden_dim=27648,
        num_heads=40,
        head_dim=128,
        num_kv_heads=8,
        norm_eps=1e-6,
        rope_theta=1_000_000,
        partial_rotary_factor=0.625,
        linear_num_key_heads=16,
        linear_num_value_heads=16,
        linear_key_head_dim=64,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
    )

  @classmethod
  def qwen3_5_32b(cls):
    return cls(
        num_layers=64,
        vocab_size=151936,
        embed_dim=5120,
        hidden_dim=25600,
        num_heads=64,
        head_dim=128,
        num_kv_heads=8,
        norm_eps=1e-6,
        rope_theta=1_000_000,
        partial_rotary_factor=0.625,
        linear_num_key_heads=16,
        linear_num_value_heads=16,
        linear_key_head_dim=64,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
    )

# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------


def shard(x: jnp.ndarray, s: Tuple[str, ...]):
  mesh = pxla.thread_resources.env.physical_mesh
  if mesh.empty or jax.devices()[0].platform == 'cpu':
    return x
  return jax.lax.with_sharding_constraint(
      x, shd.NamedSharding(mesh, shd.PartitionSpec(*s))
  )


def apply_rope(
    inputs: jaxtyping.Array,
    positions: jaxtyping.Array,
    head_dim: int,
    rope_theta: float = 1_000_000,
) -> jaxtyping.Array:
  """Applies RoPE to the full head_dim."""
  fraction = 2 * jnp.arange(0, head_dim // 2, dtype=jnp.float32) / head_dim
  timescale = rope_theta**fraction
  sinusoid_inp = (
      positions[..., jnp.newaxis] / timescale[jnp.newaxis, jnp.newaxis, :]
  )
  sinusoid_inp = sinusoid_inp[..., jnp.newaxis, :]
  sin = jnp.sin(sinusoid_inp).astype(inputs.dtype)
  cos = jnp.cos(sinusoid_inp).astype(inputs.dtype)
  first_half, second_half = jnp.split(inputs, 2, axis=-1)
  first_part = first_half * cos - second_half * sin
  second_part = second_half * cos + first_half * sin
  return jnp.concatenate([first_part, second_part], axis=-1).astype(
      inputs.dtype
  )


def apply_partial_rope(
    q: jaxtyping.Array,
    k: jaxtyping.Array,
    positions: jaxtyping.Array,
    rotary_dim: int,
    rope_theta: float,
) -> Tuple[jaxtyping.Array, jaxtyping.Array]:
  """RoPE applied only to the first ``rotary_dim`` dimensions of each head."""
  if rotary_dim <= 0:
    return q, k
  q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
  k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
  q_rot = apply_rope(q_rot, positions, rotary_dim, rope_theta)
  k_rot = apply_rope(k_rot, positions, rotary_dim, rope_theta)
  return (
      jnp.concatenate([q_rot, q_pass], axis=-1),
      jnp.concatenate([k_rot, k_pass], axis=-1),
  )


# ---------------------------------------------------------------------------
# Learnable modules
# ---------------------------------------------------------------------------


class Einsum(nnx.Module):
  """Convenience module for parameterised tensor contractions."""

  def __init__(
      self,
      einsum_str: str,
      shape: flax.typing.Shape,
      *,
      rngs: nnx.Rngs,
      sharding: Tuple[str | None, ...],
      dtype: jnp.dtype,
      param_dtype: jnp.dtype,
  ):
    self.einsum_str = einsum_str
    self.shape = shape
    self.dtype = dtype
    self.w = nnx.Param(
        nnx.initializers.glorot_uniform()(rngs.params(), shape, dtype=param_dtype),
        sharding=sharding,
    )

  @jax.named_scope('einsum')
  def __call__(self, x: jaxtyping.ArrayLike) -> jaxtyping.Array:
    x = jnp.astype(x, self.dtype)
    w = jnp.astype(self.w.value, self.dtype)
    return jnp.einsum(self.einsum_str, x, w)


class Embedder(nnx.Module):
  """Token embedding + tied LM-head decode."""

  def __init__(
      self,
      vocab_size: int,
      embed_dim: int,
      *,
      rngs: nnx.Rngs,
      shd_config: ShardingConfig,
      dtype: jnp.dtype,
      param_dtype: jnp.dtype,
  ):
    self.input_embedding = nnx.Param(
        nnx.initializers.normal(dtype=param_dtype)(
            rngs.params(), (vocab_size, embed_dim)
        ),
        sharding=shd_config.emb_vd,
    )
    self.shd_config = shd_config
    self.dtype = dtype

  @jax.named_scope('embedder_encode')
  def encode(self, x: jaxtyping.ArrayLike) -> jaxtyping.Array:
    x = self.input_embedding[(x,)]
    x = jnp.astype(x, self.dtype)
    return shard(x, self.shd_config.act_btd)

  @jax.named_scope('embedder_decode')
  def decode(self, x: jaxtyping.ArrayLike) -> jaxtyping.Array:
    x = jnp.astype(x, self.dtype)
    w = jnp.astype(self.input_embedding.value, self.dtype)
    return jnp.dot(x, w.T)


class RMSNorm(nnx.Module):
  """Standard RMSNorm (ones-init weight), used in MLP / classic layers."""

  def __init__(
      self,
      dim: int,
      *,
      norm_eps: float = 1e-6,
      rngs: nnx.Rngs,
      shd_config: ShardingConfig,
      dtype: jnp.dtype,
      param_dtype: jnp.dtype,
  ):
    self.w = nnx.Param(
        nnx.initializers.ones_init()(rngs.params(), dim, param_dtype),
        sharding=shd_config.rms_norm_weight,
    )
    self.norm_eps = norm_eps
    self.dtype = dtype

  @jax.named_scope('rms_norm')
  def __call__(self, x: jaxtyping.Array) -> jaxtyping.Array:
    x = jnp.astype(x, jnp.float32)
    rms = jnp.sqrt(jnp.mean(x**2, axis=-1, keepdims=True) + self.norm_eps)
    return jnp.astype(
        jnp.astype(self.w.value, jnp.float32) * (x / rms), self.dtype
    )


class Qwen3_5RMSNorm(nnx.Module):
  """Qwen3.5-style RMSNorm: ``x/rms * (1 + w)`` with w **zero-initialized**.

  This matches the HuggingFace Qwen3.5 convention and differs from the
  standard tunix RMSNorm (which uses ones-init and ``w * x/rms``).
  """

  def __init__(
      self,
      dim: int,
      *,
      norm_eps: float = 1e-6,
      rngs: nnx.Rngs,
      dtype: jnp.dtype,
  ):
    self.w = nnx.Param(
        nnx.initializers.zeros_init()(rngs.params(), (dim,), jnp.float32),
    )
    self.norm_eps = norm_eps
    self.dtype = dtype

  @jax.named_scope('qwen3_5_rms_norm')
  def __call__(self, x: jaxtyping.Array) -> jaxtyping.Array:
    out = x * jax.lax.rsqrt(
        jnp.mean(x * x, axis=-1, keepdims=True) + self.norm_eps
    )
    return (out * (1.0 + self.w.value)).astype(self.dtype)


class Qwen3_5RMSNormGated(nnx.Module):
  """Gated RMSNorm: ``x/rms * w * silu(gate)`` with w ones-initialized."""

  def __init__(
      self,
      dim: int,
      *,
      norm_eps: float = 1e-6,
      rngs: nnx.Rngs,
      dtype: jnp.dtype,
  ):
    self.w = nnx.Param(
        nnx.initializers.ones_init()(rngs.params(), (dim,), jnp.float32),
    )
    self.norm_eps = norm_eps
    self.dtype = dtype

  def __call__(
      self, hidden_states: jaxtyping.Array, gate: jaxtyping.Array
  ) -> jaxtyping.Array:
    out = hidden_states * jax.lax.rsqrt(
        jnp.mean(hidden_states * hidden_states, axis=-1, keepdims=True)
        + self.norm_eps
    )
    return (out * self.w.value * nnx.silu(gate)).astype(self.dtype)


# ---------------------------------------------------------------------------
# Full-attention block
# ---------------------------------------------------------------------------


class Attention(nnx.Module):
  """Full multi-head attention (GQA) with partial RoPE and optional gate."""

  def __init__(self, config: ModelConfig, *, rngs: nnx.Rngs):
    self.config = config
    self.shd_config = config.shd_config

    # Compute rotary_dim from partial_rotary_factor
    raw_rot = int(config.head_dim * config.partial_rotary_factor)
    self.rotary_dim = min(config.head_dim, raw_rot - (raw_rot % 2))
    self.rope_theta = config.rope_theta

    # Q projection outputs 2*head_dim per head: [head_features, gate_features]
    self.q_proj = Einsum(
        einsum_str='BTD,DNH->BTNH',
        shape=(config.embed_dim, config.num_heads, config.head_dim * 2),
        rngs=rngs,
        sharding=self.shd_config.q_weight_dnh,
        dtype=config.dtype,
        param_dtype=config.param_dtype,
    )
    self.k_proj = Einsum(
        einsum_str='BSD,DKH->BSKH',
        shape=(config.embed_dim, config.num_kv_heads, config.head_dim),
        rngs=rngs,
        sharding=self.shd_config.kv_weight_dnh,
        dtype=config.dtype,
        param_dtype=config.param_dtype,
    )
    self.v_proj = Einsum(
        einsum_str='BSD,DKH->BSKH',
        shape=(config.embed_dim, config.num_kv_heads, config.head_dim),
        rngs=rngs,
        sharding=self.shd_config.kv_weight_dnh,
        dtype=config.dtype,
        param_dtype=config.param_dtype,
    )
    self.o_proj = Einsum(
        einsum_str='BTNH,NHD->BTD',
        shape=(config.num_heads, config.head_dim, config.embed_dim),
        rngs=rngs,
        sharding=self.shd_config.o_weight_nhd,
        dtype=config.dtype,
        param_dtype=config.param_dtype,
    )
    self.q_norm = Qwen3_5RMSNorm(
        config.head_dim, norm_eps=config.norm_eps, rngs=rngs, dtype=config.dtype
    )
    self.k_norm = Qwen3_5RMSNorm(
        config.head_dim, norm_eps=config.norm_eps, rngs=rngs, dtype=config.dtype
    )
    self.n_rep = config.num_heads // config.num_kv_heads
    self.scale = config.head_dim**-0.5

  def block(
      self,
      x: jaxtyping.Array,
      segment_pos: jaxtyping.Array,
      cache: LayerCache | None,
      attn_mask: jaxtyping.Array | None,
  ) -> tuple[LayerCache | None, jaxtyping.Array]:
    b, t, _ = x.shape

    # q_raw: [B, T, N, 2*H]  (head_features ‖ gate_features)
    q_raw = self.q_proj(x)
    q, gate = jnp.split(q_raw, 2, axis=-1)  # each [B, T, N, H]
    gate_flat = gate.reshape(b, t, self.config.num_heads * self.config.head_dim)

    query_proj = self.q_norm(q)    # [B, T, N, H]
    key_proj = self.k_norm(self.k_proj(x))
    value_proj = self.v_proj(x)

    query_proj = shard(query_proj, self.shd_config.act_btnh)
    key_proj = shard(key_proj, self.shd_config.act_btnh)
    value_proj = shard(value_proj, self.shd_config.act_btnh)

    query_proj, key_proj = apply_partial_rope(
        query_proj, key_proj, segment_pos, self.rotary_dim, self.rope_theta
    )

    if cache is not None:
      end_index = cache['end_index'][0]
      slice_indices = (0, end_index % cache['v'].shape[1], 0, 0)
      value_proj = jax.lax.dynamic_update_slice(
          cache['v'], value_proj, slice_indices
      )
      key_proj = jax.lax.dynamic_update_slice(
          cache['k'], key_proj, slice_indices
      )

    _, _, qh, d = query_proj.shape
    _, _, kh, _ = key_proj.shape

    if self.config.use_flash_attention and t > 1:
      # Splash attention (TPU flash attention)
      query_proj = query_proj.transpose(0, 2, 1, 3)
      key_proj = key_proj.transpose(0, 2, 1, 3)
      value_proj = value_proj.transpose(0, 2, 1, 3)
      query_proj = query_proj * self.scale

      mesh = pxla.thread_resources.env.physical_mesh
      causal_mask = mask_lib.CausalMask((t, t))
      multi_head_mask = mask_lib.MultiHeadMask(
          [causal_mask for _ in range(qh)]
      )
      block_sizes = splash.BlockSizes(
          block_q=self.config.flash_attention_block_size,
          block_kv=self.config.flash_attention_block_size,
          block_q_dkv=self.config.flash_attention_block_size,
          block_kv_dkv=self.config.flash_attention_block_size,
          block_kv_dkv_compute=self.config.flash_attention_block_size,
          block_q_dq=self.config.flash_attention_block_size,
          block_kv_dq=self.config.flash_attention_block_size,
      )
      shd_b, shd_t, shd_n, shd_h = self.shd_config.act_btnh
      head_shards = (
          mesh.shape[shd_n]
          if shd_n is not None and shd_n in mesh.shape
          else 1
      )
      q_seq_shards = (
          mesh.shape[shd_t]
          if shd_t is not None and shd_t in mesh.shape
          else 1
      )
      splash_attn_kernel = splash.make_splash_mha(
          multi_head_mask,
          block_sizes=block_sizes,
          head_shards=head_shards,
          q_seq_shards=q_seq_shards,
      )
      shd_spec = P(shd_b, shd_n, shd_t, shd_h)
      unsharded_seq = P(shd_b, shd_n, None, shd_h)
      kernel_spec = splash_attn_kernel.manual_sharding_spec(
          shd.NamedSharding(mesh, P(shd_n, shd_t))
      )

      @shard_map(
          mesh=mesh,
          in_specs=(kernel_spec, shd_spec, unsharded_seq, unsharded_seq),
          out_specs=shd_spec,
          check_rep=False,
      )
      def sharded_splash_attn(kernel, q_block, k_block, v_block):
        return jax.vmap(kernel)(q_block, k_block, v_block)

      qkv = sharded_splash_attn(
          splash_attn_kernel, query_proj, key_proj, value_proj
      )
      qkv = qkv.transpose(0, 2, 1, 3)
    else:
      # Standard GQA
      query_proj = query_proj.reshape((b, t, kh, qh // kh, d))
      attn = (
          jnp.einsum('BTHGD,BSHD->BHGTS', query_proj, key_proj) * self.scale
      )
      if attn_mask is not None and cache is None:
        # Support both [B, T_k] padding mask and [B, T_q, T_k] causal+padding mask.
        # We only apply this during prefill (no cache). During cached decode the
        # KV sequence length > query sequence length so the 2-D padding mask
        # can't broadcast cleanly across the full [B, H, G, T_q, T_k] tensor.
        if attn_mask.ndim == 2:
          mask_5d = attn_mask[:, jnp.newaxis, jnp.newaxis, jnp.newaxis, :]
        else:
          mask_5d = attn_mask[:, jnp.newaxis, jnp.newaxis, :, :]
        attn = jnp.where(mask_5d, attn, K_MASK)
      attn = jax.nn.softmax(attn.astype(jnp.float32), axis=-1).astype(
          key_proj.dtype
      )
      qkv = jnp.einsum('BHGTS,BSHD->BTHGD', attn, value_proj)
      qkv = qkv.reshape((b, t, qh, d))

    # Gated output: out * sigmoid(gate)
    qkv_flat = qkv.reshape(b, t, self.config.num_heads * self.config.head_dim)
    qkv_gated = qkv_flat * nnx.sigmoid(gate_flat)
    qkv_gated = qkv_gated.reshape(b, t, self.config.num_heads, self.config.head_dim)

    outputs = self.o_proj(qkv_gated)
    outputs = shard(outputs, self.shd_config.act_btd)

    new_cache = None
    if cache is not None:
      new_cache = {
          'v': value_proj,
          'k': key_proj,
          'end_index': cache['end_index'] + t,
      }
    return new_cache, outputs

  @jax.named_scope('attention')
  def __call__(
      self,
      x: jaxtyping.Array,
      segment_pos: jaxtyping.Array,
      cache: LayerCache | None,
      attn_mask: jaxtyping.Array | None,
  ) -> tuple[LayerCache | None, jaxtyping.Array]:
    if self.config.remat_config == RematConfig.BLOCK:
      return nnx.remat(self.block.__func__)(self, x, segment_pos, cache, attn_mask)
    return self.block(x, segment_pos, cache, attn_mask)


# ---------------------------------------------------------------------------
# Linear-attention block (Gated Delta Net)
# ---------------------------------------------------------------------------


class GatedDeltaNet(nnx.Module):
  """Gated Delta Net: causal conv1d followed by the chunked delta rule."""

  def __init__(self, config: ModelConfig, *, rngs: nnx.Rngs):
    self.config = config
    D = config.embed_dim
    self.num_v_heads = config.linear_num_value_heads
    self.num_k_heads = config.linear_num_key_heads
    self.head_k_dim = config.linear_key_head_dim
    self.head_v_dim = config.linear_value_head_dim
    self.key_dim = self.head_k_dim * self.num_k_heads
    self.value_dim = self.head_v_dim * self.num_v_heads
    self.conv_kernel_size = config.linear_conv_kernel_dim
    self.conv_dim = self.key_dim * 2 + self.value_dim

    lin = lambda out: nnx.Linear(
        D,
        out,
        use_bias=False,
        rngs=rngs,
        kernel_init=nnx.with_partitioning(
            nnx.initializers.lecun_normal(), config.shd_config.linear_weight_dd
        ),
        dtype=config.dtype,
        param_dtype=config.param_dtype,
    )

    self.in_proj_qkv = lin(self.conv_dim)
    self.in_proj_z = lin(self.value_dim)
    self.in_proj_b = lin(self.num_v_heads)   # per-head beta logits
    self.in_proj_a = lin(self.num_v_heads)   # per-head dt logits

    # conv1d weight stored as [C, 1, K] = OIH format (matches HF and JAX depthwise conv).
    self.conv1d_weight = nnx.Param(
        nnx.initializers.lecun_normal()(
            rngs.params(),
            (self.conv_dim, 1, self.conv_kernel_size),
            config.param_dtype,
        ),
    )
    # A_log and dt_bias for the gating
    self.A_log = nnx.Param(
        jnp.log(
            jax.random.uniform(
                rngs.params(),
                (self.num_v_heads,),
                dtype=config.param_dtype,
                minval=1e-3,
                maxval=16.0,
            )
        ),
    )
    self.dt_bias = nnx.Param(
        nnx.initializers.ones_init()(
            rngs.params(), (self.num_v_heads,), config.param_dtype
        ),
    )

    self.norm = Qwen3_5RMSNormGated(
        self.head_v_dim, norm_eps=config.norm_eps, rngs=rngs, dtype=config.dtype
    )
    self.out_proj = nnx.Linear(
        self.value_dim,
        D,
        use_bias=False,
        rngs=rngs,
        kernel_init=nnx.with_partitioning(
            nnx.initializers.lecun_normal(), config.shd_config.linear_weight_dd
        ),
        dtype=config.dtype,
        param_dtype=config.param_dtype,
    )

  def _causal_conv(
      self,
      x: jaxtyping.Array,       # [B, C, T]
      conv_state: jaxtyping.Array | None = None,  # [B, C, K]
  ) -> tuple[jaxtyping.Array, jaxtyping.Array]:
    """Depthwise causal conv1d."""
    # weight is already [C, 1, K] = OIH layout for JAX depthwise conv
    kernel = self.conv1d_weight.value
    seq_len = x.shape[-1]

    if conv_state is None:
      left_pad = self.conv_kernel_size - 1
      x_full = jnp.pad(x, ((0, 0), (0, 0), (left_pad, 0)))
    else:
      x_full = jnp.concatenate([conv_state, x], axis=-1)

    new_state = x_full[..., -self.conv_kernel_size:]
    out_full = jax.lax.conv_general_dilated(
        x_full,
        kernel,
        window_strides=(1,),
        padding='VALID',
        feature_group_count=self.conv_dim,
        dimension_numbers=('NCH', 'OIH', 'NCH'),
    )
    return nnx.silu(out_full[..., -seq_len:]), new_state

  @jax.named_scope('gated_delta_net')
  def __call__(
      self,
      hidden_states: jaxtyping.Array,
      *,
      attn_mask: jaxtyping.Array | None = None,
      conv_state: jaxtyping.Array | None = None,
      recurrent_state: jaxtyping.Array | None = None,
  ) -> tuple[jaxtyping.Array, jaxtyping.Array, jaxtyping.Array]:
    """Returns (output, new_conv_state, new_recurrent_state)."""
    # Mask padding tokens.  GatedDeltaNet needs a per-token [B, T] bool mask.
    # If the caller passes a 3-D causal mask [B, T_q, T_k], collapse to [B, T].
    if attn_mask is not None:
      if attn_mask.ndim == 3:
        attn_mask = attn_mask.any(axis=-1)  # [B, T_q, T_k] -> [B, T_q]
      hidden_states = hidden_states * attn_mask[..., None].astype(
          hidden_states.dtype
      )

    batch_size, seq_len, _ = hidden_states.shape

    mixed_qkv = self.in_proj_qkv(hidden_states).transpose((0, 2, 1))  # [B, C, T]
    z = self.in_proj_z(hidden_states).reshape(
        batch_size, seq_len, -1, self.head_v_dim
    )
    b_logits = self.in_proj_b(hidden_states)
    a_logits = self.in_proj_a(hidden_states)

    mixed_qkv, new_conv_state = self._causal_conv(mixed_qkv, conv_state)
    mixed_qkv = mixed_qkv.transpose((0, 2, 1))   # [B, T, C]

    q_end = self.key_dim
    k_end = self.key_dim * 2
    query = mixed_qkv[..., :q_end].reshape(
        batch_size, seq_len, -1, self.head_k_dim
    )
    key = mixed_qkv[..., q_end:k_end].reshape(
        batch_size, seq_len, -1, self.head_k_dim
    )
    value = mixed_qkv[..., k_end:].reshape(
        batch_size, seq_len, -1, self.head_v_dim
    )

    beta = nnx.sigmoid(b_logits)
    g = -jnp.exp(self.A_log.value.astype(jnp.float32)) * jax.nn.softplus(
        a_logits.astype(jnp.float32) + self.dt_bias.value.astype(jnp.float32)
    )

    # GQA-style expansion if num_v_heads != num_k_heads
    if self.num_v_heads // self.num_k_heads > 1:
      repeats = self.num_v_heads // self.num_k_heads
      query = jnp.repeat(query, repeats, axis=2)
      key = jnp.repeat(key, repeats, axis=2)

    # Chunked for prefill, recurrent for decode
    if seq_len > 1:
      core_out, new_recurrent_state = chunk_gated_delta_rule(
          query, key, value, g, beta,
          chunk_size=64,
          initial_state=recurrent_state,
      )
    else:
      core_out, new_recurrent_state = recurrent_gated_delta_rule(
          query, key, value, g, beta, recurrent_state
      )

    core_out = self.norm(core_out, z).reshape(batch_size, seq_len, -1)
    out = self.out_proj(core_out)
    return out, new_conv_state, new_recurrent_state


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------


class MLP(nnx.Module):
  """SwiGLU feed-forward network."""

  def __init__(self, config: ModelConfig, *, rngs: nnx.Rngs):
    self.config = config
    self.shd_config = config.shd_config
    self.gate_proj = nnx.Linear(
        config.embed_dim,
        config.hidden_dim,
        use_bias=False,
        rngs=rngs,
        kernel_init=nnx.with_partitioning(
            nnx.initializers.zeros_init(), self.shd_config.ffw_weight_df
        ),
        dtype=config.dtype,
        param_dtype=config.param_dtype,
    )
    self.up_proj = nnx.Linear(
        config.embed_dim,
        config.hidden_dim,
        use_bias=False,
        rngs=rngs,
        kernel_init=nnx.with_partitioning(
            nnx.initializers.zeros_init(), self.shd_config.ffw_weight_df
        ),
        dtype=config.dtype,
        param_dtype=config.param_dtype,
    )
    self.down_proj = nnx.Linear(
        config.hidden_dim,
        config.embed_dim,
        use_bias=False,
        rngs=rngs,
        kernel_init=nnx.with_partitioning(
            nnx.initializers.zeros_init(), self.shd_config.ffw_weight_fd
        ),
        dtype=config.dtype,
        param_dtype=config.param_dtype,
    )

  @jax.named_scope('feed_forward')
  def __call__(self, x: jaxtyping.Array) -> jaxtyping.Array:
    activations = nnx.silu(self.gate_proj(x)) * self.up_proj(x)
    activations = shard(activations, self.shd_config.act_btf)
    return self.down_proj(activations)


# ---------------------------------------------------------------------------
# Decoder layer (hybrid dispatch)
# ---------------------------------------------------------------------------


class DecoderLayer(nnx.Module):
  """Single decoder layer: either full-attention or Gated Delta Net."""

  def __init__(
      self, config: ModelConfig, layer_type: str, *, rngs: nnx.Rngs
  ):
    self.layer_type = layer_type
    self.input_layernorm = Qwen3_5RMSNorm(
        config.embed_dim,
        norm_eps=config.norm_eps,
        rngs=rngs,
        dtype=config.dtype,
    )
    self.post_attention_layernorm = Qwen3_5RMSNorm(
        config.embed_dim,
        norm_eps=config.norm_eps,
        rngs=rngs,
        dtype=config.dtype,
    )
    if layer_type == 'full_attention':
      self.attn = Attention(config=config, rngs=rngs)
    else:
      self.linear_attn = GatedDeltaNet(config=config, rngs=rngs)
    self.mlp = MLP(config=config, rngs=rngs)

  def __call__(
      self,
      x: jaxtyping.Array,
      segment_pos: jaxtyping.Array,
      cache: LayerCache | None,
      attn_mask: jaxtyping.Array | None,
  ) -> tuple[LayerCache | None, jaxtyping.Array]:
    inputs_normalized = self.input_layernorm(x)

    if self.layer_type == 'full_attention':
      cache, attn_output = self.attn(
          inputs_normalized, segment_pos, cache, attn_mask
      )
      new_cache = cache
    else:
      # For linear-attention layers, cache stores conv_state + recurrent_state
      conv_state = cache.get('conv_state') if cache else None
      recurrent_state = cache.get('recurrent_state') if cache else None
      attn_output, new_conv_state, new_recurrent_state = self.linear_attn(
          inputs_normalized,
          attn_mask=attn_mask if cache is None else None,
          conv_state=conv_state,
          recurrent_state=recurrent_state,
      )
      if cache is not None:
        new_cache = {
            'conv_state': new_conv_state,
            'recurrent_state': new_recurrent_state,
        }
      else:
        new_cache = None

    attn_output = attn_output + x
    residual = attn_output
    attn_output = self.post_attention_layernorm(attn_output)
    outputs = self.mlp(attn_output)
    outputs = residual + outputs
    return new_cache, outputs


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------


class Qwen3_5(BackendMappingMixin, nnx.Module):
  """Qwen3.5 hybrid language model."""

  BACKEND_PACKAGE_PATH = __name__

  def __init__(self, config: ModelConfig, *, rngs: nnx.Rngs):
    self.config = config
    self.embedder = Embedder(
        vocab_size=config.vocab_size,
        embed_dim=config.embed_dim,
        rngs=rngs,
        shd_config=config.shd_config,
        dtype=config.dtype,
        param_dtype=config.param_dtype,
    )
    self.layers = compat.ModuleList([
        DecoderLayer(config=config, layer_type=lt, rngs=rngs)
        for lt in config.layer_types
    ])
    self.final_norm = Qwen3_5RMSNorm(
        config.embed_dim,
        rngs=rngs,
        norm_eps=config.norm_eps,
        dtype=config.dtype,
    )
    if not config.use_tied_embedding:
      self.lm_head = Einsum(
          einsum_str='BTD,DV->BTV',
          shape=(config.embed_dim, config.vocab_size),
          rngs=rngs,
          sharding=config.shd_config.emb_dv,
          dtype=config.dtype,
          param_dtype=config.param_dtype,
      )

  def init_cache(
      self, batch_size: int, cache_size: int, dtype: jnp.dtype
  ) -> Cache:
    """Initialises per-layer KV / conv / recurrent caches."""
    config = self.config
    kv_shape = (batch_size, cache_size, config.num_kv_heads, config.head_dim)
    k = jnp.zeros(kv_shape, dtype=dtype)
    v = jnp.zeros(kv_shape, dtype=dtype)
    end_index = jnp.zeros((batch_size,), dtype=jnp.int32)

    conv_channels = (config.linear_num_key_heads * config.linear_key_head_dim * 2
                     + config.linear_num_value_heads * config.linear_value_head_dim)
    # conv_state layout: [B, C, K]  (NCH — channels first, kernel last)
    conv_shape = (batch_size, conv_channels, config.linear_conv_kernel_dim)
    rec_shape = (batch_size, config.linear_num_value_heads,
                 config.linear_key_head_dim, config.linear_value_head_dim)

    cache = {}
    for i, lt in enumerate(config.layer_types):
      if lt == 'full_attention':
        cache[f'layer_{i}'] = {
            'k': k, 'v': v, 'end_index': end_index
        }
      else:
        cache[f'layer_{i}'] = {
            'conv_state': jnp.zeros(conv_shape, dtype=dtype),
            'recurrent_state': jnp.zeros(rec_shape, dtype=dtype),
        }
    return cache

  def __call__(
      self,
      input_tokens: jaxtyping.Array,    # [B, L]
      positions: jaxtyping.Array,       # [B, L]
      cache: Cache | None,
      attention_mask: jaxtyping.Array,  # [B, L, L'] or [B, 1, L']
      output_hidden_states: bool = False,
  ) -> tuple[jaxtyping.Array, Cache | None]:
    """Forward pass.

    Returns:
      predicted_logits: [B, L, V]
      new_cache: updated cache dict, or None if no cache was provided.
    """
    new_cache = None if cache is None else {}
    x = self.embedder.encode(input_tokens)

    for i, layer in enumerate(self.layers):
      layer_name = f'layer_{i}'
      layer_cache = cache[layer_name] if cache else None
      layer_cache, x = layer(x, positions, layer_cache, attention_mask)
      if cache is not None:
        new_cache[layer_name] = layer_cache  # pytype: disable=container-type-mismatch

    x = self.final_norm(x)
    if output_hidden_states:
      self.sow(nnx.Intermediate, 'all_hidden_states', x)
    if self.config.use_tied_embedding:
      logits = self.embedder.decode(x)
    else:
      logits = self.lm_head(x)

    return jnp.astype(logits, jnp.float32), new_cache  # pytype: disable=bad-return-type

  def get_model_input(self):
    dummy_batch_size = 2
    dummy_seq_len = 1
    return {
        'input_tokens': jnp.ones(
            (dummy_batch_size, dummy_seq_len), dtype=jnp.int32
        ),
        'positions': jnp.ones(
            (dummy_batch_size, dummy_seq_len), dtype=jnp.int32
        ),
        'cache': None,
        'attention_mask': jnp.ones(
            (dummy_batch_size, 1, dummy_seq_len), dtype=jnp.bool_
        ),
    }
