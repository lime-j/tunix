# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Gemma 4 model (text-only and VLM) in Tunix style."""

import dataclasses
import enum
import itertools
from typing import Tuple

import einops
from flax import nnx
import jax
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel as splash
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_mask as mask_lib
from jax.experimental.shard_map import shard_map
from jax.interpreters import pxla
import jax.sharding as shd
from jax.sharding import PartitionSpec as P
from jax import numpy as jnp
import jaxtyping
from functools import partial

from tunix.generate.mappings import BackendMappingMixin
from tunix.models.gemma4 import merge_embeddings as merge_embeddings_lib
from tunix.utils import compat
from tunix.utils import env_utils
from tunix.utils import sharding_utils

env_utils.setup_sharding_environment()

LayerCache = dict[str, jaxtyping.Array]
Cache = dict[str, LayerCache]


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class AttentionType(enum.Enum):
  GLOBAL = 1
  LOCAL_SLIDING = 2


GEMMA4_ATTENTION_PATTERN = (
    AttentionType.LOCAL_SLIDING,
    AttentionType.LOCAL_SLIDING,
    AttentionType.LOCAL_SLIDING,
    AttentionType.LOCAL_SLIDING,
    AttentionType.LOCAL_SLIDING,
    AttentionType.GLOBAL,
)

K_MASK = -2.3819763e38


# ---------------------------------------------------------------------------
# Sharding config
# ---------------------------------------------------------------------------

@dataclasses.dataclass(slots=True, frozen=True)
class ShardingConfig:
  emb_vd: Tuple[str | None, ...]
  q_weight_ndh: Tuple[str | None, ...]
  kv_weight_cndh: Tuple[str | None, ...]
  o_weight_nhd: Tuple[str | None, ...]
  ffw_weight_df: Tuple[str | None, ...]
  ffw_weight_fd: Tuple[str | None, ...]
  expert_weight: Tuple[str | None, ...]
  rms_norm_weight: Tuple[str | None, ...]
  act_btd: Tuple[str | None, ...]
  act_btnh: Tuple[str | None, ...]
  vision_proj: Tuple[str | None, ...]

  @staticmethod
  def get_default_sharding(is_sampling: bool = False):
    fsdp = 'fsdp' if not is_sampling else None
    return ShardingConfig(
        emb_vd=('tp', fsdp),
        q_weight_ndh=('tp', fsdp, None),
        kv_weight_cndh=(None, 'tp', fsdp, None),
        o_weight_nhd=('tp', None, fsdp),
        ffw_weight_df=(fsdp, 'tp'),
        ffw_weight_fd=('tp', fsdp),
        expert_weight=(None, fsdp, 'tp'),
        rms_norm_weight=('tp',),
        act_btd=('fsdp', None, None if is_sampling else 'tp'),
        act_btnh=('fsdp', None, 'tp', None),
        vision_proj=(fsdp, 'tp'),
    )


# ---------------------------------------------------------------------------
# Vision config
# ---------------------------------------------------------------------------

@dataclasses.dataclass(slots=True, kw_only=True, frozen=True)
class VisionConfig:
  hidden_dim: int = 1152
  num_layers: int = 27
  num_heads: int = 16
  head_dim: int = 72
  mlp_dim: int = 4304
  patch_size: int = 16
  max_position_size: int = 10240
  rope_base_frequency: int = 100
  vision_output_length: int = 280
  image_token_id: int = 258880
  param_dtype: jnp.dtype = jnp.float32


# ---------------------------------------------------------------------------
# Model config
# ---------------------------------------------------------------------------

@dataclasses.dataclass(slots=True, kw_only=True)
class ModelConfig:
  num_layers: int
  num_embed: int
  embed_dim: int
  hidden_dim: int
  num_heads: int
  head_dim: int
  num_kv_heads: int
  global_head_dim: int = 512
  num_global_kv_heads: int = 2
  sliding_window_size: int | None = 1024
  local_base_frequency: int = 10_000
  global_base_frequency: int = 1_000_000
  local_rope_proportion: float = 1.0
  global_rope_proportion: float = 0.25
  attn_logits_soft_cap: float | None = None
  num_experts: int = 1
  expert_dim: int | None = None
  num_experts_per_tok: int = 1
  vision_config: VisionConfig | None = None
  shd_config: ShardingConfig = dataclasses.field(
      default_factory=ShardingConfig.get_default_sharding)
  param_dtype: jnp.dtype = jnp.bfloat16
  use_flash_attention: bool = False
  flash_attention_block_size: int = 1024

  @classmethod
  def gemma4_26b_pt(cls, sharding_config=None, text_only=True):
    return cls(
        num_layers=30, num_embed=262144, embed_dim=2816, hidden_dim=2112,
        num_heads=16, head_dim=256, num_kv_heads=8,
        global_head_dim=512, num_global_kv_heads=2, sliding_window_size=1024,
        num_experts=128, expert_dim=704, num_experts_per_tok=8,
        vision_config=None if text_only else VisionConfig(),
        shd_config=sharding_config or ShardingConfig.get_default_sharding(),
    )

  @classmethod
  def gemma4_26b_it(cls, **kw):
    return cls.gemma4_26b_pt(**kw)

  @classmethod
  def gemma4_31b_pt(cls, sharding_config=None, text_only=True):
    return cls(
        num_layers=60, num_embed=262144, embed_dim=5376, hidden_dim=21504,
        num_heads=32, head_dim=256, num_kv_heads=16,
        global_head_dim=512, num_global_kv_heads=4, sliding_window_size=1024,
        num_experts=1,
        vision_config=None if text_only else VisionConfig(),
        shd_config=sharding_config or ShardingConfig.get_default_sharding(),
    )

  @classmethod
  def gemma4_31b_it(cls, **kw):
    return cls.gemma4_31b_pt(**kw)


# ---------------------------------------------------------------------------
# Helpers: RoPE
# ---------------------------------------------------------------------------

@jax.named_scope('rope')
def apply_rope(
    inputs: jaxtyping.Array,
    positions: jaxtyping.Array,
    *,
    head_dim: int,
    base_frequency: int,
    partial_rotary_factor: float = 1.0,
) -> jaxtyping.Array:
  rotary_dim = head_dim - (head_dim - int(head_dim * partial_rotary_factor)) // 2 * 2
  rotary_dim = int(head_dim * partial_rotary_factor)
  rotary_dim = rotary_dim - (rotary_dim % 2)
  if rotary_dim == 0:
    return inputs
  x_rot = inputs[..., :rotary_dim]
  x_pass = inputs[..., rotary_dim:]
  fraction = 2 * jnp.arange(0, rotary_dim // 2) / rotary_dim
  timescale = base_frequency ** fraction
  sinusoid = (positions[..., jnp.newaxis] / timescale)[..., jnp.newaxis, :]
  sin, cos = jnp.sin(sinusoid), jnp.cos(sinusoid)
  first_half, second_half = jnp.split(x_rot, 2, axis=-1)
  x_rotated = jnp.concatenate(
      [first_half * cos - second_half * sin,
       second_half * cos + first_half * sin], axis=-1).astype(inputs.dtype)
  if rotary_dim == head_dim:
    return x_rotated
  return jnp.concatenate([x_rotated, x_pass], axis=-1)


def _apply_rope_1d(x, pos, base_freq):
  dim = x.shape[-1]
  half = dim // 2
  fraction = 2 * jnp.arange(0, half) / dim
  ts = base_freq ** fraction
  inp = (pos[..., jnp.newaxis, jnp.newaxis] / ts)
  sin = jnp.concatenate([jnp.sin(inp)] * 2, axis=-1).astype(x.dtype)
  cos = jnp.concatenate([jnp.cos(inp)] * 2, axis=-1).astype(x.dtype)
  x1, x2 = jnp.split(x, 2, axis=-1)
  return (x * cos) + (jnp.concatenate([-x2, x1], axis=-1) * sin)


def apply_multidimensional_rope(inputs, positions, *, base_frequency):
  """2D factorized RoPE. inputs: [B,L,N,H], positions: [B,L,2]."""
  ndim = positions.shape[-1]
  h = inputs.shape[-1]
  ch = 2 * (h // (2 * ndim))
  split_pts = [(k + 1) * ch for k in range(ndim - 1)]
  parts = jnp.split(inputs, split_pts, axis=-1)
  return jnp.concatenate(
      [_apply_rope_1d(parts[k], positions[..., k], base_frequency)
       for k in range(ndim)], axis=-1)


# ---------------------------------------------------------------------------
# Helpers: vision patchification & position embedding
# ---------------------------------------------------------------------------

def patchify(images, patch_size):
  *b, h, w, c = images.shape
  p = patch_size
  reshaped = jax.lax.reshape(images, tuple(b) + (h // p, p, w // p, p, c))
  nb = len(b)
  transposed = jnp.transpose(
      reshaped,
      axes=tuple(range(nb)) + (nb, nb+2, nb+1, nb+3, nb+4))
  patches = jax.lax.reshape(transposed, tuple(b) + ((h // p) * (w // p), p * p * c))
  xy = jnp.meshgrid(jnp.arange(w // p), jnp.arange(h // p))
  positions_xy = jnp.stack(xy, axis=-1).reshape(-1, 2)
  return patches, jnp.broadcast_to(positions_xy, tuple(b) + positions_xy.shape)


def factorized_posemb(posemb, positions_xy, precision=None):
  """posemb: [max_pos, 2, d], positions_xy: [B, L, 2] → [B, L, d]."""
  one_hot = jax.nn.one_hot(positions_xy, posemb.shape[0], dtype=posemb.dtype)
  # one_hot: [B, L, 2, max_pos]
  pe = jnp.einsum('...is,sid->i...d', one_hot, posemb, precision=precision)
  return jnp.sum(pe, axis=0).astype(posemb.dtype)


# ---------------------------------------------------------------------------
# RMSNorm (Gemma-style: scale initialised at 0, applied as 1+w)
# ---------------------------------------------------------------------------

class RMSNorm(nnx.Module):
  def __init__(self, dim, *, rngs, epsilon=1e-6, with_scale=True,
               sharding=(), param_dtype=jnp.bfloat16):
    self.epsilon = epsilon
    self.with_scale = with_scale
    if with_scale:
      self.scale = nnx.Param(
          nnx.initializers.zeros_init()(rngs.params(), dim).astype(param_dtype),
          sharding=sharding)

  def __call__(self, x):
    var = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
    normed = x * jax.lax.rsqrt(var + self.epsilon)
    if self.with_scale:
      s = jnp.expand_dims(self.scale.value, axis=range(len(x.shape) - 1))
      return normed * (1 + s)
    return normed


# ---------------------------------------------------------------------------
# Vision modules
# ---------------------------------------------------------------------------

class VisionEntry(nnx.Module):
  def __init__(self, d_model, patch_size, max_position_size, *, rngs, param_dtype=jnp.float32):
    self.patch_size = patch_size
    self.input_projection = nnx.Linear(
        patch_size * patch_size * 3, d_model, use_bias=False,
        param_dtype=param_dtype, rngs=rngs)
    self.pos_emb_param = nnx.Param(
        nnx.initializers.normal(stddev=0.02)(
            rngs.params(), (max_position_size, 2, d_model), jnp.float32))

  def __call__(self, images_or_patches, positions_xy=None):
    if positions_xy is None:
      patches, positions_xy = patchify(images_or_patches, self.patch_size)
    else:
      patches = images_or_patches
    patches = 2.0 * patches - 1.0
    x = self.input_projection(patches)
    pos = factorized_posemb(self.pos_emb_param.value, positions_xy).astype(x.dtype)
    return x + pos, positions_xy


class VisionExit(nnx.Module):
  def __init__(self, d_model, output_length):
    self.d_model = d_model
    self.output_length = output_length

  def __call__(self, x):
    x = x * jnp.sqrt(self.d_model)
    cur = x.shape[1]
    if cur == self.output_length:
      return x
    w = int(cur ** 0.5)
    ow = int(self.output_length ** 0.5)
    x = einops.rearrange(x, 'b (h w) d -> b h w d', h=w, w=w)
    win = w // ow
    x = nnx.avg_pool(x, window_shape=(win, win), strides=(win, win))
    return einops.rearrange(x, 'b h w d -> b (h w) d')


class VisionAttention(nnx.Module):
  def __init__(self, hidden_dim, num_heads, head_dim, rope_base_frequency, *,
               rngs, param_dtype=jnp.float32):
    self.num_heads = num_heads
    self.head_dim = head_dim
    self.rope_base = rope_base_frequency
    kw = dict(use_bias=False, param_dtype=param_dtype, rngs=rngs)
    self.q_proj = nnx.Linear(hidden_dim, num_heads * head_dim, **kw)
    self.k_proj = nnx.Linear(hidden_dim, num_heads * head_dim, **kw)
    self.v_proj = nnx.Linear(hidden_dim, num_heads * head_dim, **kw)
    self.o_proj = nnx.Linear(num_heads * head_dim, hidden_dim, **kw)
    norm_kw = dict(rngs=rngs, param_dtype=param_dtype)
    self.q_norm = RMSNorm(head_dim, **norm_kw)
    self.k_norm = RMSNorm(head_dim, **norm_kw)
    self.v_norm = RMSNorm(head_dim, **norm_kw)

  def __call__(self, x, positions_xy=None):
    B, L, _ = x.shape
    N, H = self.num_heads, self.head_dim
    q = self.q_norm(self.q_proj(x).reshape(B, L, N, H))
    k = self.k_norm(self.k_proj(x).reshape(B, L, N, H))
    v = self.v_norm(self.v_proj(x).reshape(B, L, N, H))
    if positions_xy is not None:
      q = apply_multidimensional_rope(q, positions_xy, base_frequency=self.rope_base)
      k = apply_multidimensional_rope(k, positions_xy, base_frequency=self.rope_base)
    logits = jnp.einsum('BTNH,BSNH->BNTS', q, k) * (H ** -0.5)
    probs = jax.nn.softmax(logits, axis=-1).astype(k.dtype)
    out = jnp.einsum('BNTS,BSNH->BTNH', probs, v).reshape(B, L, N * H)
    return self.o_proj(out)


class VisionMlp(nnx.Module):
  def __init__(self, in_features, intermediate_dim, *, rngs, param_dtype=jnp.float32):
    kw = dict(use_bias=False, param_dtype=param_dtype, rngs=rngs)
    self.gate = nnx.Linear(in_features, intermediate_dim, **kw)
    self.up = nnx.Linear(in_features, intermediate_dim, **kw)
    self.down = nnx.Linear(intermediate_dim, in_features, **kw)

  def __call__(self, x):
    return self.down(jax.nn.gelu(self.gate(x), approximate=True) * self.up(x))


class Gemma4EncoderBlock(nnx.Module):
  def __init__(self, config: VisionConfig, *, rngs):
    d, p = config.hidden_dim, config.param_dtype
    nkw = dict(rngs=rngs, param_dtype=p)
    self.pre_attention_norm = RMSNorm(d, **nkw)
    self.post_attention_norm = RMSNorm(d, **nkw)
    self.attention = VisionAttention(
        d, config.num_heads, config.head_dim, config.rope_base_frequency,
        rngs=rngs, param_dtype=p)
    self.pre_ffw_norm = RMSNorm(d, **nkw)
    self.post_ffw_norm = RMSNorm(d, **nkw)
    self.mlp = VisionMlp(d, config.mlp_dim, rngs=rngs, param_dtype=p)

  def __call__(self, x, positions_xy=None):
    x = x + self.post_attention_norm(self.attention(self.pre_attention_norm(x), positions_xy))
    return x + self.post_ffw_norm(self.mlp(self.pre_ffw_norm(x)))


class Gemma4VisionEncoderLayer(nnx.Module):
  def __init__(self, config: VisionConfig, *, rngs):
    self.config = config
    self.vision_entry = VisionEntry(
        config.hidden_dim, config.patch_size, config.max_position_size,
        rngs=rngs, param_dtype=config.param_dtype)
    for i in range(config.num_layers):
      setattr(self, f'layer_{i}', Gemma4EncoderBlock(config, rngs=rngs))
    self.vision_exit = VisionExit(config.hidden_dim, config.vision_output_length)
    self.std_bias = nnx.Param(jnp.zeros((config.hidden_dim,), dtype=config.param_dtype))
    self.std_scale = nnx.Param(jnp.ones((config.hidden_dim,), dtype=config.param_dtype))

  def __call__(self, inputs):
    if inputs.ndim == 4:
      inputs = jnp.expand_dims(inputs, 1)
    b, n, h, w, c = inputs.shape
    flat = inputs.reshape(b * n, h, w, c)
    x, pos_xy = self.vision_entry(flat)
    for i in range(self.config.num_layers):
      x = getattr(self, f'layer_{i}')(x, pos_xy)
    x = self.vision_exit(x)
    x = (x - self.std_bias.value.astype(x.dtype)) * self.std_scale.value.astype(x.dtype)
    return x.reshape(b, n, x.shape[1], x.shape[2])


class Gemma4VisionProjector(nnx.Module):
  def __init__(self, vision_dim, text_dim, *, rngs, param_dtype=jnp.bfloat16):
    self.norm = RMSNorm(vision_dim, rngs=rngs, with_scale=False, param_dtype=param_dtype)
    self.projection = nnx.Linear(vision_dim, text_dim, use_bias=False,
                                 param_dtype=param_dtype, rngs=rngs)

  def __call__(self, x):
    return self.projection(self.norm(x))


# ---------------------------------------------------------------------------
# Shared text Einsum
# ---------------------------------------------------------------------------

class Einsum(nnx.Module):
  def __init__(self, einsum_str, shape, *, rngs, sharding=(), param_dtype=jnp.bfloat16):
    self.einsum_str = einsum_str
    self.w = nnx.Param(
        nnx.initializers.normal(dtype=param_dtype)(rngs.params(), shape),
        sharding=sharding)

  def __call__(self, x):
    return jnp.einsum(self.einsum_str, x, self.w.value)


# ---------------------------------------------------------------------------
# Text FeedForward (dense; used for 31B)
# ---------------------------------------------------------------------------

class FeedForward(nnx.Module):
  def __init__(self, config: ModelConfig, *, rngs):
    p, shd = config.param_dtype, config.shd_config
    self.gate_proj = nnx.Linear(
        config.embed_dim, config.hidden_dim, use_bias=False, param_dtype=p,
        kernel_init=nnx.with_partitioning(nnx.initializers.zeros_init(), shd.ffw_weight_df),
        rngs=rngs)
    self.up_proj = nnx.Linear(
        config.embed_dim, config.hidden_dim, use_bias=False, param_dtype=p,
        kernel_init=nnx.with_partitioning(nnx.initializers.zeros_init(), shd.ffw_weight_df),
        rngs=rngs)
    self.down_proj = nnx.Linear(
        config.hidden_dim, config.embed_dim, use_bias=False, param_dtype=p,
        kernel_init=nnx.with_partitioning(nnx.initializers.zeros_init(), shd.ffw_weight_fd),
        rngs=rngs)

  @jax.named_scope('feed_forward')
  def __call__(self, x):
    return self.down_proj(jax.nn.gelu(self.gate_proj(x)) * self.up_proj(x))


# ---------------------------------------------------------------------------
# MoE block (used for 26B)
# ---------------------------------------------------------------------------

class Gemma4RoutedExperts(nnx.Module):
  """Top-k routed expert stack."""

  def __init__(self, num_experts, embed_dim, expert_dim, num_experts_per_tok,
               *, rngs, param_dtype=jnp.bfloat16):
    self.num_experts = num_experts
    self.num_experts_per_tok = num_experts_per_tok

    init = nnx.initializers.normal(dtype=param_dtype)
    # Gate (router) kernel: embed_dim → num_experts
    self.gate_kernel = nnx.Param(init(rngs.params(), (embed_dim, num_experts)))
    # Per-expert output scale
    self.per_expert_scale = nnx.Param(jnp.ones((num_experts,), dtype=param_dtype))
    # Expert weights (JAX convention: in_features first)
    self.gate_proj_w = nnx.Param(init(rngs.params(), (num_experts, embed_dim, expert_dim)))
    self.up_proj_w   = nnx.Param(init(rngs.params(), (num_experts, embed_dim, expert_dim)))
    self.down_proj_w = nnx.Param(init(rngs.params(), (num_experts, expert_dim, embed_dim)))

  def __call__(self, x, gate_inputs):
    B, T, D = x.shape
    x_flat = x.reshape(B * T, D)
    g_flat = gate_inputs.reshape(B * T, D)

    # Top-k routing with sigmoid weights
    logits = jnp.dot(g_flat, self.gate_kernel.value)          # (BT, E)
    topk_logits, topk_idx = jax.lax.top_k(logits, self.num_experts_per_tok)
    weights = jax.nn.sigmoid(topk_logits)                      # (BT, k)

    # Dense expert compute for all experts
    gate_out = jnp.einsum('te,ned->tnd', x_flat, self.gate_proj_w.value)
    up_out   = jnp.einsum('te,ned->tnd', x_flat, self.up_proj_w.value)
    act = jax.nn.gelu(gate_out) * up_out                        # (BT, E, expert_dim)
    expert_out = jnp.einsum('tnd,nde->tne', act, self.down_proj_w.value)  # (BT, E, D)
    expert_out = expert_out * self.per_expert_scale.value[None, :, None]

    # Select top-k and combine
    mask = jax.nn.one_hot(topk_idx, self.num_experts)           # (BT, k, E)
    selected = jnp.einsum('tkn,tne->tke', mask, expert_out)     # (BT, k, D)
    out = jnp.einsum('tk,tke->te', weights, selected)           # (BT, D)
    return out.reshape(B, T, D)


class Gemma4MoEBlock(nnx.Module):
  """Shared expert + routed experts + Gemma-4 normalization scheme."""

  def __init__(self, config: ModelConfig, *, rngs):
    p, shd = config.param_dtype, config.shd_config
    kw = dict(use_bias=False, param_dtype=p)
    part_df = nnx.with_partitioning(nnx.initializers.zeros_init(), shd.ffw_weight_df)
    part_fd = nnx.with_partitioning(nnx.initializers.zeros_init(), shd.ffw_weight_fd)

    # Shared expert
    self.shared_gate = nnx.Linear(config.embed_dim, config.hidden_dim,
                                  kernel_init=part_df, **kw, rngs=rngs)
    self.shared_up   = nnx.Linear(config.embed_dim, config.hidden_dim,
                                  kernel_init=part_df, **kw, rngs=rngs)
    self.shared_down = nnx.Linear(config.hidden_dim, config.embed_dim,
                                  kernel_init=part_fd, **kw, rngs=rngs)

    # Norms
    nm = dict(rngs=rngs, sharding=shd.rms_norm_weight, param_dtype=p)
    self.pre_feedforward_layernorm_2  = RMSNorm(config.embed_dim, **nm)
    self.post_feedforward_layernorm_1 = RMSNorm(config.embed_dim, **nm)
    self.post_feedforward_layernorm_2 = RMSNorm(config.embed_dim, **nm)
    self.gate_norm = RMSNorm(config.embed_dim, rngs=rngs, with_scale=False, param_dtype=p)

    # Router scale (learnable per-dim vector, shape: [embed_dim])
    self.pre_forward_scale_2 = nnx.Param(jnp.ones((config.embed_dim,), dtype=p))

    # Routed experts
    self.routed_experts = Gemma4RoutedExperts(
        config.num_experts, config.embed_dim, config.expert_dim,
        config.num_experts_per_tok, rngs=rngs, param_dtype=p)
    self.embed_dim = config.embed_dim

  @jax.named_scope('moe_block')
  def __call__(self, inputs, original_inputs):
    # Shared expert
    shared = self.post_feedforward_layernorm_1(
        self.shared_down(jax.nn.gelu(self.shared_gate(inputs)) * self.shared_up(inputs)))

    # Routed experts
    routed_in = self.pre_feedforward_layernorm_2(original_inputs)
    gate_in = (self.gate_norm(original_inputs)
               * jnp.asarray(self.embed_dim ** -0.5, inputs.dtype)
               * self.pre_forward_scale_2.value.astype(inputs.dtype))
    routed = self.post_feedforward_layernorm_2(self.routed_experts(routed_in, gate_in))

    return routed + shared


# ---------------------------------------------------------------------------
# Text Attention
# ---------------------------------------------------------------------------

class Attention(nnx.Module):
  def __init__(self, *, num_heads, num_kv_heads, embed_dim, head_dim,
               attn_type, rope_base_frequency, rope_proportion,
               sliding_window_size, attn_logits_soft_cap,
               shd_config, rngs, param_dtype=jnp.bfloat16):
    self.num_heads = num_heads
    self.num_kv_heads = num_kv_heads
    self.head_dim = head_dim
    self.attn_type = attn_type
    self.rope_base = rope_base_frequency
    self.rope_prop = rope_proportion
    self.sliding_window_size = sliding_window_size
    self.soft_cap = attn_logits_soft_cap

    self.q_einsum = Einsum('BTD,NDH->BTNH', (num_heads, embed_dim, head_dim),
                           rngs=rngs, sharding=shd_config.q_weight_ndh, param_dtype=param_dtype)
    self.kv_einsum = Einsum('BSD,CKDH->CBSKH', (2, num_kv_heads, embed_dim, head_dim),
                            rngs=rngs, sharding=shd_config.kv_weight_cndh, param_dtype=param_dtype)
    self.attn_vec_einsum = Einsum('BTNH,NHD->BTD', (num_heads, head_dim, embed_dim),
                                  rngs=rngs, sharding=shd_config.o_weight_nhd, param_dtype=param_dtype)
    self._query_norm = RMSNorm(head_dim, rngs=rngs, param_dtype=param_dtype)
    self._key_norm   = RMSNorm(head_dim, rngs=rngs, param_dtype=param_dtype)
    self._value_norm = RMSNorm(head_dim, rngs=rngs, param_dtype=param_dtype)

  @jax.named_scope('attention')
  def __call__(self, x, segment_pos, cache, attn_mask):
    query = self._query_norm(self.q_einsum(x))
    key_proj, value_proj = self.kv_einsum(x)
    key_proj   = self._key_norm(key_proj)
    value_proj = self._value_norm(value_proj)

    query = apply_rope(query, segment_pos, head_dim=self.head_dim,
                       base_frequency=self.rope_base, partial_rotary_factor=self.rope_prop)
    key_proj = apply_rope(key_proj, segment_pos, head_dim=self.head_dim,
                          base_frequency=self.rope_base, partial_rotary_factor=self.rope_prop)

    seq_len = x.shape[1]
    if cache is not None:
      end = cache['end_index'][0]
      sl = (0, end % cache['v'].shape[1], 0, 0)
      value_proj = jax.lax.dynamic_update_slice(cache['v'], value_proj, sl)
      key_proj   = jax.lax.dynamic_update_slice(cache['k'], key_proj, sl)

    use_gqa = (self.num_kv_heads != self.num_heads) and (self.num_kv_heads > 1)
    scale = self.head_dim ** -0.5
    if use_gqa:
      b, t, kg, h = query.shape
      q_s = query.reshape(b, t, self.num_kv_heads, kg // self.num_kv_heads, h) * scale
      logits = jnp.einsum('BTKGH,BSKH->BTKGS', q_s, key_proj).reshape(b, t, kg, -1)
    else:
      logits = jnp.einsum('BTNH,BSNH->BTNS', query * scale, key_proj)

    if self.soft_cap is not None:
      logits = jnp.tanh(logits / self.soft_cap) * self.soft_cap

    if self.attn_type == AttentionType.LOCAL_SLIDING and self.sliding_window_size:
      s = key_proj.shape[1]
      idx = jnp.arange(s)
      sw = (idx[:, None] - idx[None, :]) < self.sliding_window_size
      sw = sw & (idx[:, None] >= idx[None, :])
      attn_mask = attn_mask & sw[None, None, :, :]

    padded = jnp.where(jnp.expand_dims(attn_mask, -2), logits, K_MASK)
    probs = jax.nn.softmax(padded, axis=-1).astype(key_proj.dtype)

    if use_gqa:
      b, t, kg, s = probs.shape
      p_s = probs.reshape(b, t, self.num_kv_heads, kg // self.num_kv_heads, s)
      encoded = jnp.einsum('BTKGS,BSKH->BTKGH', p_s, value_proj)
      b, t, k, g, h = encoded.shape
      encoded = encoded.reshape(b, t, k * g, h)
    else:
      encoded = jnp.einsum('BTNS,BSNH->BTNH', probs, value_proj)

    out = self.attn_vec_einsum(encoded)
    new_cache = (None if cache is None else
                 {'v': value_proj, 'k': key_proj, 'end_index': cache['end_index'] + seq_len})
    return new_cache, out


# ---------------------------------------------------------------------------
# Decoder layer
# ---------------------------------------------------------------------------

class DecoderLayer(nnx.Module):
  def __init__(self, config: ModelConfig, attn_type: AttentionType, *, rngs):
    p, shd = config.param_dtype, config.shd_config
    self.num_experts = config.num_experts

    head_dim  = config.global_head_dim   if attn_type == AttentionType.GLOBAL else config.head_dim
    num_kv    = config.num_global_kv_heads if attn_type == AttentionType.GLOBAL else config.num_kv_heads
    rope_freq = config.global_base_frequency if attn_type == AttentionType.GLOBAL else config.local_base_frequency
    rope_prop = config.global_rope_proportion if attn_type == AttentionType.GLOBAL else config.local_rope_proportion

    nm = dict(rngs=rngs, sharding=shd.rms_norm_weight, param_dtype=p)
    self.pre_attention_norm   = RMSNorm(config.embed_dim, **nm)
    self.post_attention_norm  = RMSNorm(config.embed_dim, **nm)
    self.pre_ffw_norm         = RMSNorm(config.embed_dim, **nm)
    self.post_ffw_norm        = RMSNorm(config.embed_dim, **nm)

    self.attn = Attention(
        num_heads=config.num_heads, num_kv_heads=num_kv,
        embed_dim=config.embed_dim, head_dim=head_dim,
        attn_type=attn_type, rope_base_frequency=rope_freq, rope_proportion=rope_prop,
        sliding_window_size=config.sliding_window_size,
        attn_logits_soft_cap=config.attn_logits_soft_cap,
        shd_config=shd, rngs=rngs, param_dtype=p)

    self.mlp = (Gemma4MoEBlock(config, rngs=rngs)
                if config.num_experts > 1
                else FeedForward(config, rngs=rngs))

    self.layer_scalar = nnx.Param(jnp.ones((1,), dtype=p))

  def __call__(self, x, segment_pos, cache, attn_mask):
    cache, attn_out = self.attn(self.pre_attention_norm(x), segment_pos, cache, attn_mask)
    x = x + self.post_attention_norm(attn_out)
    ffw_in = self.pre_ffw_norm(x)
    mlp_out = (self.mlp(ffw_in, original_inputs=x)
               if self.num_experts > 1 else self.mlp(ffw_in))
    x = x + self.post_ffw_norm(mlp_out)
    return cache, x * self.layer_scalar.value


# ---------------------------------------------------------------------------
# Embedder
# ---------------------------------------------------------------------------

class Embedder(nnx.Module):
  def __init__(self, vocab_size, embed_dim, *, rngs, shd_config, param_dtype=jnp.bfloat16):
    self.input_embedding = nnx.Param(
        nnx.initializers.normal(dtype=param_dtype)(rngs.params(), (vocab_size, embed_dim)),
        sharding=shd_config.emb_vd)
    self.shd_config = shd_config

  @jax.named_scope('embedder_encode')
  def encode(self, x):
    x = self.input_embedding[(x,)]
    x = x * jnp.sqrt(x.shape[-1]).astype(x.dtype)
    return sharding_utils.shard(x, self.shd_config.act_btd)

  @jax.named_scope('embedder_decode')
  def decode(self, x):
    return jnp.dot(x, self.input_embedding.value.T)

  @property
  def embed_dim(self):
    return self.input_embedding.value.shape[1]

  @property
  def num_embed(self):
    return self.input_embedding.value.shape[0]


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------

class Gemma4(BackendMappingMixin, nnx.Module):
  """Gemma 4 transformer."""

  BACKEND_PACKAGE_PATH = __name__

  def __init__(self, config: ModelConfig, *, rngs: nnx.Rngs):
    self.config = config
    self.embedder = Embedder(
        config.num_embed, config.embed_dim,
        rngs=rngs, shd_config=config.shd_config, param_dtype=config.param_dtype)
    self.layers = compat.ModuleList([
        DecoderLayer(config=config, attn_type=at, rngs=rngs)
        for _, at in zip(range(config.num_layers), itertools.cycle(GEMMA4_ATTENTION_PATTERN))
    ])
    self.final_norm = RMSNorm(
        config.embed_dim, rngs=rngs,
        sharding=config.shd_config.rms_norm_weight, param_dtype=config.param_dtype)

    if config.vision_config is not None:
      self.vision_encoder = Gemma4VisionEncoderLayer(config.vision_config, rngs=rngs)
      self.vision_projector = Gemma4VisionProjector(
          config.vision_config.hidden_dim, config.embed_dim,
          rngs=rngs, param_dtype=config.param_dtype)
    else:
      self.vision_encoder = None
      self.vision_projector = None

  def __call__(
      self,
      last_tokens: jaxtyping.Array,
      positions: jaxtyping.Array | None = None,
      cache: Cache | None = None,
      attention_mask: jaxtyping.Array | None = None,
      output_hidden_states: bool = False,
      *,
      images: jaxtyping.Array | None = None,
  ):
    new_cache = None if cache is None else {}
    x = self._encode_and_get_inputs(tokens=last_tokens, images=images)

    for i, layer in enumerate(self.layers):
      name = f'layer_{i}'
      lc = cache[name] if cache else None
      with jax.named_scope(name):
        lc, x = layer(x, positions, lc, attention_mask)
      if cache is not None:
        new_cache[name] = lc

    x = self.final_norm(x)
    if output_hidden_states:
      self.sow(nnx.Intermediate, 'all_hidden_states', x)
    return self.embedder.decode(x), new_cache

  def _encode_and_get_inputs(self, *, tokens, images=None):
    x = self.embedder.encode(tokens)
    if images is not None and self.vision_encoder is not None:
      if images.ndim == 4:
        images = einops.rearrange(images, 'b h w c -> b 1 h w c')
      soft = self.vision_encoder(images)          # (B, N, L_vis, d_vis)
      soft = self.vision_projector(soft)          # (B, N, L_vis, embed_dim)
      image_token_id = self.config.vision_config.image_token_id
      mask = (tokens == image_token_id)
      x = merge_embeddings_lib.merge_embeddings(
          text_embeddings=x, vision_embeddings=soft, mask=mask)
    return x

  def get_attention_mask(self, tokens, *, inputs_mask=None):
    if inputs_mask is None:
      inputs_mask = tokens != 0
    seq_len = tokens.shape[-1]
    causal = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))
    return inputs_mask[:, None, :] * causal[None, ...]

  def get_model_input(self):
    b, l = 2, 1
    inp = {
        'last_tokens': jnp.ones((b, l), dtype=jnp.int32),
        'positions':   jnp.ones((b, l), dtype=jnp.int32),
        'cache': None,
        'attention_mask': jnp.ones((b, 1, l), dtype=jnp.bool_),
    }
    if self.vision_encoder is not None:
      inp['images'] = jnp.ones((b, 1, 896, 896, 3), dtype=jnp.float32)
    return inp

  @property
  def embed_dim(self):
    return self.embedder.embed_dim

  @property
  def num_embed(self):
    return self.embedder.num_embed

  @property
  def num_layers(self):
    return len(self.layers)
