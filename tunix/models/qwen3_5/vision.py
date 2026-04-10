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

"""Qwen3.5 SigLIP-style vision encoder.

Architecture (0.8B config):
  - VisionPatchEmbed   : Conv3d(3, 768, [2,16,16]) rewritten as einsum
  - pos_embed          : Embedding(2304, 768) — learned 2D spatial positions
  - 12× VisionBlock    : LayerNorm → full (non-causal) Attention + 2D RoPE
                         LayerNorm → MLP (GELU)
  - VisionPatchMerger  : spatial_merge_size² fold → MLP → out_hidden_size=1024

Input convention (same as HF):
  pixel_patches : [total_patches, C * temporal_patch_size * patch_h * patch_w]
                  pre-packed flat patches, one row per spatiotemporal patch.
  grid_thw      : [N_images, 3]  each row = (T, H_grids, W_grids).
                  T * H_grids * W_grids == number of patches for that image.

Output:
  merged_tokens : [total_patches // spatial_merge_size², out_hidden_size]
                  ready to be spliced into the LLM token sequence.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Tuple

import jax
import jax.numpy as jnp
import jaxtyping
from flax import nnx


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclasses.dataclass(slots=True)
class VisionConfig:
  """Configuration for Qwen3.5 vision encoder."""

  depth: int = 12
  hidden_size: int = 768
  num_heads: int = 12
  intermediate_size: int = 3072
  # input patch geometry
  patch_size: int = 16
  temporal_patch_size: int = 2
  in_channels: int = 3
  # positional embedding table size (sqrt must be integer → max grid side)
  num_position_embeddings: int = 2304  # 48 * 48
  # spatial downsampling factor in PatchMerger
  spatial_merge_size: int = 2
  # output dimension (must match LLM embed_dim)
  out_hidden_size: int = 1024
  norm_eps: float = 1e-6
  # dtype for computation
  dtype: jnp.dtype = jnp.float32
  param_dtype: jnp.dtype = jnp.float32

  @classmethod
  def qwen3_5_0p8b(cls) -> 'VisionConfig':
    """Matches vision_config from Qwen/Qwen3.5-0.8B config.json."""
    return cls(
        depth=12,
        hidden_size=768,
        num_heads=12,
        intermediate_size=3072,
        patch_size=16,
        temporal_patch_size=2,
        in_channels=3,
        num_position_embeddings=2304,
        spatial_merge_size=2,
        out_hidden_size=1024,
    )


# ---------------------------------------------------------------------------
# 2-D rotary positional embedding for the vision encoder
# ---------------------------------------------------------------------------


class VisionRotaryEmbedding(nnx.Module):
  """Per-head 2-D RoPE used inside vision attention.

  Computes frequency table: ``freqs[pos, i] = pos / theta^(2i / dim)``.
  When called with ``seqlen``, returns a table of shape ``[seqlen, dim//2]``.
  At call time we look up (row, col) grid coordinates and concatenate the
  corresponding row/col freq vectors before applying the standard RoPE rotation.
  """

  def __init__(self, dim: int, theta: float = 10000.0):
    self.dim = dim
    self.theta = theta
    # inv_freq is a non-trained buffer; store as a plain array attribute.
    inv_freq = 1.0 / (
        theta ** (jnp.arange(0, dim, 2, dtype=jnp.float32) / dim)
    )
    self.inv_freq = inv_freq  # [dim // 2]

  def __call__(self, seqlen: int) -> jaxtyping.Array:
    """Returns ``[seqlen, dim // 2]`` frequencies."""
    seq = jnp.arange(seqlen, dtype=jnp.float32)
    freqs = jnp.outer(seq, self.inv_freq)  # [seqlen, dim // 2]
    return freqs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _gelu_tanh(x: jaxtyping.Array) -> jaxtyping.Array:
  """GELU with tanh approximation (matches HF hidden_act='gelu_pytorch_tanh')."""
  return jax.nn.gelu(x, approximate=True)


def _apply_rotary_pos_emb_vision(
    q: jaxtyping.Array,  # [seq, num_heads, head_dim]
    k: jaxtyping.Array,  # [seq, num_heads, head_dim]
    cos: jaxtyping.Array,  # [seq, head_dim]
    sin: jaxtyping.Array,  # [seq, head_dim]
) -> Tuple[jaxtyping.Array, jaxtyping.Array]:
  """Applies RoPE to q and k.  Matches HF apply_rotary_pos_emb_vision.

  HF rotate_half: x = [x1 | x2]  →  [-x2 | x1]
  """
  # unsqueeze head dim: [seq, 1, head_dim]
  cos = cos[:, None, :]
  sin = sin[:, None, :]
  # rotate_half: split last dim in half, negate second half then swap
  half = q.shape[-1] // 2
  def rotate_half(x):
    x1, x2 = x[..., :half], x[..., half:]
    return jnp.concatenate([-x2, x1], axis=-1)
  q_embed = (q * cos + rotate_half(q) * sin).astype(q.dtype)
  k_embed = (k * cos + rotate_half(k) * sin).astype(k.dtype)
  return q_embed, k_embed


# ---------------------------------------------------------------------------
# Sub-modules
# ---------------------------------------------------------------------------


class VisionPatchEmbed(nnx.Module):
  """3-D patch embedding (matches HF Conv3d with stride == kernel).

  HF stores the weight as ``[out_ch, in_ch, T, H, W]`` = ``[768, 3, 2, 16, 16]``.
  We reimplement the Conv3d as a linear projection over the flattened kernel
  volume, which is equivalent for stride == kernel (non-overlapping patches).

  Input: ``[N, C * T * H * W]``  (pre-flattened by the caller)
  Output: ``[N, embed_dim]``
  """

  def __init__(self, config: VisionConfig, *, rngs: nnx.Rngs):
    self.embed_dim = config.hidden_size
    patch_dim = (
        config.in_channels
        * config.temporal_patch_size
        * config.patch_size
        * config.patch_size
    )
    self.proj = nnx.Linear(
        patch_dim,
        config.hidden_size,
        use_bias=True,
        rngs=rngs,
        dtype=config.dtype,
        param_dtype=config.param_dtype,
    )

  def __call__(self, hidden_states: jaxtyping.Array) -> jaxtyping.Array:
    """``[N, patch_dim]`` → ``[N, embed_dim]``."""
    return self.proj(hidden_states.astype(self.proj.dtype))


class VisionMLP(nnx.Module):
  """Vision encoder feed-forward block (GELU, biases enabled)."""

  def __init__(self, config: VisionConfig, *, rngs: nnx.Rngs):
    self.fc1 = nnx.Linear(
        config.hidden_size, config.intermediate_size,
        use_bias=True, rngs=rngs,
        dtype=config.dtype, param_dtype=config.param_dtype,
    )
    self.fc2 = nnx.Linear(
        config.intermediate_size, config.hidden_size,
        use_bias=True, rngs=rngs,
        dtype=config.dtype, param_dtype=config.param_dtype,
    )

  def __call__(self, x: jaxtyping.Array) -> jaxtyping.Array:
    return self.fc2(_gelu_tanh(self.fc1(x)))


class VisionAttention(nnx.Module):
  """Full (non-causal) multi-head attention with 2-D RoPE.

  Processes all patches of one or more images as a flat sequence; no causal
  mask is applied.  For batched images each is treated independently (the
  caller passes per-image segments).
  """

  def __init__(self, config: VisionConfig, *, rngs: nnx.Rngs):
    self.num_heads = config.num_heads
    self.head_dim = config.hidden_size // config.num_heads
    self.scale = self.head_dim ** -0.5
    self.dtype = config.dtype
    self.qkv = nnx.Linear(
        config.hidden_size, config.hidden_size * 3,
        use_bias=True, rngs=rngs,
        dtype=config.dtype, param_dtype=config.param_dtype,
    )
    self.proj = nnx.Linear(
        config.hidden_size, config.hidden_size,
        use_bias=True, rngs=rngs,
        dtype=config.dtype, param_dtype=config.param_dtype,
    )

  def __call__(
      self,
      hidden_states: jaxtyping.Array,  # [N, D]
      position_embeddings: Tuple[jaxtyping.Array, jaxtyping.Array],
      # (cos, sin) each [N, head_dim]
  ) -> jaxtyping.Array:
    N, D = hidden_states.shape
    H, Hd = self.num_heads, self.head_dim

    # QKV
    qkv = self.qkv(hidden_states)           # [N, 3*D]
    qkv = qkv.reshape(N, 3, H, Hd)         # [N, 3, H, Hd]
    q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]  # each [N, H, Hd]

    # 2-D RoPE
    cos, sin = position_embeddings
    q, k = _apply_rotary_pos_emb_vision(q, k, cos, sin)

    # Scaled dot-product attention  (non-causal, no mask)
    # Transpose to [H, N, Hd] for batched matmul
    q = q.transpose(1, 0, 2) * self.scale  # [H, N, Hd]
    k = k.transpose(1, 0, 2)               # [H, N, Hd]
    v = v.transpose(1, 0, 2)               # [H, N, Hd]

    attn = jnp.einsum('hnd,hmd->hnm', q, k)  # [H, N, N]
    attn = jax.nn.softmax(attn, axis=-1).astype(self.dtype)
    out = jnp.einsum('hnm,hmd->hnd', attn, v)  # [H, N, Hd]
    out = out.transpose(1, 0, 2).reshape(N, D)  # [N, D]
    return self.proj(out)


class VisionBlock(nnx.Module):
  """One ViT transformer block: LN → Attn → residual, LN → MLP → residual."""

  def __init__(self, config: VisionConfig, *, rngs: nnx.Rngs):
    self.norm1 = nnx.LayerNorm(
        config.hidden_size, epsilon=config.norm_eps,
        use_bias=True, rngs=rngs,
        dtype=config.dtype, param_dtype=config.param_dtype,
    )
    self.norm2 = nnx.LayerNorm(
        config.hidden_size, epsilon=config.norm_eps,
        use_bias=True, rngs=rngs,
        dtype=config.dtype, param_dtype=config.param_dtype,
    )
    self.attn = VisionAttention(config, rngs=rngs)
    self.mlp = VisionMLP(config, rngs=rngs)

  def __call__(
      self,
      hidden_states: jaxtyping.Array,  # [N, D]
      position_embeddings: Tuple[jaxtyping.Array, jaxtyping.Array],
  ) -> jaxtyping.Array:
    hidden_states = hidden_states + self.attn(
        self.norm1(hidden_states), position_embeddings
    )
    hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
    return hidden_states


class VisionPatchMerger(nnx.Module):
  """Spatial 2×2 patch merger with MLP projection.

  Reshapes ``[N, D]`` → ``[N // M², D * M²]`` then projects to
  ``out_hidden_size`` via ``LN → Linear → GELU → Linear``.
  Matches HF ``Qwen3_5VisionPatchMerger(use_postshuffle_norm=False)``.
  """

  def __init__(self, config: VisionConfig, *, rngs: nnx.Rngs):
    M = config.spatial_merge_size
    self.merge_unit = M * M  # 4
    inner_dim = config.hidden_size * self.merge_unit  # 768 * 4 = 3072

    # norm is applied to the *unmerged* tokens (use_postshuffle_norm=False)
    self.norm = nnx.LayerNorm(
        config.hidden_size, epsilon=config.norm_eps,
        use_bias=True, rngs=rngs,
        dtype=config.dtype, param_dtype=config.param_dtype,
    )
    self.fc1 = nnx.Linear(
        inner_dim, inner_dim,
        use_bias=True, rngs=rngs,
        dtype=config.dtype, param_dtype=config.param_dtype,
    )
    self.fc2 = nnx.Linear(
        inner_dim, config.out_hidden_size,
        use_bias=True, rngs=rngs,
        dtype=config.dtype, param_dtype=config.param_dtype,
    )

  def __call__(self, x: jaxtyping.Array) -> jaxtyping.Array:
    """``[N, D]`` → ``[N // M², out_hidden_size]``."""
    N, D = x.shape
    M = self.merge_unit
    assert N % M == 0, f'N={N} not divisible by spatial_merge_unit={M}'
    # HF: norm(x).view(-1, hidden*M²) then fc2(gelu(fc1(...)))
    x = self.norm(x)                        # [N, D]
    x = x.reshape(N // M, M * D)           # [N//M², 3072]
    x = self.fc2(_gelu_tanh(self.fc1(x)))  # [N//M², out_hidden_size]
    return x


# ---------------------------------------------------------------------------
# Top-level vision model
# ---------------------------------------------------------------------------


class Qwen3_5VisionModel(nnx.Module):
  """Qwen3.5 SigLIP-style ViT vision encoder.

  Usage::

      vis = Qwen3_5VisionModel(VisionConfig.qwen3_5_0p8b(), rngs=nnx.Rngs(0))
      # pixel_patches: [total_patches, 768]  (C*T*H*W = 3*2*16*16)
      # grid_thw:      [N_images, 3]  (T, H_grids, W_grids)
      tokens = vis(pixel_patches, grid_thw)  # [total_patches // 4, 1024]
  """

  def __init__(self, config: VisionConfig, *, rngs: nnx.Rngs):
    self.config = config

    self.patch_embed = VisionPatchEmbed(config, rngs=rngs)

    # Learned absolute 2-D position embeddings (max grid 48×48).
    self.pos_embed = nnx.Embed(
        num_embeddings=config.num_position_embeddings,
        features=config.hidden_size,
        rngs=rngs,
        dtype=config.dtype,
        param_dtype=config.param_dtype,
    )
    self.num_grid_per_side = int(math.isqrt(config.num_position_embeddings))

    head_dim = config.hidden_size // config.num_heads
    self.rotary_pos_emb = VisionRotaryEmbedding(head_dim // 2)

    self.blocks = nnx.List([VisionBlock(config, rngs=rngs) for _ in range(config.depth)])
    self.merger = VisionPatchMerger(config, rngs=rngs)

  # ------------------------------------------------------------------
  # Positional embedding helpers
  # ------------------------------------------------------------------

  def _pos_embed_indices(
      self, grid_thw: jaxtyping.Array  # [N, 3]
  ) -> jaxtyping.Array:
    """Build flat 2-D position indices for all patches.

    For each image ``(T, H, W)`` we generate ``T * H * W`` indices into the
    ``pos_embed`` table (which is a 48×48 grid, so index = row * 48 + col).
    Returns ``[total_patches]`` int32 indices.
    """
    indices = []
    for t, h, w in grid_thw.tolist():
      t, h, w = int(t), int(h), int(w)
      G = self.num_grid_per_side
      rows = jnp.repeat(jnp.arange(h), w)    # [h*w]
      cols = jnp.tile(jnp.arange(w), h)      # [h*w]
      flat = rows * G + cols                  # [h*w]
      if t > 1:
        flat = jnp.tile(flat, t)             # [t*h*w]
      indices.append(flat)
    return jnp.concatenate(indices, axis=0).astype(jnp.int32)

  def _rot_pos_emb(
      self, grid_thw: jaxtyping.Array  # [N_images, 3]
  ) -> Tuple[jaxtyping.Array, jaxtyping.Array]:
    """Build 2-D RoPE ``(cos, sin)`` for all patches.

    Matches HF VisionModel.forward::

        rotary_pos_emb = self.rot_pos_emb(grid_thw)   # [N, head_dim//2]
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)  # [N, head_dim]
        position_embeddings = (emb.cos(), emb.sin())

    The freq table has shape ``[max_hw, head_dim//4]``.  For each patch we look
    up the (row_freq, col_freq) vectors (each ``head_dim//4``), concatenate them
    to ``head_dim//2``, then duplicate that to ``head_dim`` before taking
    cos/sin.
    """
    max_hw = max(
        max(int(h), int(w))
        for _, h, w in grid_thw.tolist()
    )
    freq_table = self.rotary_pos_emb(max_hw)  # [max_hw, head_dim//4]
    M = self.config.spatial_merge_size        # = 2

    all_emb = []
    for t, h, w in grid_thw.tolist():
      t, h, w = int(t), int(h), int(w)
      # HF rot_pos_emb uses merge-size interleaved ordering:
      #   merged grid: mH x mW, each cell is M x M full-res patches.
      #   row_idx = block_row * M + intra_row  (similar for col)
      # The nested loops produce: for each (block_r, block_c, intra_r, intra_c)
      mH, mW = h // M, w // M
      block_rows = jnp.arange(mH)   # [mH]
      block_cols = jnp.arange(mW)   # [mW]
      intra_row  = jnp.arange(M)    # [M]
      intra_col  = jnp.arange(M)    # [M]
      # Full row indices: [mH, mW, M, M]
      row_idx = (block_rows[:, None, None, None] * M
                 + intra_row[None, None, :, None])
      col_idx = (block_cols[None, :, None, None] * M
                 + intra_col[None, None, None, :])
      # Broadcast to [mH, mW, M, M] and flatten to [h*w]
      row_idx = jnp.broadcast_to(row_idx, (mH, mW, M, M)).reshape(-1)
      col_idx = jnp.broadcast_to(col_idx, (mH, mW, M, M)).reshape(-1)

      row_freqs = freq_table[row_idx]  # [h*w, head_dim//4]
      col_freqs = freq_table[col_idx]  # [h*w, head_dim//4]
      # cat row+col → [h*w, head_dim//2], then duplicate → [h*w, head_dim]
      emb = jnp.concatenate([row_freqs, col_freqs], axis=-1)
      emb = jnp.concatenate([emb, emb], axis=-1)
      if t > 1:
        emb = jnp.tile(emb, (t, 1))
      all_emb.append(emb)

    emb = jnp.concatenate(all_emb, axis=0)   # [total_patches, head_dim]
    cos = jnp.cos(emb).astype(self.config.dtype)
    sin = jnp.sin(emb).astype(self.config.dtype)
    return cos, sin

  # ------------------------------------------------------------------
  # Forward
  # ------------------------------------------------------------------

  @jax.named_scope('qwen3_5_vision')
  def __call__(
      self,
      pixel_patches: jaxtyping.Array,   # [total_patches, patch_dim]
      grid_thw: jaxtyping.Array,        # [N_images, 3]
  ) -> jaxtyping.Array:
    """Encode image/video patches into LLM-compatible token embeddings.

    Args:
      pixel_patches: ``[total_patches, C * T * H * W]`` float32 values in
          ``[0, 1]``.  Pre-packed by the data pipeline.
      grid_thw: ``[N_images, 3]`` int32 ``(T, H_grids, W_grids)`` per image.

    Returns:
      ``[total_patches // spatial_merge_size², out_hidden_size]`` float32
      visual token embeddings, ready to be inserted into the LLM sequence.
    """
    # 1. Patch projection
    x = self.patch_embed(pixel_patches)   # [N, 768]

    # 2. Absolute positional embedding (2-D bilinear-interpolated grid)
    pos_ids = self._pos_embed_indices(grid_thw)  # [N]
    x = x + self.pos_embed(pos_ids)       # [N, 768]

    # 3. 2-D RoPE for attention
    position_embeddings = self._rot_pos_emb(grid_thw)  # (cos, sin) [N, 64]

    # 4. ViT transformer blocks
    for block in self.blocks:
      x = block(x, position_embeddings)  # [N, 768]

    # 5. Spatial merger: [N, 768] → [N//4, 1024]
    x = self.merger(x)
    return x
