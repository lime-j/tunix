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

"""Gated Delta Rule kernels for Qwen3.5 linear attention layers.

Reference: "Gated Linear Attention Transformers with Hardware-Efficient Training"
https://arxiv.org/pdf/2412.06464
"""

from __future__ import annotations

import math

import jax
from jax import numpy as jnp


def l2norm(x: jax.Array, axis: int = -1, eps: float = 1e-6) -> jax.Array:
  """L2-normalise x along axis."""
  inv_norm = jax.lax.rsqrt(jnp.sum(x * x, axis=axis, keepdims=True) + eps)
  return x * inv_norm


def recurrent_gated_delta_rule(
    query: jax.Array,        # [B, T, H, D_k]
    key: jax.Array,          # [B, T, H, D_k]
    value: jax.Array,        # [B, T, H, D_v]
    g: jax.Array,            # [B, T, H]  log decay gate
    beta: jax.Array,         # [B, T, H]  update gate
    initial_state: jax.Array | None = None,  # [B, H, D_k, D_v]
) -> tuple[jax.Array, jax.Array]:
  """Token-by-token (recurrent) implementation of the Gated Delta Rule.

  Preferred at decode time (seq_len == 1). Equivalent to
  chunk_gated_delta_rule but processes one token at a time via jax.lax.scan.

  Returns:
    output: [B, T, H, D_v]
    final_state: [B, H, D_k, D_v]
  """
  dtype = query.dtype
  query = l2norm(query, axis=-1)
  key = l2norm(key, axis=-1)
  query = query * (1.0 / math.sqrt(query.shape[-1]))

  # [B, T, H, D] -> [T, B, H, D] for jax.lax.scan
  query = jnp.swapaxes(query, 0, 1)
  key = jnp.swapaxes(key, 0, 1)
  value = jnp.swapaxes(value, 0, 1)
  g = jnp.swapaxes(g, 0, 1)
  beta = jnp.swapaxes(beta, 0, 1)

  batch_size = query.shape[1]
  num_heads = query.shape[2]
  k_head_dim = query.shape[3]
  v_head_dim = value.shape[3]

  if initial_state is None:
    state0 = jnp.zeros(
        (batch_size, num_heads, k_head_dim, v_head_dim), dtype=dtype
    )
  else:
    state0 = initial_state.astype(dtype)

  def step_fn(state, inputs):
    q_t, k_t, v_t, g_t, beta_t = inputs
    decay = jnp.exp(g_t).astype(dtype)[..., None, None]
    state = state * decay
    kv_mem = jnp.sum(state * k_t[..., :, None], axis=-2)
    delta = (v_t - kv_mem) * beta_t[..., None]
    state = state + k_t[..., :, None] * delta[..., None, :]
    out_t = jnp.sum(state * q_t[..., :, None], axis=-2)
    return state, out_t

  final_state, outputs = jax.lax.scan(
      step_fn, state0, (query, key, value, g, beta)
  )
  outputs = jnp.swapaxes(outputs, 0, 1).astype(dtype)
  return outputs, final_state.astype(dtype)


def chunk_gated_delta_rule(
    query: jax.Array,        # [B, T, H, D_k]
    key: jax.Array,          # [B, T, H, D_k]
    value: jax.Array,        # [B, T, H, D_v]
    g: jax.Array,            # [B, T, H]
    beta: jax.Array,         # [B, T, H]
    chunk_size: int = 64,
    initial_state: jax.Array | None = None,  # [B, H, D_k, D_v]
) -> tuple[jax.Array, jax.Array]:
  """Chunked parallel implementation of the Gated Delta Rule.

  Processes tokens in chunks for better parallelisation via the WY
  representation (Section 3.3 of https://arxiv.org/pdf/2412.06464).

  Notation:
    Q, K, V  — query, key, value
    β        — update gate
    γ[r]     — cumulative decay exp(cumsum(g))[r]
    γ^C      — γ at last position in chunk
    Γ[i,j]   — γ[i]/γ[j] for i >= j (lower-triangular decay mask)
    S        — recurrent state [D_k, D_v]

  Returns:
    output: [B, T, H, D_v]
    final_state: [B, H, D_k, D_v]
  """
  dtype = query.dtype
  query = l2norm(query, axis=-1)
  key = l2norm(key, axis=-1)

  # [B, T, H, D] -> [B, H, T, D]
  query = jnp.transpose(query, (0, 2, 1, 3))
  key = jnp.transpose(key, (0, 2, 1, 3))
  value = jnp.transpose(value, (0, 2, 1, 3))
  beta = jnp.transpose(beta, (0, 2, 1))
  g = jnp.transpose(g, (0, 2, 1))

  batch_size, num_heads, seq_len, k_head_dim = key.shape
  v_head_dim = value.shape[-1]
  query = query * (1.0 / math.sqrt(k_head_dim))

  # Pad sequence to be divisible by chunk_size
  pad_size = (chunk_size - seq_len % chunk_size) % chunk_size
  if pad_size > 0:
    query = jnp.pad(query, ((0, 0), (0, 0), (0, pad_size), (0, 0)))
    key = jnp.pad(key, ((0, 0), (0, 0), (0, pad_size), (0, 0)))
    value = jnp.pad(value, ((0, 0), (0, 0), (0, pad_size), (0, 0)))
    beta = jnp.pad(beta, ((0, 0), (0, 0), (0, pad_size)))
    g = jnp.pad(g, ((0, 0), (0, 0), (0, pad_size)))

  total_seq_len = seq_len + pad_size
  num_chunks = total_seq_len // chunk_size

  # [B, H, T, D] -> [B, H, C, L, D]
  query = query.reshape(batch_size, num_heads, num_chunks, chunk_size, k_head_dim)
  key = key.reshape(batch_size, num_heads, num_chunks, chunk_size, k_head_dim)
  value = value.reshape(batch_size, num_heads, num_chunks, chunk_size, v_head_dim)
  beta = beta.reshape(batch_size, num_heads, num_chunks, chunk_size)
  g = g.reshape(batch_size, num_heads, num_chunks, chunk_size)

  k_beta = key * beta[..., None]
  v_beta = value * beta[..., None]

  # γ[j] = exp(cumsum(g)[j])
  g_cumsum = jnp.cumsum(g, axis=-1)
  gamma = jnp.exp(g_cumsum).astype(dtype)

  # Γ[i,j] = exp(g_cumsum[i] - g_cumsum[j])  (lower-triangular)
  decay_mask = jnp.tril(
      jnp.exp(jnp.tril(g_cumsum[..., :, None] - g_cumsum[..., None, :]))
  ).astype(dtype)

  # L = strictLower(diag(β)(Γ ⊙ K K^T))
  L = jnp.tril(
      (k_beta @ jnp.swapaxes(key, -1, -2)) * decay_mask, k=-1
  )

  # Solve (I + L) x = rhs  for Ũ (corrected values) and ←W (decayed keys)
  rhs = jnp.concatenate([v_beta, k_beta * gamma[..., None]], axis=-1)
  solution = jax.lax.linalg.triangular_solve(
      L, rhs, left_side=True, lower=True, unit_diagonal=True
  )
  U = solution[..., :v_head_dim]
  W_decay = solution[..., v_head_dim:]

  if initial_state is None:
    state0 = jnp.zeros(
        (batch_size, num_heads, k_head_dim, v_head_dim), dtype=dtype
    )
  else:
    state0 = initial_state.astype(dtype)

  def chunk_step(S, inputs):
    Q_t, K_t, U_t, W_decay_t, gamma_t, Gamma_t = inputs

    # Intra-chunk attention with decay mask
    intra_attn = Q_t @ jnp.swapaxes(K_t, -1, -2) * Gamma_t

    # Corrected values minus state contribution
    U_minus_W_S = U_t - W_decay_t @ S

    # Inter-chunk contribution (←Q = γ * Q)
    inter_out = (Q_t * gamma_t[..., None]) @ S

    # O[t] = ←Q S^T + (Q K^T ⊙ Γ)(Ũ - ←W S^T)
    O_t = inter_out + intra_attn @ U_minus_W_S

    # S[t+1] = γ^C * S + →K^T (Ũ - ←W S^T)
    gamma_C = gamma_t[..., -1, None, None]
    key_decay_t = Gamma_t[..., -1, :][..., None]
    S = gamma_C * S + jnp.swapaxes(K_t * key_decay_t, -1, -2) @ U_minus_W_S

    return S, O_t

  # Transpose chunk dim to front for scan: [B, H, C, ...] -> [C, B, H, ...]
  scan_inputs = (
      jnp.transpose(query, (2, 0, 1, 3, 4)),
      jnp.transpose(key, (2, 0, 1, 3, 4)),
      jnp.transpose(U, (2, 0, 1, 3, 4)),
      jnp.transpose(W_decay, (2, 0, 1, 3, 4)),
      jnp.transpose(gamma, (2, 0, 1, 3)),
      jnp.transpose(decay_mask, (2, 0, 1, 3, 4)),
  )

  final_state, outputs = jax.lax.scan(chunk_step, state0, scan_inputs)

  # [C, B, H, L, D_v] -> [B, T, H, D_v]
  outputs = jnp.transpose(outputs, (1, 2, 0, 3, 4))
  outputs = outputs.reshape(batch_size, num_heads, total_seq_len, v_head_dim)
  outputs = jnp.transpose(outputs[:, :, :seq_len, :], (0, 2, 1, 3)).astype(dtype)

  return outputs, final_state.astype(dtype)
