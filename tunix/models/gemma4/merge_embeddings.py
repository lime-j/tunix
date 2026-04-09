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

"""Utility to merge vision soft tokens into text embedding sequences for Gemma 4."""

import jax
from jax import numpy as jnp
import jaxtyping


def merge_embeddings(
    *,
    text_embeddings: jaxtyping.Array,
    vision_embeddings: jaxtyping.Array,
    mask: jaxtyping.Array,
) -> jaxtyping.Array:
  """Replaces image‑token positions in text_embeddings with vision features.

  Args:
    text_embeddings: (B, T, D) — token embeddings including placeholder tokens.
    vision_embeddings: (B, N_images, L_vis, D) — projected vision soft tokens.
    mask: (B, T) bool — True at positions where image_token_id appears.

  Returns:
    (B, T, D) with image‑token positions replaced by the corresponding vision
    soft tokens (in row‑major order across images and patches).
  """
  b, t, d = text_embeddings.shape
  b2, n, l, d2 = vision_embeddings.shape
  assert b == b2 and d == d2

  # Flatten vision tokens to (B, N*L, D)
  flat_vision = vision_embeddings.reshape(b, n * l, d)

  # For each batch element, scatter flat_vision into the image-token positions.
  # We use a simple sequential assignment via jnp.where with a running index.
  # Build a (B, T) index mapping: the k-th True position in row b gets vision token k.
  cumsum = jnp.cumsum(mask, axis=-1) - 1  # 0-indexed rank of each image token
  # Clip to [0, n*l-1] to avoid OOB on non-image positions
  safe_idx = jnp.clip(cumsum, 0, n * l - 1)  # (B, T)

  # Gather vision tokens for each position
  gathered = flat_vision[jnp.arange(b)[:, None], safe_idx, :]  # (B, T, D)

  return jnp.where(mask[..., None], gathered, text_embeddings)
