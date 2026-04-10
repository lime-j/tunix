# -*- coding: utf-8 -*-
"""Qwen3.5-0.8B VLM inference on TPU.

This script uses the full tunix stack (no CPU stubs) and is designed to run
on a JAX-visible TPU host.

Usage:
    # Text-only
    python examples/qwen3_5_tpu_inference.py \
        --ckpt_dir ~/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/<hash> \
        --prompt "What is the capital of France?"

    # With image
    python examples/qwen3_5_tpu_inference.py \
        --ckpt_dir ... \
        --prompt "Describe this image." \
        --image /path/to/image.jpg

The checkpoint directory must contain the HF safetensors file(s) for
Qwen/Qwen3.5-0.8B.  The directory is passed to both the LM and vision loaders.
"""

import os
import sys
from typing import Any, Callable

# 强行设置环境变量，优先级最高
os.environ["JAX_PROCESS_COUNT"] = "1"
os.environ["JAX_PROCESS_ID"] = "0"
os.environ["JAX_COORDINATION_SERVICE_ADDR"] = "localhost:8888"
os.environ["TPU_CHIPS_PER_HOST_BOUNDS"] = "2,2,1"
os.environ["TPU_HOST_BOUNDS"] = "1,1,1"
os.environ["TPU_MIN_DRIVERS"] = "1"

# 禁用 TensorFlow 抢占 TPU (如果脚本里有 TF/Keras)
os.environ["TPU_VISIBLE_DEVICES"] = "" 

import jax
# 强制屏蔽分布式初始化
def dummy_init(*args, **kwargs):
    print('JAX Distributed Init Suppressed (Single-node mode)')
    return
try:
    import jax.distributed
    jax.distributed.initialize = dummy_init
except:
    pass

# ---------------------------------------------------------------------------
# Persistent XLA compilation cache
# Must be set BEFORE any jax.jit call (i.e., before model loading).
# On first run: compiles and saves. On subsequent runs: loads from disk.
# Cache key = hash(XLA computation) + JAX version + hardware.
# ---------------------------------------------------------------------------
_DEFAULT_JAX_CACHE = os.path.expanduser('~/.cache/jax_compile_cache/qwen3_5')

def _enable_jax_cache(cache_dir: str):
    os.makedirs(cache_dir, exist_ok=True)
    jax.config.update('jax_compilation_cache_dir', cache_dir)
    # Only persist compilations that took >= 5 s (avoids caching trivial ops).
    jax.config.update('jax_persistent_cache_min_compile_time_secs', 5.0)
    print(f'[JAX cache] Persistent compilation cache: {cache_dir}')

print("Local Devices:", jax.local_device_count())
from pathlib import Path

import argparse
import math
import os

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from transformers import AutoTokenizer

from tunix.models.qwen3_5 import model as qwen_model
from tunix.models.qwen3_5 import params as qwen_params
from tunix.models.qwen3_5 import vision as vision_lib
from tunix.utils import env_utils

# --------------------------------------------------------------------------
# Special token IDs
# --------------------------------------------------------------------------
IMAGE_TOKEN_ID = 151654   # <|image_pad|>

# Vision preprocessing constants (SigLIP-style)
VIS_PATCH_SIZE    = 16
VIS_SPATIAL_MERGE = 2
VIS_NORM_MEAN     = (0.5, 0.5, 0.5)
VIS_NORM_STD      = (0.5, 0.5, 0.5)


# --------------------------------------------------------------------------
# Image preprocessing
# --------------------------------------------------------------------------

def preprocess_image(image_path: str):
  """Resize, normalize, and patch an image for the Qwen3.5-VL vision encoder.

  Returns:
    pixel_patches: jnp.ndarray  [N, 1536]  float32
    grid_thw:      jnp.ndarray  [1, 3]     int32   (T=1, H_grids, W_grids)
    num_image_tokens: int  — number of <image_pad> slots after spatial merger
  """
  try:
    from PIL import Image
  except ImportError:
    raise ImportError('pip install Pillow')

  img = Image.open(image_path).convert('RGB')
  W_orig, H_orig = img.size

  P = VIS_PATCH_SIZE
  target_patches = 256
  scale = math.sqrt(target_patches * P * P / (H_orig * W_orig))
  H_new = max(P, round(H_orig * scale / P) * P)
  W_new = max(P, round(W_orig * scale / P) * P)
  M = P * VIS_SPATIAL_MERGE   # must be divisible by this
  H_new = max(M, round(H_new / M) * M)
  W_new = max(M, round(W_new / M) * M)

  img_resized = img.resize((W_new, H_new), Image.BICUBIC)
  arr = np.array(img_resized, dtype=np.float32) / 255.0
  mean = np.array(VIS_NORM_MEAN, dtype=np.float32)
  std  = np.array(VIS_NORM_STD,  dtype=np.float32)
  arr  = (arr - mean) / std

  H_grids = H_new // P
  W_grids = W_new // P
  T = 1

  arr = arr.reshape(H_grids, P, W_grids, P, 3)
  arr = arr.transpose(0, 2, 4, 1, 3)            # [H_g, W_g, C, P, P]
  arr = arr.reshape(H_grids * W_grids, 3, P, P)
  arr = np.stack([arr, arr], axis=2)             # temporal dup: [N, 3, 2, P, P]
  arr = arr.reshape(H_grids * W_grids, 3 * 2 * P * P)  # [N, 1536]

  pixel_patches = jnp.array(arr)
  grid_thw = jnp.array([[T, H_grids, W_grids]], dtype=jnp.int32)
  num_image_tokens = (H_grids * W_grids * T) // (VIS_SPATIAL_MERGE ** 2)
  print(f'  [{W_orig}x{H_orig}] -> [{W_new}x{H_new}]'
        f' -> {H_grids * W_grids} patches -> {num_image_tokens} visual tokens')
  return pixel_patches, grid_thw, num_image_tokens


# --------------------------------------------------------------------------
# VLM embedding builder
# --------------------------------------------------------------------------

def build_vlm_embeddings(lm_model, vis_model, input_ids, pixel_patches, grid_thw):
  """Encode image and splice visual embeddings into the text token sequence.

  Returns embeddings: jnp.ndarray [1, L, embed_dim].
  """
  tokens = jnp.array(input_ids, dtype=jnp.int32)[None, :]
  text_embeds = lm_model.embedder.encode(tokens)      # [1, L, D]
  visual_tokens = vis_model(pixel_patches, grid_thw)  # [N_vis, D]

  ids_arr = np.array(input_ids)
  image_positions = np.where(ids_arr == IMAGE_TOKEN_ID)[0]
  n_vis = min(len(image_positions), int(visual_tokens.shape[0]))
  if len(image_positions) != int(visual_tokens.shape[0]):
    print(f'  WARNING: {len(image_positions)} <image_pad> slots but '
          f'{visual_tokens.shape[0]} visual vectors — using {n_vis}')

  embeds = text_embeds
  for idx in range(n_vis):
    embeds = embeds.at[0, int(image_positions[idx])].set(visual_tokens[idx])
  return embeds


# --------------------------------------------------------------------------
# Greedy sampler — power-of-2 padded prefill + lax.scan decode
# --------------------------------------------------------------------------

def _next_pow2(n: int) -> int:
  """Smallest power of 2 >= n (minimum 1). Limits recompilation to O(log N) buckets."""
  p = 1
  while p < n:
    p <<= 1
  return p


def _make_prefill_fn(lm_model):
  """JIT-compiled prefill (text tokens)."""
  @jax.jit
  def prefill(tokens, positions, attn_mask, cache):
    return lm_model(
        input_tokens=tokens, positions=positions,
        cache=cache, attention_mask=attn_mask,
    )
  return prefill


def _make_prefill_embed_fn(lm_model):
  """JIT-compiled prefill (pre-built embeddings, VLM path)."""
  @jax.jit
  def prefill_embed(dummy_tokens, positions, attn_mask, cache, embeddings):
    return lm_model(
        input_tokens=dummy_tokens, positions=positions,
        cache=cache, attention_mask=attn_mask,
        input_embeddings=embeddings,
    )
  return prefill_embed


def _make_scan_decode_fn(lm_model, T_pad: int, n_steps: int):
  """JIT-compiled lax.scan decode.

  Using T_pad (not actual prompt_len) as the base position means every prompt
  padded to the same power-of-2 bucket shares one compiled decode kernel.

  Args:
    T_pad:   power-of-2 padded prefill length (static, baked into XLA).
    n_steps: max_new_tokens - 1  (static scan length, baked into XLA).
  """
  @jax.jit
  def run_decode(first_token, cache, dec_mask):
    """
    first_token : scalar int32 — first generated token (from prefill logit).
    cache       : per-layer KV/conv/recurrent cache.
    dec_mask    : [1, 1, cache_size] bool — prefill positions already True.
    Returns     : [n_steps] int32 token ids.
    """
    def body(carry, step):
      cache, dec_mask, last_token = carry
      pos     = T_pad + step                                   # traced scalar
      tok     = last_token[None, None]                         # [1, 1]
      pos_arr = jnp.full((1, 1), pos, dtype=jnp.int32)        # [1, 1]
      # Mark this position valid before running attention.
      dec_mask = jax.lax.dynamic_update_slice(
          dec_mask, jnp.ones((1, 1, 1), jnp.bool_), (0, 0, pos)
      )
      logits, new_cache = lm_model(
          input_tokens=tok, positions=pos_arr,
          cache=cache, attention_mask=dec_mask,
      )
      next_tok = jnp.argmax(logits[0, 0]).astype(jnp.int32)
      return (new_cache, dec_mask, next_tok), next_tok

    steps = jnp.arange(n_steps, dtype=jnp.int32)
    init  = (cache, dec_mask, jnp.asarray(first_token, jnp.int32))
    (_, _, _), all_tokens = jax.lax.scan(body, init, steps)
    return all_tokens  # [n_steps]

  return run_decode


def greedy_generate(lm_model, input_ids, max_new_tokens=200,
                    eos_token_id=151645, input_embeddings=None):
  """Greedy decode: power-of-2 padded prefill + lax.scan decode loop."""
  import time
  prompt_len    = len(input_ids)
  T_pad         = _next_pow2(prompt_len)          # static prefill length
  cache_size    = T_pad + max_new_tokens           # static cache length
  pad_len       = T_pad - prompt_len
  compute_dtype = lm_model.config.dtype

  # ---- Pad inputs to T_pad ----
  ids_np      = np.array(input_ids, dtype=np.int32)
  tokens_pad  = jnp.pad(jnp.array(ids_np)[None], ((0, 0), (0, pad_len)))
  pos_pad     = jnp.pad(
      jnp.arange(prompt_len, dtype=jnp.int32)[None], ((0, 0), (0, pad_len))
  )

  # ---- Prefill mask [1, T_pad, cache_size] ----
  # Real rows: lower-triangular causal.
  # Padded rows: attend to position 0 only (prevents softmax NaN on all-inf rows).
  mask_np = np.zeros((1, T_pad, cache_size), dtype=bool)
  for i in range(prompt_len):
    mask_np[0, i, :i + 1] = True
  if pad_len > 0:
    mask_np[0, prompt_len:, 0] = True   # safe dummy for padded queries
  attn_mask = jnp.array(mask_np)

  # ---- Cache + initial decode mask ----
  cache    = lm_model.init_cache(batch_size=1, cache_size=cache_size, dtype=compute_dtype)
  # Decode will start at T_pad; mark the real prefill positions as valid.
  dec_mask = jnp.zeros((1, 1, cache_size), dtype=jnp.bool_)
  dec_mask = dec_mask.at[0, 0, :prompt_len].set(True)

  # ---- Prefill ----
  t0 = time.perf_counter()
  if input_embeddings is not None:
    embed_pad = jnp.pad(input_embeddings, ((0, 0), (0, pad_len), (0, 0)))
    logits, cache = _make_prefill_embed_fn(lm_model)(
        tokens_pad, pos_pad, attn_mask, cache, embed_pad
    )
  else:
    logits, cache = _make_prefill_fn(lm_model)(tokens_pad, pos_pad, attn_mask, cache)
  logits.block_until_ready()
  t_pre = time.perf_counter() - t0
  print(f'     Prefill ({prompt_len} tok, pad→{T_pad}): '
        f'{t_pre*1000:.1f} ms  [{prompt_len/t_pre:.0f} tok/s]')

  # Logit at the last *real* token (padded positions are garbage).
  first_token = int(jnp.argmax(logits[0, prompt_len - 1]))
  generated   = [first_token]
  if first_token == eos_token_id or max_new_tokens <= 1:
    return generated

  # ---- lax.scan decode ----
  n_steps        = max_new_tokens - 1
  scan_decode_fn = _make_scan_decode_fn(lm_model, T_pad, n_steps)

  t1         = time.perf_counter()
  all_tokens = scan_decode_fn(first_token, cache, dec_mask)
  all_tokens.block_until_ready()
  t_dec      = time.perf_counter() - t1

  # Truncate at first EOS (scan always runs the full n_steps in XLA).
  all_np  = np.array(all_tokens, dtype=np.int32)
  eos_pos = np.where(all_np == eos_token_id)[0]
  n_keep  = int(eos_pos[0]) + 1 if len(eos_pos) else len(all_np)
  generated.extend(all_np[:n_keep].tolist())

  print(f'     Decode  ({n_steps} XLA steps → {n_keep} used): '
        f'{t_dec*1000:.1f} ms  [{n_steps/t_dec:.1f} tok/s]')
  return generated


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
  parser = argparse.ArgumentParser(description='Qwen3.5-0.8B VLM TPU inference')
  parser.add_argument(
      '--ckpt_dir',
      default=os.path.expanduser(
          '~/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/'
          'snapshots/2fc06364715b967f1860aea9cf38778875588b17'
      ),
      help='Directory containing the HF safetensors checkpoint',
  )
  parser.add_argument('--prompt', default='The capital of France is')
  parser.add_argument('--image', default=None, help='Path to an image file')
  parser.add_argument('--max_new_tokens', type=int, default=200)
  parser.add_argument('--dtype', default='bfloat16',
                      choices=['float32', 'bfloat16'],
                      help='Model weight dtype')
  parser.add_argument('--flash_attention', action='store_true',
                      help='Enable Pallas Splash Attention (TPU flash attention) '
                           'for prefill. Requires a mesh; uses all available devices.')
  parser.add_argument('--flash_block_size', type=int, default=512,
                      help='Block size for Splash Attention (default 512)')
  parser.add_argument('--jax_cache_dir', default=_DEFAULT_JAX_CACHE,
                      help='Directory for JAX persistent compilation cache '
                           '(set to empty string "" to disable)')
  args = parser.parse_args()

  # Enable compile cache BEFORE any jax.jit / model loading.
  if args.jax_cache_dir:
    _enable_jax_cache(args.jax_cache_dir)

  # --- JAX / TPU init ---
  env_utils.setup_sharding_environment()
  devices = jax.devices()
  print(f'JAX devices: {devices}')
  dtype = jnp.bfloat16 if args.dtype == 'bfloat16' else jnp.float32

  # --- Tokenizer ---
  print('[1] Loading tokenizer...')
  tokenizer = AutoTokenizer.from_pretrained(
      'Qwen/Qwen3.5-0.8B', trust_remote_code=True)

  # --- Language model ---
  print(f'[2] Loading LM from {args.ckpt_dir} (dtype={args.dtype})...')
  lm_cfg = qwen_model.ModelConfig.qwen3_5_0p8b()
  lm_cfg.dtype = dtype
  if args.flash_attention:
    lm_cfg.use_flash_attention = True
    lm_cfg.flash_attention_block_size = args.flash_block_size
    # Splash Attention needs a mesh with a 'tp' axis for shard_map.
    mesh = jax.sharding.Mesh(
        np.array(jax.devices()).reshape(1, -1, 1, 1),
        axis_names=('fsdp', 'tp', 'sp', 'expert'),
    )
    # Replicate weights so all 4 devices participate; avoids TP divisibility
    # errors from GQA (num_kv_heads=2 not divisible by tp=4).
    lm_cfg.shd_config = qwen_model.ShardingConfig.get_replicated_sharding()
    print(f'     Flash attention ON  (block={args.flash_block_size}, mesh={mesh.shape})')
    print(f'     Weights replicated across {len(jax.devices())} devices')
  else:
    mesh = None
  # Do NOT pass mesh= to the loader — mesh is only activated inside @jax.jit
  # via `with mesh:` so Splash Attention can find it through thread resources.
  lm = qwen_params.create_model_from_safe_tensors(
      file_dir=args.ckpt_dir,
      config=lm_cfg,
      dtype=dtype,
  )
  print(f'     LM loaded. Params: '
        f'{sum(x.size for x in jax.tree_util.tree_leaves(nnx.state(lm))):,}')

  # --- Vision encoder ---
  print(f'[3] Loading vision encoder from {args.ckpt_dir}...')
  vis_cfg = vision_lib.VisionConfig.qwen3_5_0p8b()
  vis = qwen_params.create_vision_model_from_safe_tensors(
      file_dir=args.ckpt_dir,
      config=vis_cfg,
      dtype=dtype,
  )
  print(f'     Vision encoder loaded.')

  # --- Build prompt & embeddings ---
  input_embeddings = None
  if args.image:
    print(f'[4] Processing image: {args.image}')
    pixel_patches, grid_thw, n_img_toks = preprocess_image(args.image)

    image_placeholder = (
        '<|vision_start|>' + '<|image_pad|>' * n_img_toks + '<|vision_end|>'
    )
    messages = [{'role': 'user', 'content': image_placeholder + args.prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=False,
    )
    input_ids = tokenizer.encode(text)
    print(f'     Prompt: {len(input_ids)} tokens ({n_img_toks} image pads)')

    print('[5] Encoding image with ViT...')
    input_embeddings = build_vlm_embeddings(
        lm, vis, input_ids, pixel_patches, grid_thw)
  else:
    messages = [{'role': 'user', 'content': args.prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=False,
    )
    input_ids = tokenizer.encode(text)
    print(f'[4] Text prompt: {len(input_ids)} tokens')

  # --- Generate ---
  print('[6] Generating...')
  generated_ids = greedy_generate(
      lm, input_ids,
      max_new_tokens=args.max_new_tokens,
      eos_token_id=tokenizer.eos_token_id,
      input_embeddings=input_embeddings,
  )

  response = tokenizer.decode(generated_ids, skip_special_tokens=True)
  print('\n' + '=' * 60)
  print(f'Prompt   : {args.prompt}')
  if args.image:
    print(f'Image    : {args.image}')
  print(f'Response : {response}')
  print('=' * 60)


if __name__ == '__main__':
  main()
