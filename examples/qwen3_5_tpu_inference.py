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
# Greedy sampler
# --------------------------------------------------------------------------

def greedy_generate(lm_model, input_ids, max_new_tokens=200,
                    eos_token_id=151645, input_embeddings=None):
  """Greedy autoregressive decoding.

  Prefill uses a 3-D causal mask [B, T_q, cache_size] which is applied even
  when the KV cache is non-null (fix in model.py).  Decode step uses a
  [B, 1, cache_size] mask attending to all positions written so far.
  """
  prompt_len = len(input_ids)
  cache_size = prompt_len + max_new_tokens + 4

  tokens = jnp.array(input_ids, dtype=jnp.int32)[None, :]
  positions = jnp.arange(prompt_len, dtype=jnp.int32)[None, :]

  causal = jnp.tril(jnp.ones((prompt_len, prompt_len), dtype=jnp.bool_))
  attn_mask = jnp.pad(
      causal[None], ((0, 0), (0, 0), (0, cache_size - prompt_len))
  )  # [1, prompt_len, cache_size]

  cache = lm_model.init_cache(
      batch_size=1, cache_size=cache_size, dtype=jnp.bfloat16,
  )

  logits, cache = lm_model(
      input_tokens=tokens, positions=positions, cache=cache,
      attention_mask=attn_mask, input_embeddings=input_embeddings,
  )
  next_token = int(jnp.argmax(logits[0, -1]))
  generated = [next_token]
  if next_token == eos_token_id:
    return generated

  for step in range(max_new_tokens - 1):
    pos = prompt_len + step
    tok = jnp.array([[next_token]], dtype=jnp.int32)
    pos_arr = jnp.array([[pos]], dtype=jnp.int32)
    dec_mask = jnp.zeros((1, 1, cache_size), dtype=jnp.bool_)
    dec_mask = dec_mask.at[0, 0, :pos + 1].set(True)
    logits, cache = lm_model(
        input_tokens=tok, positions=pos_arr,
        cache=cache, attention_mask=dec_mask,
    )
    next_token = int(jnp.argmax(logits[0, 0]))
    generated.append(next_token)
    if next_token == eos_token_id:
      break

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
  args = parser.parse_args()

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
