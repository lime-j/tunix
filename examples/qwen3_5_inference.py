"""Qwen3.5-0.8B VLM inference (text + vision).

Usage:
    # Text-only
    JAX_PLATFORMS=cpu python examples/qwen3_5_inference.py \
        --prompt "What is the capital of France?"

    # With image
    JAX_PLATFORMS=cpu python examples/qwen3_5_inference.py \
        --prompt "Describe this image." \
        --image /path/to/image.jpg
"""

import argparse
import math
import os
import sys

# --------------------------------------------------------------------------
# Environment / stub setup (must run before any tunix imports)
# --------------------------------------------------------------------------
os.environ.setdefault('JAX_PROCESS_COUNT', '1')
os.environ.setdefault('JAX_PROCESS_ID', '0')
os.environ.setdefault('TPU_VISIBLE_DEVICES', '')
os.environ.setdefault('JAX_COORDINATION_SERVICE_ADDR', 'localhost:8888')

import types

_tunix = types.ModuleType('tunix')
_tunix.__path__ = [os.path.join(os.path.dirname(__file__), '..', 'tunix')]
_tunix.__package__ = 'tunix'
sys.modules['tunix'] = _tunix

for _mod in [
    'tunix.oss', 'tunix.oss.utils', 'kagglehub',
    'tunix.generate', 'tunix.generate.mappings',
    'tunix.utils', 'tunix.utils.env_utils', 'tunix.utils.compat',
    'tunix.models.safetensors_loader', 'tunix.models.safetensors_saver',
]:
  sys.modules[_mod] = types.ModuleType(_mod)

import tunix.generate.mappings as _gm


class _BMM:
  BACKEND_PACKAGE_PATH = ''


_gm.BackendMappingMixin = _BMM
sys.modules['tunix.utils.env_utils'].setup_sharding_environment = lambda: None
sys.modules['tunix.utils.compat'].ModuleList = list

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# --------------------------------------------------------------------------
# Real imports
# --------------------------------------------------------------------------
import re

import jax
import jax.numpy as jnp
import numpy as np
import safetensors.torch as st
from flax import nnx
from transformers import AutoTokenizer

from tunix.models.qwen3_5 import model as qwen_model
from tunix.models.qwen3_5.vision import VisionConfig, Qwen3_5VisionModel

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
_SNAP = os.path.expanduser(
    '~/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/'
    '2fc06364715b967f1860aea9cf38778875588b17'
)
CKPT_FILE = os.path.join(
    _SNAP, 'model.safetensors-00001-of-00001.safetensors'
)

# Special token IDs
IMAGE_TOKEN_ID = 151654   # <|image_pad|>

# Vision preprocessing constants
VIS_PATCH_SIZE    = 16
VIS_SPATIAL_MERGE = 2
VIS_NORM_MEAN     = (0.5, 0.5, 0.5)
VIS_NORM_STD      = (0.5, 0.5, 0.5)


# --------------------------------------------------------------------------
# Weight loading helpers
# --------------------------------------------------------------------------


def _path_to_key(path):
  """Convert JAX tree path to dot-separated string."""
  parts = []
  for p in path:
    if hasattr(p, 'key'):
      parts.append(str(p.key))
    elif hasattr(p, 'idx'):
      parts.append(str(p.idx))
    else:
      parts.append(str(p))
  return '.'.join(parts)


def _text_weight_rules(cfg: qwen_model.ModelConfig):
  """Returns (hf_pattern, tunix_key_template, (perm, shape)) triples."""
  D, H, Hd, Hkv = cfg.embed_dim, cfg.num_heads, cfg.head_dim, cfg.num_kv_heads
  return [
      (r'model\.language_model\.embed_tokens\.weight',
       'embedder.input_embedding', None),
      (r'model\.language_model\.layers\.(\d+)\.self_attn\.q_proj\.weight',
       r'layers.\1.attn.q_proj.w', ((1, 0), (D, H, Hd * 2))),
      (r'model\.language_model\.layers\.(\d+)\.self_attn\.k_proj\.weight',
       r'layers.\1.attn.k_proj.w', ((1, 0), (D, Hkv, Hd))),
      (r'model\.language_model\.layers\.(\d+)\.self_attn\.v_proj\.weight',
       r'layers.\1.attn.v_proj.w', ((1, 0), (D, Hkv, Hd))),
      (r'model\.language_model\.layers\.(\d+)\.self_attn\.o_proj\.weight',
       r'layers.\1.attn.o_proj.w', ((1, 0), (H, Hd, D))),
      (r'model\.language_model\.layers\.(\d+)\.self_attn\.q_norm\.weight',
       r'layers.\1.attn.q_norm.w', None),
      (r'model\.language_model\.layers\.(\d+)\.self_attn\.k_norm\.weight',
       r'layers.\1.attn.k_norm.w', None),
      (r'model\.language_model\.layers\.(\d+)\.linear_attn\.in_proj_qkv\.weight',
       r'layers.\1.linear_attn.in_proj_qkv.kernel', ((1, 0), None)),
      (r'model\.language_model\.layers\.(\d+)\.linear_attn\.in_proj_z\.weight',
       r'layers.\1.linear_attn.in_proj_z.kernel', ((1, 0), None)),
      (r'model\.language_model\.layers\.(\d+)\.linear_attn\.in_proj_b\.weight',
       r'layers.\1.linear_attn.in_proj_b.kernel', ((1, 0), None)),
      (r'model\.language_model\.layers\.(\d+)\.linear_attn\.in_proj_a\.weight',
       r'layers.\1.linear_attn.in_proj_a.kernel', ((1, 0), None)),
      (r'model\.language_model\.layers\.(\d+)\.linear_attn\.conv1d\.weight',
       r'layers.\1.linear_attn.conv1d_weight', None),
      (r'model\.language_model\.layers\.(\d+)\.linear_attn\.A_log',
       r'layers.\1.linear_attn.A_log', None),
      (r'model\.language_model\.layers\.(\d+)\.linear_attn\.dt_bias',
       r'layers.\1.linear_attn.dt_bias', None),
      (r'model\.language_model\.layers\.(\d+)\.linear_attn\.norm\.weight',
       r'layers.\1.linear_attn.norm.w', None),
      (r'model\.language_model\.layers\.(\d+)\.linear_attn\.out_proj\.weight',
       r'layers.\1.linear_attn.out_proj.kernel', ((1, 0), None)),
      (r'model\.language_model\.layers\.(\d+)\.mlp\.gate_proj\.weight',
       r'layers.\1.mlp.gate_proj.kernel', ((1, 0), None)),
      (r'model\.language_model\.layers\.(\d+)\.mlp\.up_proj\.weight',
       r'layers.\1.mlp.up_proj.kernel', ((1, 0), None)),
      (r'model\.language_model\.layers\.(\d+)\.mlp\.down_proj\.weight',
       r'layers.\1.mlp.down_proj.kernel', ((1, 0), None)),
      (r'model\.language_model\.norm\.weight', 'final_norm.w', None),
      (r'model\.language_model\.layers\.(\d+)\.input_layernorm\.weight',
       r'layers.\1.input_layernorm.w', None),
      (r'model\.language_model\.layers\.(\d+)\.post_attention_layernorm\.weight',
       r'layers.\1.post_attention_layernorm.w', None),
      (r'lm_head\.weight', 'lm_head.w', ((1, 0), None)),
  ]


def load_text_weights(model: qwen_model.Qwen3_5, ckpt_file: str) -> int:
  """Load LM weights from safetensors.

  Strategy: build a flat dict (tunix_key -> np.ndarray) from the safetensors,
  then use nnx.split / nnx.merge to populate the model correctly.

  The abs_state pure_dict has keys like ``embedder.input_embedding.value``
  (with single dot before value, not double) — we use path_to_key on the
  graph traversal to get exact keys, then look up with or without '.value'.
  """
  rules = _text_weight_rules(model.config)

  # Build tunix_key -> tensor mapping from safetensors
  loaded_tensors = {}
  with st.safe_open(ckpt_file, framework='pt') as sf:
    for hf_key in sf.keys():
      if 'visual' in hf_key:
        continue
      tensor = sf.get_tensor(hf_key).float().numpy()
      for pattern, tpl, transform in rules:
        if not re.fullmatch(pattern, hf_key):
          continue
        tunix_key = re.sub(pattern, tpl, hf_key)
        if transform is not None:
          perm, shape = transform
          if perm:
            tensor = tensor.transpose(perm)
          if shape:
            tensor = tensor.reshape(shape)
        loaded_tensors[tunix_key] = jnp.array(tensor)
        break

  # Use nnx.split to get the pure state dict, update it, then nnx.merge back.
  graph_def, state = nnx.split(model)
  state_dict = state.to_pure_dict()  # nested dict with values (no .value suffix)

  loaded = 0

  def _stoi(s):
    try:
      return int(s)
    except (ValueError, TypeError):
      return s

  for path, leaf in jax.tree_util.tree_leaves_with_path(state_dict):
    key = _path_to_key(path)
    bare = key[:-6] if key.endswith('.value') else key
    if bare in loaded_tensors:
      parts = key.split('.')
      d = state_dict
      for p in parts[:-1]:
        k_typed = _stoi(p)
        d = d[k_typed]
      last = _stoi(parts[-1])
      d[last] = loaded_tensors[bare]
      loaded += 1

  nnx.update(model, nnx.State(state_dict))
  return loaded


def load_vision_weights(vis_model: Qwen3_5VisionModel, ckpt_file: str) -> int:
  """Load vision encoder weights from safetensors."""
  loaded = 0
  with st.safe_open(ckpt_file, framework='pt') as sf:
    sd = {k[len('model.visual.'):]: sf.get_tensor(k).float()
          for k in sf.keys() if k.startswith('model.visual.')}

  for k, tensor in sd.items():
    np_t = tensor.numpy()
    if k == 'patch_embed.proj.weight':
      vis_model.patch_embed.proj.kernel.value = jnp.array(
          np_t.reshape(768, -1).T)
    elif k == 'patch_embed.proj.bias':
      vis_model.patch_embed.proj.bias.value = jnp.array(np_t)
    elif k == 'pos_embed.weight':
      vis_model.pos_embed.embedding.value = jnp.array(np_t)
    elif k.startswith('merger.'):
      parts = k.split('.')
      if parts[1] == 'norm':
        attr = 'scale' if parts[2] == 'weight' else 'bias'
        getattr(vis_model.merger.norm, attr).value = jnp.array(np_t)
      else:
        fc = 'fc1' if parts[1] == 'linear_fc1' else 'fc2'
        attr = 'kernel' if parts[2] == 'weight' else 'bias'
        v = np_t.T if attr == 'kernel' else np_t
        getattr(getattr(vis_model.merger, fc), attr).value = jnp.array(v)
    elif k.startswith('blocks.'):
      parts = k.split('.')
      i_b = int(parts[1])
      rest = '.'.join(parts[2:])
      b = vis_model.blocks[i_b]
      MAP = {
          'norm1.weight': ('norm1', 'scale'), 'norm1.bias': ('norm1', 'bias'),
          'norm2.weight': ('norm2', 'scale'), 'norm2.bias': ('norm2', 'bias'),
          'attn.qkv.bias': ('attn.qkv', 'bias'),
          'attn.proj.bias': ('attn.proj', 'bias'),
          'mlp.linear_fc1.bias': ('mlp.fc1', 'bias'),
          'mlp.linear_fc2.bias': ('mlp.fc2', 'bias'),
      }
      TMAP = {
          'attn.qkv.weight': ('attn.qkv', 'kernel'),
          'attn.proj.weight': ('attn.proj', 'kernel'),
          'mlp.linear_fc1.weight': ('mlp.fc1', 'kernel'),
          'mlp.linear_fc2.weight': ('mlp.fc2', 'kernel'),
      }
      if rest in MAP:
        path_str, attr = MAP[rest]
        obj = b
        for p in path_str.split('.'):
          obj = getattr(obj, p)
        getattr(obj, attr).value = jnp.array(np_t)
      elif rest in TMAP:
        path_str, attr = TMAP[rest]
        obj = b
        for p in path_str.split('.'):
          obj = getattr(obj, p)
        getattr(obj, attr).value = jnp.array(np_t.T)
      else:
        continue
    else:
      continue
    loaded += 1
  return loaded


# --------------------------------------------------------------------------
# Image preprocessing
# --------------------------------------------------------------------------

def preprocess_image(image_path: str):
  """Load and preprocess an image into patch tensors."""
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
  M = P * VIS_SPATIAL_MERGE
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
  arr = arr.transpose(0, 2, 4, 1, 3)
  arr = arr.reshape(H_grids * W_grids, 3, P, P)
  arr = np.stack([arr, arr], axis=2)
  arr = arr.reshape(H_grids * W_grids, 3 * 2 * P * P)

  pixel_patches = jnp.array(arr)
  grid_thw = jnp.array([[T, H_grids, W_grids]], dtype=jnp.int32)
  num_image_tokens = (H_grids * W_grids * T) // (VIS_SPATIAL_MERGE ** 2)
  print(f'  Image {W_orig}x{H_orig} -> {W_new}x{H_new}'
        f' -> {H_grids*W_grids} patches -> {num_image_tokens} visual tokens')
  return pixel_patches, grid_thw, num_image_tokens


# --------------------------------------------------------------------------
# VLM embedding builder
# --------------------------------------------------------------------------

def build_vlm_embeddings(lm_model, vis_model, input_ids, pixel_patches, grid_thw):
  """Replace IMAGE_TOKEN_ID positions with visual embeddings."""
  tokens = jnp.array(input_ids, dtype=jnp.int32)[None, :]
  text_embeds = lm_model.embedder.encode(tokens)
  visual_tokens = vis_model(pixel_patches, grid_thw)

  ids_arr = np.array(input_ids)
  image_positions = np.where(ids_arr == IMAGE_TOKEN_ID)[0]
  n_vis = min(len(image_positions), int(visual_tokens.shape[0]))
  if len(image_positions) != visual_tokens.shape[0]:
    print(f'  WARNING: {len(image_positions)} image pads but '
          f'{visual_tokens.shape[0]} visual vectors -- using {n_vis}')

  embeds = text_embeds
  for idx in range(n_vis):
    embeds = embeds.at[0, int(image_positions[idx])].set(visual_tokens[idx])
  return embeds


# --------------------------------------------------------------------------
# Greedy sampler
# --------------------------------------------------------------------------

def greedy_generate(lm_model, input_ids, max_new_tokens=200,
                    eos_token_id=151645, input_embeddings=None):
  """Greedy decoding with optional VLM visual embeddings."""
  prompt_len = len(input_ids)
  cache_size = prompt_len + max_new_tokens + 4

  tokens = jnp.array(input_ids, dtype=jnp.int32)[None, :]
  positions = jnp.arange(prompt_len, dtype=jnp.int32)[None, :]

  # 3D causal mask for prefill [1, prompt_len, cache_size]
  causal = jnp.tril(jnp.ones((prompt_len, prompt_len), dtype=jnp.bool_))
  attn_mask = jnp.pad(causal[None], ((0, 0), (0, 0), (0, cache_size - prompt_len)))

  cache = lm_model.init_cache(batch_size=1, cache_size=cache_size, dtype=jnp.float32)

  logits, cache = lm_model(
      input_tokens=tokens, positions=positions,
      cache=cache, attention_mask=attn_mask,
      input_embeddings=input_embeddings,
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
  parser = argparse.ArgumentParser(description='Qwen3.5-0.8B VLM inference')
  parser.add_argument('--prompt', default='The capital of France is')
  parser.add_argument('--image', default=None)
  parser.add_argument('--max_new_tokens', type=int, default=200)
  parser.add_argument('--ckpt', default=CKPT_FILE)
  args = parser.parse_args()

  print('[1] Loading tokenizer...')
  tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3.5-0.8B', trust_remote_code=True)

  print('[2] Building LM...')
  cfg = qwen_model.ModelConfig.qwen3_5_0p8b()
  lm = qwen_model.Qwen3_5(cfg, rngs=nnx.Rngs(0))

  print('[3] Loading LM weights...')
  n = load_text_weights(lm, args.ckpt)
  print(f'  Loaded {n} text tensors.')

  print('[4] Building vision encoder...')
  vcfg = VisionConfig.qwen3_5_0p8b()
  vis = Qwen3_5VisionModel(vcfg, rngs=nnx.Rngs(1))

  print('[5] Loading vision weights...')
  n_vis = load_vision_weights(vis, args.ckpt)
  print(f'  Loaded {n_vis} vision tensors.')

  input_embeddings = None
  if args.image:
    print(f'[6] Processing image: {args.image}')
    pixel_patches, grid_thw, n_img_toks = preprocess_image(args.image)
    image_placeholder = '<|vision_start|>' + '<|image_pad|>' * n_img_toks + '<|vision_end|>'
    messages = [{'role': 'user', 'content': image_placeholder + args.prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False,
        add_generation_prompt=True, enable_thinking=False)
    input_ids = tokenizer.encode(text)
    print(f'     Prompt: {len(input_ids)} tokens ({n_img_toks} image pads)')
    print('[7] Encoding image with ViT...')
    input_embeddings = build_vlm_embeddings(lm, vis, input_ids, pixel_patches, grid_thw)
  else:
    messages = [{'role': 'user', 'content': args.prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False,
        add_generation_prompt=True, enable_thinking=False)
    input_ids = tokenizer.encode(text)
    print(f'[6] Prompt: {len(input_ids)} tokens')

  print('[8] Generating...')
  generated_ids = greedy_generate(
      lm, input_ids, max_new_tokens=args.max_new_tokens,
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
