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

"""Utils for loading and converting Qwen3.5 HF safetensors weights."""

import re

import jax
import jax.numpy as jnp
from tunix.models import safetensors_loader
from tunix.models import safetensors_saver
from tunix.models.qwen3_5 import model as model_lib
from tunix.models.qwen3_5 import vision as vision_lib


def _get_key_and_transform_mapping(cfg: model_lib.ModelConfig):
  """Build the regex mapping from HF param names -> tunix param paths.

  HF Qwen3.5 checkpoint structure (abbreviated):
    model.embed_tokens.weight
    model.layers.N.self_attn.{q,k,v,o}_proj.weight
    model.layers.N.self_attn.{q,k}_norm.weight
    model.layers.N.mlp.{gate,up,down}_proj.weight
    # linear-attention layers:
    model.layers.N.linear_attn.in_proj_{qkv,z,b,a}.weight
    model.layers.N.linear_attn.conv1d.weight      (shape: [1, C, K])
    model.layers.N.linear_attn.A_log
    model.layers.N.linear_attn.dt_bias
    model.layers.N.linear_attn.norm.weight
    model.layers.N.linear_attn.out_proj.weight
    model.norm.weight
    lm_head.weight                      (absent when tie_word_embeddings=True)

  HF key prefix for this checkpoint is ``model.language_model.*``
  (Qwen3.5-0.8B is a VLM; the text tower lives under language_model).
  """
  mapping = {
      # Embeddings
      r'model\.language_model\.embed_tokens\.weight': ('embedder.input_embedding', None),

      # --- Full-attention layers ---
      r'model\.language_model\.layers\.([0-9]+)\.self_attn\.q_proj\.weight': (
          r'layers.\1.attn.q_proj.w',
          ((1, 0), (cfg.embed_dim, cfg.num_heads, cfg.head_dim * 2)),
      ),
      r'model\.language_model\.layers\.([0-9]+)\.self_attn\.k_proj\.weight': (
          r'layers.\1.attn.k_proj.w',
          ((1, 0), (cfg.embed_dim, cfg.num_kv_heads, cfg.head_dim)),
      ),
      r'model\.language_model\.layers\.([0-9]+)\.self_attn\.v_proj\.weight': (
          r'layers.\1.attn.v_proj.w',
          ((1, 0), (cfg.embed_dim, cfg.num_kv_heads, cfg.head_dim)),
      ),
      r'model\.language_model\.layers\.([0-9]+)\.self_attn\.o_proj\.weight': (
          r'layers.\1.attn.o_proj.w',
          ((1, 0), (cfg.num_heads, cfg.head_dim, cfg.embed_dim)),
      ),
      r'model\.language_model\.layers\.([0-9]+)\.self_attn\.q_norm\.weight': (
          r'layers.\1.attn.q_norm.w',
          None,
      ),
      r'model\.language_model\.layers\.([0-9]+)\.self_attn\.k_norm\.weight': (
          r'layers.\1.attn.k_norm.w',
          None,
      ),

      # --- Linear-attention (GatedDeltaNet) layers ---
      r'model\.language_model\.layers\.([0-9]+)\.linear_attn\.in_proj_qkv\.weight': (
          r'layers.\1.linear_attn.in_proj_qkv.kernel',
          ((1, 0), None),
      ),
      r'model\.language_model\.layers\.([0-9]+)\.linear_attn\.in_proj_z\.weight': (
          r'layers.\1.linear_attn.in_proj_z.kernel',
          ((1, 0), None),
      ),
      r'model\.language_model\.layers\.([0-9]+)\.linear_attn\.in_proj_b\.weight': (
          r'layers.\1.linear_attn.in_proj_b.kernel',
          ((1, 0), None),
      ),
      r'model\.language_model\.layers\.([0-9]+)\.linear_attn\.in_proj_a\.weight': (
          r'layers.\1.linear_attn.in_proj_a.kernel',
          ((1, 0), None),
      ),
      # conv1d: HF stores [C, 1, K] = OIH layout — load as-is, no permute
      r'model\.language_model\.layers\.([0-9]+)\.linear_attn\.conv1d\.weight': (
          r'layers.\1.linear_attn.conv1d_weight',
          None,
      ),
      r'model\.language_model\.layers\.([0-9]+)\.linear_attn\.A_log': (
          r'layers.\1.linear_attn.A_log',
          None,
      ),
      r'model\.language_model\.layers\.([0-9]+)\.linear_attn\.dt_bias': (
          r'layers.\1.linear_attn.dt_bias',
          None,
      ),
      r'model\.language_model\.layers\.([0-9]+)\.linear_attn\.norm\.weight': (
          r'layers.\1.linear_attn.norm.w',
          None,
      ),
      r'model\.language_model\.layers\.([0-9]+)\.linear_attn\.out_proj\.weight': (
          r'layers.\1.linear_attn.out_proj.kernel',
          ((1, 0), None),
      ),

      # --- MLP ---
      r'model\.language_model\.layers\.([0-9]+)\.mlp\.gate_proj\.weight': (
          r'layers.\1.mlp.gate_proj.kernel',
          ((1, 0), None),
      ),
      r'model\.language_model\.layers\.([0-9]+)\.mlp\.up_proj\.weight': (
          r'layers.\1.mlp.up_proj.kernel',
          ((1, 0), None),
      ),
      r'model\.language_model\.layers\.([0-9]+)\.mlp\.down_proj\.weight': (
          r'layers.\1.mlp.down_proj.kernel',
          ((1, 0), None),
      ),

      # --- Layer norms ---
      r'model\.language_model\.norm\.weight': ('final_norm.w', None),
      r'model\.language_model\.layers\.([0-9]+)\.input_layernorm\.weight': (
          r'layers.\1.input_layernorm.w',
          None,
      ),
      r'model\.language_model\.layers\.([0-9]+)\.post_attention_layernorm\.weight': (
          r'layers.\1.post_attention_layernorm.w',
          None,
      ),

      # --- LM head (absent when tie_word_embeddings=True) ---
      r'lm_head\.weight': ('lm_head.w', ((1, 0), None)),
  }
  return mapping


def _get_vision_key_mapping():
  """Regex mapping from safetensors vision keys to tunix vision_model paths.

  Checkpoint key prefix: ``model.visual.*``
  """
  return {
      # Patch embedding (Conv3d stored as [out, in, T, H, W])
      # We load the kernel and reshape to [patch_dim, embed_dim] for Linear.
      # patch_dim = in_ch * T * H * W = 3 * 2 * 16 * 16 = 1536
      r'model\.visual\.patch_embed\.proj\.weight': (
          'patch_embed.proj.kernel',
          'reshape_conv3d',  # special transform tag
      ),
      r'model\.visual\.patch_embed\.proj\.bias': (
          'patch_embed.proj.bias', None
      ),
      # Learned position embeddings
      r'model\.visual\.pos_embed\.weight': ('pos_embed.embedding', None),
      # ViT blocks
      r'model\.visual\.blocks\.([0-9]+)\.norm1\.weight': (
          r'blocks.\1.norm1.scale', None
      ),
      r'model\.visual\.blocks\.([0-9]+)\.norm1\.bias': (
          r'blocks.\1.norm1.bias', None
      ),
      r'model\.visual\.blocks\.([0-9]+)\.norm2\.weight': (
          r'blocks.\1.norm2.scale', None
      ),
      r'model\.visual\.blocks\.([0-9]+)\.norm2\.bias': (
          r'blocks.\1.norm2.bias', None
      ),
      r'model\.visual\.blocks\.([0-9]+)\.attn\.qkv\.weight': (
          r'blocks.\1.attn.qkv.kernel', ((1, 0), None)
      ),
      r'model\.visual\.blocks\.([0-9]+)\.attn\.qkv\.bias': (
          r'blocks.\1.attn.qkv.bias', None
      ),
      r'model\.visual\.blocks\.([0-9]+)\.attn\.proj\.weight': (
          r'blocks.\1.attn.proj.kernel', ((1, 0), None)
      ),
      r'model\.visual\.blocks\.([0-9]+)\.attn\.proj\.bias': (
          r'blocks.\1.attn.proj.bias', None
      ),
      r'model\.visual\.blocks\.([0-9]+)\.mlp\.linear_fc1\.weight': (
          r'blocks.\1.mlp.fc1.kernel', ((1, 0), None)
      ),
      r'model\.visual\.blocks\.([0-9]+)\.mlp\.linear_fc1\.bias': (
          r'blocks.\1.mlp.fc1.bias', None
      ),
      r'model\.visual\.blocks\.([0-9]+)\.mlp\.linear_fc2\.weight': (
          r'blocks.\1.mlp.fc2.kernel', ((1, 0), None)
      ),
      r'model\.visual\.blocks\.([0-9]+)\.mlp\.linear_fc2\.bias': (
          r'blocks.\1.mlp.fc2.bias', None
      ),
      # Patch merger
      r'model\.visual\.merger\.norm\.weight': ('merger.norm.scale', None),
      r'model\.visual\.merger\.norm\.bias': ('merger.norm.bias', None),
      r'model\.visual\.merger\.linear_fc1\.weight': (
          'merger.fc1.kernel', ((1, 0), None)
      ),
      r'model\.visual\.merger\.linear_fc1\.bias': ('merger.fc1.bias', None),
      r'model\.visual\.merger\.linear_fc2\.weight': (
          'merger.fc2.kernel', ((1, 0), None)
      ),
      r'model\.visual\.merger\.linear_fc2\.bias': ('merger.fc2.bias', None),
  }


def create_model_from_safe_tensors(
    file_dir: str,
    config: model_lib.ModelConfig,
    mesh: jax.sharding.Mesh | None = None,
    dtype: jnp.dtype | None = None,
    mode: str = 'auto',
) -> model_lib.Qwen3_5:
  """Load tensors from the safetensors file and create a Qwen3_5 model."""
  return safetensors_loader.load_and_create_model(
      file_dir=file_dir,
      model_class=model_lib.Qwen3_5,
      config=config,
      key_mapping=_get_key_and_transform_mapping,
      mesh=mesh,
      dtype=dtype,
      mode=mode,
  )


def create_vision_model_from_safe_tensors(
    file_dir: str,
    config: vision_lib.VisionConfig,
    mesh: jax.sharding.Mesh | None = None,
    dtype: jnp.dtype | None = None,
) -> vision_lib.Qwen3_5VisionModel:
  """Load vision encoder weights from the safetensors file.

  Handles the ``reshape_conv3d`` special transform for the patch-embed kernel:
  HF stores it as [out_ch=768, in_ch=3, T=2, H=16, W=16]; we reshape to
  [1536, 768] to match the ``nnx.Linear`` kernel layout [in_features, out_features].
  """
  def _preprocess_fn(loaded: dict) -> dict:
    key = 'patch_embed.proj.kernel'
    if key in loaded:
      w = loaded[key]
      # HF Conv3d: [768, 3, 2, 16, 16] → transpose → [3,2,16,16,768] → flatten → [1536, 768]
      # The mapping already transposed via ((1,0), None), giving [3*2*16*16, 768] = [1536, 768]
      # but safetensors_loader applies (perm, reshape); here transform='reshape_conv3d'
      # means we need to do it manually because the loader can't handle the 5D→2D case.
      # At this point the array is [768, 3, 2, 16, 16] (raw, before any transform).
      loaded[key] = w.reshape(768, -1).T  # [1536, 768]
    return loaded

  # Build a wrapped key_mapping that replaces 'reshape_conv3d' tags with
  # a normal None transform (the actual reshape is done in preprocess_fn).
  raw_map = _get_vision_key_mapping()

  def _key_mapping(_cfg):
    result = {}
    for pattern, (tunix_key, transform) in raw_map.items():
      result[pattern] = (tunix_key, None if transform == 'reshape_conv3d' else transform)
    return result

  return safetensors_loader.load_and_create_model(
      file_dir=file_dir,
      model_class=vision_lib.Qwen3_5VisionModel,
      config=config,
      key_mapping=_key_mapping,
      mesh=mesh,
      dtype=dtype,
      preprocess_fn=_preprocess_fn,
      mode='optimized',
  )


def _qwen3_5_state_key_to_safetensors_key(lora_name: str) -> str:
  """Transform tunix layer path to HF safetensors state dict key."""
  key = f'model.{lora_name}.weight'
  key = key.replace('.attn.', '.self_attn.')
  key = key.replace('.linear_attn.', '.linear_attn.')
  key = re.sub(r'\.(gate|up|down)_proj\.kernel$', r'.\1_proj.weight', key)
  return key


_QWEN3_5_HUGGINGFACE_TRANSPOSE_RULES = {
    'q_proj': (1, 0),
    'k_proj': (1, 0),
    'v_proj': (1, 0),
    'o_proj': (1, 0),
    'up_proj': (1, 0),
    'down_proj': (1, 0),
    'gate_proj': (1, 0),
    'out_proj': (1, 0),
    'in_proj_qkv': (1, 0),
    'in_proj_z': (1, 0),
    'in_proj_b': (1, 0),
    'in_proj_a': (1, 0),
}


def save_lora_merged_model_as_safetensors(
    local_model_path: str,
    output_dir: str,
    lora_model: model_lib.Qwen3_5,
    rank: int,
    alpha: float,
):
  """Saves a Qwen3_5 model with LoRA weights merged in safetensors format."""
  safetensors_saver.save_lora_merged_model_as_safetensors(
      local_model_path=local_model_path,
      output_dir=output_dir,
      lora_model=lora_model,
      rank=rank,
      alpha=alpha,
      state_key_transform_fn=_qwen3_5_state_key_to_safetensors_key,
      transpose_rules=_QWEN3_5_HUGGINGFACE_TRANSPOSE_RULES,
  )
