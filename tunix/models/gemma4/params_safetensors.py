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

"""Load Gemma 4 from HuggingFace safetensors checkpoints."""

import numpy as np

from tunix.models import safetensors_loader
from tunix.models.gemma4 import model as model_lib


# ---------------------------------------------------------------------------
# Key & transform mapping
# ---------------------------------------------------------------------------

def _get_key_and_transform_mapping(config: model_lib.ModelConfig):
  """Returns {hf_key: (tunix_key, transform | None)} for all model params.

  Transforms are (permute, reshape) tuples consumed by safetensors_loader.
  None means copy as-is.

  HuggingFace weight convention: Linear.weight is transposed vs JAX (out, in).
  We store Einsum weights as (in …, out …), so we transpose all 2-D linear
  kernels on load.
  """
  has_vision = config.vision_config is not None
  is_moe = config.num_experts > 1

  # Axes swap for a plain 2-D weight (out, in) → (in, out)
  T = ((1, 0), None)  # (permute, reshape)

  mapping = {}

  # --- Embedder ---
  tbase = 'model.language_model' if has_vision else 'model'
  mapping['model.language_model.embed_tokens.weight' if has_vision
          else 'model.embed_tokens.weight'] = ('embedder.input_embedding', None)
  mapping[f'{tbase}.norm.weight'] = ('final_norm.scale', None)

  # --- Text decoder layers ---
  for i in range(config.num_layers):
    lp = f'{tbase}.layers.{i}'
    tp = f'layers.{i}'
    is_global = (i % 6 == 5)

    mapping.update({
        f'{lp}.input_layernorm.weight':          (f'{tp}.pre_attention_norm.scale', None),
        f'{lp}.post_attention_layernorm.weight':  (f'{tp}.post_attention_norm.scale', None),
        f'{lp}.pre_feedforward_layernorm.weight': (f'{tp}.pre_ffw_norm.scale', None),
        f'{lp}.post_feedforward_layernorm.weight':(f'{tp}.post_ffw_norm.scale', None),
        f'{lp}.layer_scalar':                    (f'{tp}.layer_scalar', None),
        # Attention QKV
        f'{lp}.self_attn.q_proj.weight':(f'{tp}.attn.q_einsum.w', T),
        f'{lp}.self_attn.k_proj.weight':(f'{tp}.attn.kv_einsum.w', None),  # handled in preprocess
        f'{lp}.self_attn.v_proj.weight':(f'{tp}.attn.kv_einsum.w', None),  # handled in preprocess
        f'{lp}.self_attn.o_proj.weight':(f'{tp}.attn.attn_vec_einsum.w', None),  # preprocess
        # QKV norms
        f'{lp}.self_attn.q_norm.weight':(f'{tp}.attn._query_norm.scale', None),
        f'{lp}.self_attn.k_norm.weight':(f'{tp}.attn._key_norm.scale', None),
        f'{lp}.self_attn.v_norm.weight':(f'{tp}.attn._value_norm.scale', None),
    })

    if is_moe:
      mapping.update({
          f'{lp}.pre_feedforward_layernorm_2.weight': (f'{tp}.mlp.pre_feedforward_layernorm_2.scale', None),
          f'{lp}.post_feedforward_layernorm_1.weight':(f'{tp}.mlp.post_feedforward_layernorm_1.scale', None),
          f'{lp}.post_feedforward_layernorm_2.weight':(f'{tp}.mlp.post_feedforward_layernorm_2.scale', None),
          # Shared expert
          f'{lp}.mlp.gate_proj.weight':(f'{tp}.mlp.shared_gate.kernel', T),
          f'{lp}.mlp.up_proj.weight':  (f'{tp}.mlp.shared_up.kernel', T),
          f'{lp}.mlp.down_proj.weight':(f'{tp}.mlp.shared_down.kernel', T),
          # Router
          f'{lp}.router.proj.weight':  (f'{tp}.mlp.routed_experts.gate_kernel', T),
          f'{lp}.router.scale':        (f'{tp}.mlp.pre_forward_scale_2', None),
          # Routed experts: moe.gate_up_proj → split in preprocess
          f'{lp}.moe.gate_up_proj':    (f'{tp}.mlp.routed_experts._fused_gate_up', None),
          f'{lp}.moe.down_proj':       (f'{tp}.mlp.routed_experts._fused_down', None),
          f'{lp}.moe.per_expert_scale':(f'{tp}.mlp.routed_experts.per_expert_scale', None),
      })
    else:
      mapping.update({
          f'{lp}.mlp.gate_proj.weight':(f'{tp}.mlp.gate_proj.kernel', T),
          f'{lp}.mlp.up_proj.weight':  (f'{tp}.mlp.up_proj.kernel', T),
          f'{lp}.mlp.down_proj.weight':(f'{tp}.mlp.down_proj.kernel', T),
      })

  # --- Vision encoder ---
  if has_vision:
    vc = config.vision_config
    vp = 'vision_encoder'
    mapping.update({
        'model.vision_tower.patch_embedder.input_proj.weight':
            (f'{vp}.vision_entry.input_projection.kernel', T),
        'model.vision_tower.patch_embedder.position_embedding_table':
            (f'{vp}.vision_entry.pos_emb_param', None),
        f'model.vision_tower.std_bias': (f'{vp}.std_bias', None),
        f'model.vision_tower.std_scale':(f'{vp}.std_scale', None),
    })
    for i in range(vc.num_layers):
      hp = f'model.vision_tower.encoder.layers.{i}'
      tp2 = f'{vp}.layer_{i}'
      mapping.update({
          f'{hp}.input_layernorm.weight':          (f'{tp2}.pre_attention_norm.scale', None),
          f'{hp}.post_attention_layernorm.weight':  (f'{tp2}.post_attention_norm.scale', None),
          f'{hp}.pre_feedforward_layernorm.weight': (f'{tp2}.pre_ffw_norm.scale', None),
          f'{hp}.post_feedforward_layernorm.weight':(f'{tp2}.post_ffw_norm.scale', None),
          f'{hp}.self_attn.q_proj.linear.weight':  (f'{tp2}.attention.q_proj.kernel', T),
          f'{hp}.self_attn.k_proj.linear.weight':  (f'{tp2}.attention.k_proj.kernel', T),
          f'{hp}.self_attn.v_proj.linear.weight':  (f'{tp2}.attention.v_proj.kernel', T),
          f'{hp}.self_attn.o_proj.linear.weight':  (f'{tp2}.attention.o_proj.kernel', T),
          f'{hp}.self_attn.q_norm.weight':          (f'{tp2}.attention.q_norm.scale', None),
          f'{hp}.self_attn.k_norm.weight':          (f'{tp2}.attention.k_norm.scale', None),
          f'{hp}.mlp.gate_proj.linear.weight':     (f'{tp2}.mlp.gate.kernel', T),
          f'{hp}.mlp.up_proj.linear.weight':       (f'{tp2}.mlp.up.kernel', T),
          f'{hp}.mlp.down_proj.linear.weight':     (f'{tp2}.mlp.down.kernel', T),
      })

    # Vision projector
    mapping.update({
        'model.embed_vision.embedding_projection.weight':
            ('vision_projector.projection.kernel', T),
    })

  return mapping


# ---------------------------------------------------------------------------
# Preprocess: kv_einsum fusion, attn_vec reshape, MoE expert split
# ---------------------------------------------------------------------------

def _make_preprocess_fn(config: model_lib.ModelConfig):
  """Returns a function that post-processes the loaded state dict."""

  is_moe = config.num_experts > 1
  has_vision = config.vision_config is not None

  def preprocess(state_dict):
    for i in range(config.num_layers):
      is_global = (i % 6 == 5)
      head_dim   = config.global_head_dim if is_global else config.head_dim
      num_kv     = config.num_global_kv_heads if is_global else config.num_kv_heads

      kp = f'layers.{i}.attn'

      # --- Q einsum: HF (out, in) already transposed in mapping → reshape to (N, D, H) ---
      q_key = f'{kp}.q_einsum.w'
      if q_key in state_dict:
        w = state_dict[q_key]  # (embed_dim, num_heads*head_dim) after transpose
        state_dict[q_key] = w.reshape(
            config.embed_dim, config.num_heads, head_dim
        ).transpose(1, 0, 2)  # (N, D, H)

      # --- attn_vec_einsum: (embed_dim, num_heads*head_dim) → (N, H, D) ---
      o_key = f'{kp}.attn_vec_einsum.w'
      if o_key in state_dict:
        w = state_dict[o_key]  # (num_heads*head_dim, embed_dim) after HF transpose
        state_dict[o_key] = w.reshape(
            config.num_heads, head_dim, config.embed_dim
        )

      # --- KV einsum: fuse K (already transposed) and V (already transposed) ---
      k_key = f'{kp}.kv_einsum.w'
      k_src = k_key  # placeholder — both K and V are stored under same key via preprocess
      # The loader stores K and V separately as _k and _v, then we fuse here.
      k_tmp = f'{kp}._k_tmp'
      v_tmp = f'{kp}._v_tmp'
      if k_tmp in state_dict and v_tmp in state_dict:
        k = state_dict.pop(k_tmp)  # (embed_dim, num_kv*head_dim)
        v = state_dict.pop(v_tmp)
        # Reshape each to (num_kv, embed_dim, head_dim) then stack → (2, num_kv, D, H)
        k = k.reshape(config.embed_dim, num_kv, head_dim).transpose(1, 0, 2)
        v = v.reshape(config.embed_dim, num_kv, head_dim).transpose(1, 0, 2)
        state_dict[k_key] = np.stack([k, v], axis=0)  # (2, num_kv, D, H)

      # --- MoE: split fused gate_up → gate_proj_w and up_proj_w ---
      if is_moe:
        mp = f'layers.{i}.mlp.routed_experts'
        fused_key = f'{mp}._fused_gate_up'
        down_key  = f'{mp}._fused_down'
        if fused_key in state_dict:
          # HF shape: (num_experts, embed_dim, 2*expert_dim) — already in JAX convention
          fused = state_dict.pop(fused_key)
          gate, up = np.split(fused, 2, axis=-1)  # each (E, D, expert_dim)
          state_dict[f'{mp}.gate_proj_w'] = gate
          state_dict[f'{mp}.up_proj_w']   = up
        if down_key in state_dict:
          # HF shape: (num_experts, embed_dim, expert_dim) → we want (E, expert_dim, D)
          down = state_dict.pop(down_key)
          state_dict[f'{mp}.down_proj_w'] = np.transpose(down, (0, 2, 1))

    return state_dict

  return preprocess


# ---------------------------------------------------------------------------
# Revised mapping for the loader (with _tmp sentinel keys for K and V)
# ---------------------------------------------------------------------------

def _get_loader_mapping(config: model_lib.ModelConfig):
  """Mapping for the safetensors loader with _tmp keys for K/V before fusion."""
  m = _get_key_and_transform_mapping(config)
  T = ((1, 0), None)

  # Replace K and V kv_einsum.w entries with temporary keys
  for i in range(config.num_layers):
    tp = f'layers.{i}.attn'
    lp_k = (f'model.language_model.layers.{i}.self_attn.k_proj.weight'
            if config.vision_config is not None
            else f'model.layers.{i}.self_attn.k_proj.weight')
    lp_v = (f'model.language_model.layers.{i}.self_attn.v_proj.weight'
            if config.vision_config is not None
            else f'model.layers.{i}.self_attn.v_proj.weight')
    m[lp_k] = (f'{tp}._k_tmp', T)
    m[lp_v] = (f'{tp}._v_tmp', T)

  return m


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def create_model_from_safe_tensors(
    file_dir: str,
    config: model_lib.ModelConfig,
    mesh=None,
    dtype=None,
    mode: str = 'auto',
) -> model_lib.Gemma4:
  """Loads a Gemma 4 model from HuggingFace safetensors."""
  key_mapping = _get_loader_mapping
  preprocess_fn = _make_preprocess_fn(config)

  return safetensors_loader.load_and_create_model(
      file_dir=file_dir,
      model_class=model_lib.Gemma4,
      config=config,
      key_mapping=key_mapping,
      mesh=mesh,
      preprocess_fn=preprocess_fn,
      dtype=dtype,
      mode=mode,
  )
