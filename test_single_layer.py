"""Pinpoint divergence by running HF layer-by-layer and comparing hiddens."""
import os, sys, types
os.environ.update({'JAX_PROCESS_COUNT':'1','JAX_PROCESS_ID':'0','TPU_VISIBLE_DEVICES':'',
                   'JAX_COORDINATION_SERVICE_ADDR':'localhost:8888'})

tunix_stub = types.ModuleType('tunix'); tunix_stub.__path__ = ['/home/limingjia1999/tunix/tunix']
tunix_stub.__package__ = 'tunix'; sys.modules['tunix'] = tunix_stub
for mod in ['tunix.oss','tunix.oss.utils','kagglehub','tunix.generate','tunix.generate.mappings',
            'tunix.utils','tunix.utils.env_utils','tunix.utils.compat',
            'tunix.models.safetensors_loader','tunix.models.safetensors_saver']:
    sys.modules[mod] = types.ModuleType(mod)
import tunix.generate.mappings as _gm
class BMM: BACKEND_PACKAGE_PATH = ''
_gm.BackendMappingMixin = BMM
sys.modules['tunix.utils.env_utils'].setup_sharding_environment = lambda: None
sys.modules['tunix.utils.compat'].ModuleList = list
sys.path.insert(0, '/home/limingjia1999/tunix')

import jax, jax.numpy as jnp
import torch, numpy as np
from flax import nnx
from transformers import AutoModelForCausalLM
from tunix.models.qwen3_5 import model as m

print("Loading HF model...")
hf = AutoModelForCausalLM.from_pretrained('Qwen/Qwen3.5-0.8B', trust_remote_code=True).float()
hf.eval()
sd = {k: v.float() for k, v in hf.state_dict().items()}

cfg = m.ModelConfig.qwen3_5_0p8b()
tm = m.Qwen3_5(cfg, rngs=nnx.Rngs(0))

def set_val(module_path, val_np):
    obj = tm; parts = module_path.split('.')
    for p in parts[:-1]: obj = obj[int(p)] if p.isdigit() else getattr(obj, p)
    getattr(obj, parts[-1]).value = jnp.array(val_np)

# Minimal load: just what we need for layer-by-layer comparison
set_val('embedder.input_embedding', sd['model.embed_tokens.weight'].numpy())
for i in range(24):
    set_val(f'layers.{i}.input_layernorm.w', sd[f'model.layers.{i}.input_layernorm.weight'].numpy())
    set_val(f'layers.{i}.post_attention_layernorm.w', sd[f'model.layers.{i}.post_attention_layernorm.weight'].numpy())
    lt = cfg.layer_types[i]
    if lt == 'full_attention':
        set_val(f'layers.{i}.attn.q_proj.w', sd[f'model.layers.{i}.self_attn.q_proj.weight'].numpy().T.reshape(cfg.embed_dim, cfg.num_heads, cfg.head_dim*2))
        set_val(f'layers.{i}.attn.k_proj.w', sd[f'model.layers.{i}.self_attn.k_proj.weight'].numpy().T.reshape(cfg.embed_dim, cfg.num_kv_heads, cfg.head_dim))
        set_val(f'layers.{i}.attn.v_proj.w', sd[f'model.layers.{i}.self_attn.v_proj.weight'].numpy().T.reshape(cfg.embed_dim, cfg.num_kv_heads, cfg.head_dim))
        set_val(f'layers.{i}.attn.o_proj.w', sd[f'model.layers.{i}.self_attn.o_proj.weight'].numpy().T.reshape(cfg.num_heads, cfg.head_dim, cfg.embed_dim))
        set_val(f'layers.{i}.attn.q_norm.w', sd[f'model.layers.{i}.self_attn.q_norm.weight'].numpy())
        set_val(f'layers.{i}.attn.k_norm.w', sd[f'model.layers.{i}.self_attn.k_norm.weight'].numpy())
    else:
        for p in ['in_proj_qkv','in_proj_z','in_proj_b','in_proj_a']:
            wt = sd[f'model.layers.{i}.linear_attn.{p}.weight'].numpy().T
            getattr(tm.layers[i].linear_attn, p).kernel.value = jnp.array(wt)
        tm.layers[i].linear_attn.conv1d_weight.value = jnp.array(sd[f'model.layers.{i}.linear_attn.conv1d.weight'].numpy())
        set_val(f'layers.{i}.linear_attn.A_log', sd[f'model.layers.{i}.linear_attn.A_log'].numpy())
        set_val(f'layers.{i}.linear_attn.dt_bias', sd[f'model.layers.{i}.linear_attn.dt_bias'].numpy())
        set_val(f'layers.{i}.linear_attn.norm.w', sd[f'model.layers.{i}.linear_attn.norm.weight'].numpy())
        tm.layers[i].linear_attn.out_proj.kernel.value = jnp.array(sd[f'model.layers.{i}.linear_attn.out_proj.weight'].numpy().T)
    for ml in ['gate_proj','up_proj','down_proj']:
        getattr(tm.layers[i].mlp, ml).kernel.value = jnp.array(sd[f'model.layers.{i}.mlp.{ml}.weight'].numpy().T)
set_val('final_norm.w', sd['model.norm.weight'].numpy())

INPUT_IDS = [1, 9707, 374, 264, 1296, 315, 279, 2015, 11]
positions = list(range(len(INPUT_IDS)))

# Run HF step by step
toks_pt = torch.tensor([INPUT_IDS])
mask_pt = torch.ones(1, len(INPUT_IDS))
pos_pt = torch.tensor([positions])

with torch.no_grad():
    hf_h = hf.model.embed_tokens(toks_pt).float()  # [1, T, D]
    # Build causal mask from HF internals
    hf_internal = hf.model
    # get position embeddings from HF
    pos_emb = hf_internal.rotary_emb(hf_h, pos_pt)

# Run tunix embedding
pos_jnp = jnp.array([positions])
mask_jnp = jnp.ones((1, len(INPUT_IDS)), dtype=jnp.bool_)
tx_h = np.array(tm.embedder.encode(jnp.array([INPUT_IDS])))
hf_h_np = hf_h.numpy()
print(f"After embed: max_diff={np.abs(hf_h_np - tx_h).max():.8f}")

# Step through layers comparing
for i in range(min(6, 24)):
    lt = cfg.layer_types[i]
    with torch.no_grad():
        # HF layer i
        hf_h_t = torch.tensor(hf_h_np)
        # Build minimal causal attention mask
        T = hf_h_np.shape[1]
        # HF expects 4D causal mask through sliding window or regular
        out = hf_internal.layers[i](
            hf_h_t,
            attention_mask=None,
            position_ids=pos_pt,
            position_embeddings=pos_emb,
        )
        hf_h_new = out[0].float().numpy()
    # tunix layer i
    tx_h_j = jnp.array(hf_h_np)
    # Tunix layer
    _, tx_h_new_j = tm.layers[i](tx_h_j, pos_jnp, None, mask_jnp)
    tx_h_new = np.array(tx_h_new_j)
    diff = np.abs(hf_h_new - tx_h_new).max()
    print(f"After layer {i:2d} ({lt[:4]}): max_diff={diff:.6f}")
    hf_h_np = hf_h_new
