import os, sys, types
os.environ.update({'JAX_PROCESS_COUNT':'1','JAX_PROCESS_ID':'0','TPU_VISIBLE_DEVICES':'',
                   'JAX_COORDINATION_SERVICE_ADDR':'localhost:8888','TPU_CHIPS_PER_HOST_BOUNDS':'2,2,1',
                   'TPU_HOST_BOUNDS':'1,1,1'})

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

print("Building tunix model...")
cfg = m.ModelConfig.qwen3_5_0p8b()
tm = m.Qwen3_5(cfg, rngs=nnx.Rngs(0))

def load(path, key): 
    val = jnp.array(sd[key].numpy())
    obj = tm; parts = path.split('.')
    for p in parts[:-1]:
        obj = obj[int(p)] if p.isdigit() else getattr(obj, p)
    attr = getattr(obj, parts[-1])
    attr.value = val

# Load all text weights
load('embedder.input_embedding', 'model.embed_tokens.weight')
load('final_norm.w', 'model.norm.weight')
for i in range(24):
    load(f'layers.{i}.input_layernorm.w',         f'model.layers.{i}.input_layernorm.weight')
    load(f'layers.{i}.post_attention_layernorm.w', f'model.layers.{i}.post_attention_layernorm.weight')
    lt = cfg.layer_types[i]
    if lt == 'full_attention':
        qw = sd[f'model.layers.{i}.self_attn.q_proj.weight'].numpy().T.reshape(cfg.embed_dim, cfg.num_heads, cfg.head_dim*2)
        tm.layers[i].attn.q_proj.w.value = jnp.array(qw)
        kw = sd[f'model.layers.{i}.self_attn.k_proj.weight'].numpy().T.reshape(cfg.embed_dim, cfg.num_kv_heads, cfg.head_dim)
        tm.layers[i].attn.k_proj.w.value = jnp.array(kw)
        vw = sd[f'model.layers.{i}.self_attn.v_proj.weight'].numpy().T.reshape(cfg.embed_dim, cfg.num_kv_heads, cfg.head_dim)
        tm.layers[i].attn.v_proj.w.value = jnp.array(vw)
        ow = sd[f'model.layers.{i}.self_attn.o_proj.weight'].numpy().T.reshape(cfg.num_heads, cfg.head_dim, cfg.embed_dim)
        tm.layers[i].attn.o_proj.w.value = jnp.array(ow)
        load(f'layers.{i}.attn.q_norm.w', f'model.layers.{i}.self_attn.q_norm.weight')
        load(f'layers.{i}.attn.k_norm.w', f'model.layers.{i}.self_attn.k_norm.weight')
    else:
        for p in ['in_proj_qkv','in_proj_z','in_proj_b','in_proj_a']:
            wt = sd[f'model.layers.{i}.linear_attn.{p}.weight'].numpy().T
            attr = getattr(tm.layers[i].linear_attn, p)
            attr.kernel.value = jnp.array(wt)
        tm.layers[i].linear_attn.conv1d_weight.value = jnp.array(sd[f'model.layers.{i}.linear_attn.conv1d.weight'].numpy())
        tm.layers[i].linear_attn.A_log.value = jnp.array(sd[f'model.layers.{i}.linear_attn.A_log'].numpy())
        tm.layers[i].linear_attn.dt_bias.value = jnp.array(sd[f'model.layers.{i}.linear_attn.dt_bias'].numpy())
        tm.layers[i].linear_attn.norm.w.value = jnp.array(sd[f'model.layers.{i}.linear_attn.norm.weight'].numpy())
        out_p = sd[f'model.layers.{i}.linear_attn.out_proj.weight'].numpy().T
        tm.layers[i].linear_attn.out_proj.kernel.value = jnp.array(out_p)
    for ml in ['gate_proj','up_proj','down_proj']:
        attr = getattr(tm.layers[i].mlp, ml)
        attr.kernel.value = jnp.array(sd[f'model.layers.{i}.mlp.{ml}.weight'].numpy().T)

print("Weights loaded. Running HF forward...")
INPUT_IDS = [1, 9707, 374, 264, 1296, 315, 279, 2015, 11]
toks_pt = torch.tensor([INPUT_IDS])
with torch.no_grad():
    hf_out = hf(input_ids=toks_pt, attention_mask=torch.ones_like(toks_pt))
hf_logits = hf_out.logits[0, -1].float().numpy()
print(f"HF  top5: {hf_logits.argsort()[-5:][::-1].tolist()}")

print("Running tunix forward...")
toks_jnp = jnp.array([INPUT_IDS])
pos_jnp = jnp.arange(len(INPUT_IDS))[None]
mask_jnp = jnp.ones((1, len(INPUT_IDS)), dtype=jnp.bool_)
tx_logits_full, _ = tm(toks_jnp, pos_jnp, None, mask_jnp)
tx_logits = np.array(tx_logits_full[0, -1])
print(f"TX  top5: {tx_logits.argsort()[-5:][::-1].tolist()}")

diff = np.abs(tx_logits - hf_logits)
cosim = np.dot(tx_logits - tx_logits.mean(), hf_logits - hf_logits.mean()) / \
        (np.linalg.norm(tx_logits - tx_logits.mean()) * np.linalg.norm(hf_logits - hf_logits.mean()) + 1e-12)
print(f"max_diff={diff.max():.4f}  mean_diff={diff.mean():.4f}  cosine={cosim:.6f}")
top5_overlap = len(set(tx_logits.argsort()[-5:]) & set(hf_logits.argsort()[-5:]))
print(f"top5 overlap: {top5_overlap}/5")
