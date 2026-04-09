"""Directly compare HF GDR vs our GDR on identical inputs."""
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

import jax, jax.numpy as jnp, numpy as np, torch
from flax import nnx, linen as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from tunix.models.qwen3_5 import model as m
from tunix.models.qwen3_5.gated_delta_rule import recurrent_gated_delta_rule, chunk_gated_delta_rule, l2norm

hf = AutoModelForCausalLM.from_pretrained('Qwen/Qwen3.5-0.8B', trust_remote_code=True).float()
hf.eval()
la = hf.model.layers[0].linear_attn  # HF linear-attn module

# Random [B=1, T=8, D=1024] hidden state
torch.manual_seed(42)
B, T, D = 1, 8, 1024
h_pt = torch.randn(B, T, D)
h_jnp = jnp.array(h_pt.numpy())

sd = {k: v.float() for k, v in hf.state_dict().items()}

# ===== HF forward (manual) =====
with torch.no_grad():
    # in_proj_qkv
    mixed_qkv = la.in_proj_qkv(h_pt).transpose(1, 2)    # [B, C, T]
    z = la.in_proj_z(h_pt).reshape(B, T, -1, la.head_v_dim)
    b = la.in_proj_b(h_pt)
    a = la.in_proj_a(h_pt)

    # conv1d (no fast kernel available, uses F.silu(conv(x)))
    conv_out = F.silu(la.conv1d(mixed_qkv)[:, :, :T])
    mixed_qkv_t = conv_out.transpose(1, 2)                 # [B, T, C]

    key_dim = la.key_dim    # 16*128=2048
    value_dim = la.value_dim # 16*128=2048
    q = mixed_qkv_t[..., :key_dim].reshape(B, T, -1, la.head_k_dim)   # [B,T,16,128]
    k = mixed_qkv_t[..., key_dim:2*key_dim].reshape(B, T, -1, la.head_k_dim)
    v = mixed_qkv_t[..., 2*key_dim:].reshape(B, T, -1, la.head_v_dim)

    beta = b.sigmoid()
    g = -la.A_log.float().exp() * F.softplus(a.float() + la.dt_bias.float())

    # GQA expand (num_v_heads=16, num_k_heads=16, ratio=1 so no expand)
    # chunk mode
    import inspect
    hf_chunk_fn = la.chunk_gated_delta_rule
    core_out, _ = hf_chunk_fn(q, k, v, g=g, beta=beta,
                               initial_state=None, output_final_state=False,
                               use_qk_l2norm_in_kernel=True)
    core_out = core_out.reshape(-1, la.head_v_dim)
    z2 = z.reshape(-1, la.head_v_dim)
    normed = la.norm(core_out, z2)
    out_hf = la.out_proj(normed.reshape(B, T, -1)).float().numpy()
print(f"HF linear-attn output: shape={out_hf.shape}  sample={out_hf[0,-1,:4]}")

# ===== Tunix forward =====
cfg = m.ModelConfig.qwen3_5_0p8b()
tm = m.Qwen3_5(cfg, rngs=nnx.Rngs(0))
tla = tm.layers[0].linear_attn

# Load weights for this one layer
def sv(path, val):
    obj = tla; parts = path.split('.')
    for p in parts[:-1]: obj = getattr(obj, p)
    getattr(obj, parts[-1]).value = jnp.array(val.numpy())

sv('in_proj_qkv.kernel', sd['model.layers.0.linear_attn.in_proj_qkv.weight'].T)
sv('in_proj_z.kernel',   sd['model.layers.0.linear_attn.in_proj_z.weight'].T)
sv('in_proj_b.kernel',   sd['model.layers.0.linear_attn.in_proj_b.weight'].T)
sv('in_proj_a.kernel',   sd['model.layers.0.linear_attn.in_proj_a.weight'].T)
tla.conv1d_weight.value = jnp.array(sd['model.layers.0.linear_attn.conv1d.weight'].numpy())
sv('A_log',   sd['model.layers.0.linear_attn.A_log'])
sv('dt_bias', sd['model.layers.0.linear_attn.dt_bias'])
sv('norm.w',  sd['model.layers.0.linear_attn.norm.weight'])
tla.out_proj.kernel.value = jnp.array(sd['model.layers.0.linear_attn.out_proj.weight'].T.numpy())

out_tx, _, _ = tla(h_jnp)
out_tx_np = np.array(out_tx)
print(f"TX linear-attn output: shape={out_tx_np.shape}  sample={out_tx_np[0,-1,:4]}")

diff = np.abs(out_hf - out_tx_np).max()
print(f"Linear-attn max diff: {diff:.6f}")

# Also compare intermediate tensors
# conv output comparison  
with torch.no_grad():
    mixed_qkv_hf_raw = la.in_proj_qkv(h_pt).transpose(1,2)
mixed_qkv_tx_raw = np.array(tla.in_proj_qkv(h_jnp).transpose((0,2,1)))
print(f"in_proj_qkv diff: {np.abs(mixed_qkv_hf_raw.numpy() - mixed_qkv_tx_raw).max():.6f}")

conv_hf = F.silu(la.conv1d(torch.tensor(mixed_qkv_hf_raw.numpy()))[:,:,:T]).numpy()
# conv tunix
mixed_qkv_tx_t = jnp.array(mixed_qkv_tx_raw)
conv_tx, _ = tla._causal_conv(mixed_qkv_tx_t, None)
print(f"conv1d output diff: {np.abs(conv_hf - np.array(conv_tx)).max():.6f}")
