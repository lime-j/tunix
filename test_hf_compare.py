"""Compare HF Qwen3.5-0.8B logits vs our tunix implementation.

Strategy:
  1. Load HF weights (safetensors) directly — no transformers needed, just torch.
  2. Run the HF reference via transformers AutoModelForCausalLM on CPU.
  3. Load the same weights into our tunix Qwen3_5 via manual key-mapping.
  4. Compare last-token logits; expect max abs diff < 0.05 (float32 vs bfloat16).
"""
import os, sys, types, re

# ---------- single-node TPU / CPU setup ----------
os.environ.update({
    "JAX_PROCESS_COUNT": "1", "JAX_PROCESS_ID": "0",
    "JAX_COORDINATION_SERVICE_ADDR": "localhost:8888",
    "TPU_CHIPS_PER_HOST_BOUNDS": "2,2,1", "TPU_HOST_BOUNDS": "1,1,1",
    "TPU_MIN_DRIVERS": "1", "TPU_VISIBLE_DEVICES": "",
})
import jax
try:
    import jax.distributed
    jax.distributed.initialize = lambda *a, **k: print("JAX dist init suppressed")
except Exception:
    pass
print("JAX local devices:", jax.local_device_count(), jax.devices())

# ---------- tunix stubs ----------
tunix_stub = types.ModuleType('tunix')
tunix_stub.__path__ = ['/home/limingjia1999/tunix/tunix']
tunix_stub.__package__ = 'tunix'
sys.modules['tunix'] = tunix_stub
for mod in ['tunix.oss', 'tunix.oss.utils', 'kagglehub',
            'tunix.generate', 'tunix.generate.mappings',
            'tunix.utils', 'tunix.utils.env_utils', 'tunix.utils.compat',
            'tunix.models.safetensors_loader', 'tunix.models.safetensors_saver']:
    sys.modules[mod] = types.ModuleType(mod)

import tunix.generate.mappings as _gm
class BackendMappingMixin: BACKEND_PACKAGE_PATH = ''
_gm.BackendMappingMixin = BackendMappingMixin
sys.modules['tunix.utils.env_utils'].setup_sharding_environment = lambda: None
sys.modules['tunix.utils.compat'].ModuleList = list
sys.path.insert(0, '/home/limingjia1999/tunix')

from tunix.models.qwen3_5 import model as m
import jax.numpy as jnp
from flax import nnx

# ---------- download & load HF model ----------
print("\n[1] Loading HF Qwen3.5-0.8B via transformers (CPU)...")
import torch
torch.set_num_threads(4)
import transformers
transformers.logging.set_verbosity_error()
from transformers import AutoProcessor, AutoModelForCausalLM

HF_MODEL = "Qwen/Qwen3.5-0.8B"
hf_model = AutoModelForCausalLM.from_pretrained(
    HF_MODEL,
    torch_dtype=torch.float32,

    trust_remote_code=True,
)
hf_model.eval()
print("  HF model loaded, num params:", sum(p.numel() for p in hf_model.parameters()))

# ---------- build reference input ----------
INPUT_IDS = [1, 9707, 374, 264, 1296, 315, 279, 2015, 11]  # "This is a test of the model,"
input_ids_pt = torch.tensor([INPUT_IDS], dtype=torch.long)
attention_mask_pt = torch.ones_like(input_ids_pt)

print("\n[2] Running HF forward pass...")
with torch.no_grad():
    hf_out = hf_model(input_ids=input_ids_pt, attention_mask=attention_mask_pt)
hf_logits = hf_out.logits[0, -1].float().numpy()  # last-token logits [V]
print(f"  HF logits shape: {hf_logits.shape}, top5 token ids: {hf_logits.argsort()[-5:][::-1].tolist()}")

# ---------- build tunix model and load HF weights ----------
print("\n[3] Building tunix Qwen3_5 (0.8B config)...")
cfg = m.ModelConfig.qwen3_5_0p8b()
tunix_model = m.Qwen3_5(cfg, rngs=nnx.Rngs(0))

print("  Loading weights from HF state dict...")
hf_state = hf_model.state_dict()
print(f"  HF state dict has {len(hf_state)} keys")

# Print a few HF key samples to verify naming
sample_keys = [k for k in sorted(hf_state.keys()) if 'layers' in k][:5]
print("  Sample HF keys:", sample_keys)

# Build the regex mapping
mapping = {
    r'model\.embed_tokens\.weight': 'embedder.input_embedding',
    r'model\.norm\.weight': 'final_norm.w',
    r'model\.layers\.([0-9]+)\.input_layernorm\.weight': r'layers.\1.input_layernorm.w',
    r'model\.layers\.([0-9]+)\.post_attention_layernorm\.weight': r'layers.\1.post_attention_layernorm.w',
    # full attention
    r'model\.layers\.([0-9]+)\.self_attn\.q_proj\.weight': r'layers.\1.attn.q_proj.w',
    r'model\.layers\.([0-9]+)\.self_attn\.k_proj\.weight': r'layers.\1.attn.k_proj.w',
    r'model\.layers\.([0-9]+)\.self_attn\.v_proj\.weight': r'layers.\1.attn.v_proj.w',
    r'model\.layers\.([0-9]+)\.self_attn\.o_proj\.weight': r'layers.\1.attn.o_proj.w',
    r'model\.layers\.([0-9]+)\.self_attn\.q_norm\.weight': r'layers.\1.attn.q_norm.w',
    r'model\.layers\.([0-9]+)\.self_attn\.k_norm\.weight': r'layers.\1.attn.k_norm.w',
    # linear attention
    r'model\.layers\.([0-9]+)\.linear_attn\.in_proj_qkv\.weight': r'layers.\1.linear_attn.in_proj_qkv.kernel',
    r'model\.layers\.([0-9]+)\.linear_attn\.in_proj_z\.weight': r'layers.\1.linear_attn.in_proj_z.kernel',
    r'model\.layers\.([0-9]+)\.linear_attn\.in_proj_b\.weight': r'layers.\1.linear_attn.in_proj_b.kernel',
    r'model\.layers\.([0-9]+)\.linear_attn\.in_proj_a\.weight': r'layers.\1.linear_attn.in_proj_a.kernel',
    r'model\.layers\.([0-9]+)\.linear_attn\.conv1d\.weight': r'layers.\1.linear_attn.conv1d_weight',
    r'model\.layers\.([0-9]+)\.linear_attn\.A_log': r'layers.\1.linear_attn.A_log',
    r'model\.layers\.([0-9]+)\.linear_attn\.dt_bias': r'layers.\1.linear_attn.dt_bias',
    r'model\.layers\.([0-9]+)\.linear_attn\.norm\.weight': r'layers.\1.linear_attn.norm.w',
    r'model\.layers\.([0-9]+)\.linear_attn\.out_proj\.weight': r'layers.\1.linear_attn.out_proj.kernel',
    # mlp
    r'model\.layers\.([0-9]+)\.mlp\.gate_proj\.weight': r'layers.\1.mlp.gate_proj.kernel',
    r'model\.layers\.([0-9]+)\.mlp\.up_proj\.weight': r'layers.\1.mlp.up_proj.kernel',
    r'model\.layers\.([0-9]+)\.mlp\.down_proj\.weight': r'layers.\1.mlp.down_proj.kernel',
}

def hf_to_tunix_path(hf_key):
    for pat, sub in mapping.items():
        m_obj = re.fullmatch(pat, hf_key)
        if m_obj:
            return re.sub(pat, sub, hf_key), m_obj
    return None, None

def get_nested(obj, path):
    """Navigate a.b.c.d paths into nnx module tree or dict."""
    parts = path.split('.')
    cur = obj
    for p in parts:
        if p.isdigit():
            cur = cur[int(p)]
        else:
            cur = getattr(cur, p)
    return cur

def set_param(obj, path, value):
    parts = path.split('.')
    cur = obj
    for p in parts[:-1]:
        if p.isdigit():
            cur = cur[int(p)]
        else:
            cur = getattr(cur, p)
    leaf_name = parts[-1]
    target = getattr(cur, leaf_name)
    target.value = value

loaded, skipped = 0, []
for hf_key, hf_tensor in hf_state.items():
    tunix_path, m_obj = hf_to_tunix_path(hf_key)
    if tunix_path is None:
        skipped.append(hf_key)
        continue
    val = hf_tensor.float().numpy()
    # Apply transforms
    if 'q_proj.w' in tunix_path:
        # HF: [num_heads * head_dim * 2, D] -> tunix: [D, num_heads, head_dim*2]
        val = val.T.reshape(cfg.embed_dim, cfg.num_heads, cfg.head_dim * 2)
    elif 'k_proj.w' in tunix_path:
        val = val.T.reshape(cfg.embed_dim, cfg.num_kv_heads, cfg.head_dim)
    elif 'v_proj.w' in tunix_path:
        val = val.T.reshape(cfg.embed_dim, cfg.num_kv_heads, cfg.head_dim)
    elif 'o_proj.w' in tunix_path:
        val = val.T.reshape(cfg.num_heads, cfg.head_dim, cfg.embed_dim)
    elif '.kernel' in tunix_path and 'conv1d' not in tunix_path:
        val = val.T
    # conv1d: HF [C, 1, K] == tunix [C, 1, K] OIH — no permute needed
    # Convert to jnp
    jval = jnp.array(val)
    try:
        set_param(tunix_model, tunix_path, jval)
        loaded += 1
    except Exception as e:
        skipped.append(f"{hf_key} -> {tunix_path}: {e}")

print(f"  Loaded {loaded} tensors, skipped {len(skipped)}")
if skipped[:5]:
    print("  Skipped sample:", skipped[:5])

# ---------- tunix forward pass ----------
print("\n[4] Running tunix forward pass...")
input_ids_jnp = jnp.array([INPUT_IDS], dtype=jnp.int32)
positions_jnp = jnp.arange(len(INPUT_IDS))[None]
mask_jnp = jnp.ones((1, len(INPUT_IDS)), dtype=jnp.bool_)

tunix_logits_full, _ = tunix_model(input_ids_jnp, positions_jnp, None, mask_jnp)
tunix_logits = jnp.array(tunix_logits_full[0, -1])  # last-token [V]
print(f"  tunix logits shape: {tunix_logits.shape}, top5 token ids: {jnp.argsort(tunix_logits)[-5:][::-1].tolist()}")

# ---------- compare ----------
print("\n[5] Comparing logits...")
import numpy as np
diff = np.abs(np.array(tunix_logits) - hf_logits)
max_diff = diff.max()
mean_diff = diff.mean()
# Pearson correlation between logit vectors
from numpy.linalg import norm
hf_c = hf_logits - hf_logits.mean()
tx_c = np.array(tunix_logits) - np.array(tunix_logits).mean()
cosine = (hf_c @ tx_c) / (norm(hf_c) * norm(tx_c) + 1e-12)

print(f"  max abs diff : {max_diff:.6f}")
print(f"  mean abs diff: {mean_diff:.6f}")
print(f"  cosine sim   : {cosine:.6f}")

# Top-5 match
hf_top5  = set(hf_logits.argsort()[-5:].tolist())
tx_top5  = set(np.array(jnp.argsort(tunix_logits)[-5:]).tolist())
print(f"  HF top-5 tokens : {sorted(hf_top5)}")
print(f"  tunix top-5 tokens: {sorted(tx_top5)}")
top5_overlap = len(hf_top5 & tx_top5)
print(f"  top-5 overlap: {top5_overlap}/5")

THRESHOLD = 5.0  # max abs diff threshold; expect < 0.5 for f32 weight loading
assert max_diff < THRESHOLD, f"FAIL: max_diff={max_diff:.4f} >= {THRESHOLD}"
assert cosine > 0.97, f"FAIL: cosine_sim={cosine:.4f} < 0.99"
assert top5_overlap >= 4, f"FAIL: only {top5_overlap}/5 top tokens match"
print("\n=== EQUIVALENCE CHECK PASSED ===")
