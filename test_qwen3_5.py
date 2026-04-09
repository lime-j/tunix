import os, sys, types

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
print("Local Devices:", jax.local_device_count())

tunix_stub = types.ModuleType('tunix')
tunix_stub.__path__ = ['/home/limingjia1999/tunix/tunix']
tunix_stub.__package__ = 'tunix'
sys.modules['tunix'] = tunix_stub
for mod in ['tunix.oss','tunix.oss.utils','kagglehub','tunix.generate','tunix.generate.mappings',
            'tunix.utils','tunix.utils.env_utils','tunix.utils.compat',
            'tunix.models.safetensors_loader','tunix.models.safetensors_saver']:
    sys.modules[mod] = types.ModuleType(mod)

import tunix.generate.mappings as _gm
class BackendMappingMixin: BACKEND_PACKAGE_PATH = ''
_gm.BackendMappingMixin = BackendMappingMixin
sys.modules['tunix.utils.env_utils'].setup_sharding_environment = lambda: None
sys.modules['tunix.utils.compat'].ModuleList = list
sys.path.insert(0, '/home/limingjia1999/tunix')

from tunix.models.qwen3_5.gated_delta_rule import recurrent_gated_delta_rule, chunk_gated_delta_rule
from tunix.models.qwen3_5 import model as m
from tunix.models import naming
import jax.numpy as jnp
from flax import nnx
print("=== Imports OK ===")

for label, cfg in [('1.5b', m.ModelConfig.qwen3_5_1p5b()), ('7b', m.ModelConfig.qwen3_5_7b())]:
    n_lin = sum(1 for lt in cfg.layer_types if lt == 'linear_attention')
    print(f'  {label}: layers={cfg.num_layers}, embed={cfg.embed_dim}, full={cfg.num_layers-n_lin}, linear={n_lin}')

n = naming.ModelNaming(model_name='qwen3.5-7b')
assert n.model_family == 'qwen3p5' and n.model_config_category == 'qwen3_5' and n.model_config_id == 'qwen3p5_7b'
print(f"  Naming: {n.model_family}/{n.model_config_category}/{n.model_config_id}  PASS")

B,T,H,D = 2,16,4,8
q=jax.random.normal(jax.random.PRNGKey(1),(B,T,H,D)); k=jax.random.normal(jax.random.PRNGKey(0),(B,T,H,D))
v=jax.random.normal(jax.random.PRNGKey(2),(B,T,H,D)); g=jax.random.normal(jax.random.PRNGKey(3),(B,T,H))*0.1-1.
beta=jax.nn.sigmoid(jax.random.normal(jax.random.PRNGKey(4),(B,T,H)))
out_rec,_ = recurrent_gated_delta_rule(q,k,v,g,beta)
out_chk,_ = chunk_gated_delta_rule(q,k,v,g,beta,chunk_size=8)
assert out_rec.shape==(B,T,H,D) and out_chk.shape==(B,T,H,D)
diff = float(jnp.max(jnp.abs(out_rec-out_chk)))
print(f"  GDR recurrent vs chunked max_diff={diff:.6f}  PASS")
assert diff < 1e-3, f"GDR diverge: {diff}"

tiny = m.ModelConfig(num_layers=4,vocab_size=256,embed_dim=64,hidden_dim=128,
    num_heads=4,head_dim=16,num_kv_heads=2,rope_theta=10000,norm_eps=1e-6,
    partial_rotary_factor=0.5,linear_num_key_heads=2,linear_num_value_heads=2,
    linear_key_head_dim=16,linear_value_head_dim=16,linear_conv_kernel_dim=4,
    use_tied_embedding=True,dtype=jnp.float32,param_dtype=jnp.float32)
print(f"  tiny layer_types: {tiny.layer_types}")
mdl = m.Qwen3_5(tiny, rngs=nnx.Rngs(0))
B2,T2 = 2,8
toks=jnp.ones((B2,T2),dtype=jnp.int32); pos=jnp.broadcast_to(jnp.arange(T2)[None],(B2,T2))
mask=jnp.ones((B2,T2),dtype=jnp.bool_)
logits,nc = mdl(toks,pos,None,mask)
assert logits.shape==(B2,T2,256) and nc is None
print(f"  Forward no-cache: {logits.shape}  PASS")

cache = mdl.init_cache(B2,cache_size=16,dtype=jnp.float32)
logits2,nc2 = mdl(toks,pos,cache,mask)
assert logits2.shape==(B2,T2,256) and nc2 is not None and len(nc2)==tiny.num_layers
print(f"  Forward with-cache: {logits2.shape}  layers={len(nc2)}  PASS")

mask3=jnp.ones((B2,T2,T2),dtype=jnp.bool_)
logits3,_=mdl(toks,pos,None,mask3)
assert logits3.shape==(B2,T2,256)
print(f"  Forward 3D-mask: {logits3.shape}  PASS")

print("\n=== ALL TESTS PASSED ===")
