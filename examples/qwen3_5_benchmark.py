# -*- coding: utf-8 -*-
"""Qwen3.5-0.8B TPU throughput benchmark.

Measures prefill and decode throughput across a grid of batch sizes and
sequence lengths.  Uses fully synthetic token IDs so no real data is needed.
JIT compilation happens during warmup; only steady-state iterations are timed.

Usage:
    /home/limingjia1999/.venv/bin/python examples/qwen3_5_benchmark.py \\
        --ckpt_dir ~/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/<hash> \\
        --batch_sizes 1,2,4 \\
        --prompt_lens 32,64,128 \\
        --decode_steps 64 \\
        --n_warmup 1 --n_iters 3
"""

import os, sys, argparse, time
import numpy as np

os.environ.setdefault('JAX_PROCESS_COUNT', '1')
os.environ.setdefault('JAX_PROCESS_ID', '0')
os.environ.setdefault('JAX_COORDINATION_SERVICE_ADDR', 'localhost:8888')
os.environ.setdefault('TPU_CHIPS_PER_HOST_BOUNDS', '2,2,1')
os.environ.setdefault('TPU_HOST_BOUNDS', '1,1,1')

import jax
import jax.numpy as jnp
from flax import nnx

from tunix.models.qwen3_5 import model as qwen_model
from tunix.models.qwen3_5 import params as qwen_params
from tunix.utils import env_utils


# ---------------------------------------------------------------------------
# Persistent XLA compilation cache
# Set BEFORE any jax.jit.  First run compiles+saves; subsequent runs reuse.
# ---------------------------------------------------------------------------
_DEFAULT_JAX_CACHE = os.path.expanduser('~/.cache/jax_compile_cache/qwen3_5')

def _enable_jax_cache(cache_dir: str):
    os.makedirs(cache_dir, exist_ok=True)
    jax.config.update('jax_compilation_cache_dir', cache_dir)
    jax.config.update('jax_persistent_cache_min_compile_time_secs', 5.0)
    print(f'[JAX cache] {cache_dir}')


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


def _build_causal_mask_np(T: int, cache_size: int) -> np.ndarray:
    """[1, T, cache_size] causal attention mask."""
    m = np.zeros((1, T, cache_size), dtype=bool)
    for i in range(T):
        m[0, i, :i + 1] = True
    return m


# ---------------------------------------------------------------------------
# Prefill benchmark
# ---------------------------------------------------------------------------

def make_prefill_fn(lm_model):
    @jax.jit
    def prefill(tokens, positions, attn_mask, cache):
        return lm_model(
            input_tokens=tokens, positions=positions,
            cache=cache, attention_mask=attn_mask,
        )
    return prefill


def bench_prefill(lm_model, batch_size: int, prompt_len: int,
                  dtype, n_warmup: int, n_iters: int) -> float:
    """Returns mean prefill throughput in tokens/sec (batch * prompt_len / sec)."""
    T_pad      = _next_pow2(prompt_len)
    cache_size = T_pad + 4  # minimal cache for prefill-only

    # Synthetic inputs
    tokens = jnp.ones((batch_size, T_pad), dtype=jnp.int32)
    positions = jnp.broadcast_to(
        jnp.arange(T_pad, dtype=jnp.int32)[None], (batch_size, T_pad)
    )
    mask_np   = _build_causal_mask_np(T_pad, cache_size)
    attn_mask = jnp.broadcast_to(jnp.array(mask_np), (batch_size, T_pad, cache_size))
    cache     = lm_model.init_cache(batch_size=batch_size, cache_size=cache_size, dtype=dtype)

    prefill_fn = make_prefill_fn(lm_model)

    # Warmup (triggers XLA compilation)
    for _ in range(n_warmup):
        logits, _ = prefill_fn(tokens, positions, attn_mask, cache)
        logits.block_until_ready()

    # Timed iterations
    times = []
    for _ in range(n_iters):
        cache_i = lm_model.init_cache(batch_size=batch_size, cache_size=cache_size, dtype=dtype)
        t0 = time.perf_counter()
        logits, _ = prefill_fn(tokens, positions, attn_mask, cache_i)
        logits.block_until_ready()
        times.append(time.perf_counter() - t0)

    return batch_size * prompt_len / np.mean(times)


# ---------------------------------------------------------------------------
# Decode benchmark  (lax.scan over decode_steps, batch_size >= 1)
# ---------------------------------------------------------------------------

def make_scan_decode_fn(lm_model, T_pad: int, decode_steps: int, batch_size: int):
    """JIT-compiled batched lax.scan decode.

    Carry: (cache, dec_mask [B,1,S], last_tokens [B])
    Output per step: next_tokens [B]
    """
    @jax.jit
    def run(first_tokens, cache, dec_mask):
        def body(carry, step):
            cache, dec_mask, last_tokens = carry
            pos     = T_pad + step                                       # scalar
            toks    = last_tokens[:, None]                               # [B, 1]
            pos_arr = jnp.full((batch_size, 1), pos, dtype=jnp.int32)   # [B, 1]
            # Mark position as valid for all sequences
            update  = jnp.ones((batch_size, 1, 1), dtype=jnp.bool_)
            dec_mask = jax.lax.dynamic_update_slice(
                dec_mask, update, (0, 0, pos)
            )
            logits, new_cache = lm_model(
                input_tokens=toks, positions=pos_arr,
                cache=cache, attention_mask=dec_mask,
            )
            next_toks = jnp.argmax(logits[:, 0, :], axis=-1).astype(jnp.int32)  # [B]
            return (new_cache, dec_mask, next_toks), next_toks

        steps = jnp.arange(decode_steps, dtype=jnp.int32)
        init  = (cache, dec_mask, first_tokens)
        (_, _, _), all_tokens = jax.lax.scan(body, init, steps)
        return all_tokens  # [decode_steps, B]

    return run


def bench_decode(lm_model, batch_size: int, prompt_len: int, decode_steps: int,
                 dtype, n_warmup: int, n_iters: int) -> float:
    """Returns mean decode throughput in tokens/sec (batch * decode_steps / sec)."""
    T_pad      = _next_pow2(prompt_len)
    cache_size = T_pad + decode_steps + 4

    # Run a quick prefill to get a warm cache, then benchmark decode from that.
    tokens    = jnp.ones((batch_size, T_pad), dtype=jnp.int32)
    positions = jnp.broadcast_to(
        jnp.arange(T_pad, dtype=jnp.int32)[None], (batch_size, T_pad)
    )
    mask_np   = _build_causal_mask_np(T_pad, cache_size)
    attn_mask = jnp.broadcast_to(jnp.array(mask_np), (batch_size, T_pad, cache_size))

    prefill_fn     = make_prefill_fn(lm_model)
    scan_decode_fn = make_scan_decode_fn(lm_model, T_pad, decode_steps, batch_size)

    def _prefill_and_decode():
        cache = lm_model.init_cache(batch_size=batch_size, cache_size=cache_size, dtype=dtype)
        logits, cache = prefill_fn(tokens, positions, attn_mask, cache)
        first_tokens  = jnp.argmax(logits[:, prompt_len - 1, :], axis=-1).astype(jnp.int32)
        # Initial decode mask: prefill positions are valid
        dec_mask = jnp.zeros((batch_size, 1, cache_size), dtype=jnp.bool_)
        dec_mask = dec_mask.at[:, 0, :prompt_len].set(True)
        return first_tokens, cache, dec_mask

    # Warmup (JIT compilation happens here)
    for _ in range(n_warmup):
        first_tokens, cache, dec_mask = _prefill_and_decode()
        out = scan_decode_fn(first_tokens, cache, dec_mask)
        out.block_until_ready()

    # Timed iterations — time only the decode scan, not prefill
    times = []
    for _ in range(n_iters):
        first_tokens, cache, dec_mask = _prefill_and_decode()
        jax.effects_barrier()  # flush any async prefill work
        t0  = time.perf_counter()
        out = scan_decode_fn(first_tokens, cache, dec_mask)
        out.block_until_ready()
        times.append(time.perf_counter() - t0)

    return batch_size * decode_steps / np.mean(times)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Qwen3.5-0.8B TPU throughput benchmark')
    parser.add_argument('--ckpt_dir', required=True)
    parser.add_argument('--dtype', default='bfloat16', choices=['float32', 'bfloat16'])
    parser.add_argument('--batch_sizes',  default='1,2,4',
                        help='Comma-separated batch sizes to sweep')
    parser.add_argument('--prompt_lens',  default='32,64,128',
                        help='Comma-separated prompt lengths (padded to next pow2)')
    parser.add_argument('--decode_steps', type=int, default=64,
                        help='Number of decode tokens to generate per benchmark run')
    parser.add_argument('--n_warmup', type=int, default=1,
                        help='Warmup iterations (trigger JIT, not timed)')
    parser.add_argument('--n_iters',  type=int, default=3,
                        help='Timed iterations (mean is reported)')
    parser.add_argument('--prefill_only', action='store_true',
                        help='Skip decode benchmark (faster)')
    parser.add_argument('--flash_attention', action='store_true',
                        help='Enable Pallas Splash Attention for prefill (t > 256)')
    parser.add_argument('--flash_block_size', type=int, default=512,
                        help='Splash Attention block size (default 512)')
    parser.add_argument('--jax_cache_dir', default=_DEFAULT_JAX_CACHE,
                        help='JAX persistent compilation cache dir ("" to disable)')
    args = parser.parse_args()

    # Enable cache BEFORE any jax.jit / model loading.
    if args.jax_cache_dir:
        _enable_jax_cache(args.jax_cache_dir)

    env_utils.setup_sharding_environment()
    dtype = jnp.bfloat16 if args.dtype == 'bfloat16' else jnp.float32

    batch_sizes = [int(x) for x in args.batch_sizes.split(',')]
    prompt_lens = [int(x) for x in args.prompt_lens.split(',')]

    # --- Load model ---
    print(f'Loading model from {args.ckpt_dir} ...')
    cfg = qwen_model.ModelConfig.qwen3_5_0p8b()
    cfg.dtype = dtype
    if args.flash_attention:
        cfg.use_flash_attention = True
        cfg.flash_attention_block_size = args.flash_block_size
        mesh = jax.sharding.Mesh(
            np.array(jax.devices()).reshape(1, -1, 1, 1),
            axis_names=('fsdp', 'tp', 'sp', 'expert'),
        )
        print(f'Flash attention ON (block={args.flash_block_size}, mesh={mesh.shape})')
    else:
        mesh = None
    lm = qwen_params.create_model_from_safe_tensors(
        file_dir=args.ckpt_dir, config=cfg, mesh=mesh, dtype=dtype,
    )
    print(f'Model loaded. Devices: {jax.devices()}\n')

    # --- Benchmark grid ---
    COL = 14
    hdr  = f"{'batch':>{COL}} {'prompt_len':>{COL}} {'T_pad':>{COL}}"
    hdr += f" {'prefill tok/s':>{COL}}"
    if not args.prefill_only:
        hdr += f" {'decode tok/s':>{COL}} {'decode steps':>{COL}}"
    print(hdr)
    print('-' * len(hdr))

    for bs in batch_sizes:
        for pl in prompt_lens:
            T_pad = _next_pow2(pl)
            row   = f"{bs:>{COL}} {pl:>{COL}} {T_pad:>{COL}}"

            # Prefill
            try:
                pre_tps = bench_prefill(lm, bs, pl, dtype, args.n_warmup, args.n_iters)
                row += f" {pre_tps:>{COL}.1f}"
            except Exception as e:
                row += f" {'ERR':>{COL}}"
                print(f'  [prefill bs={bs} pl={pl}] {e}', file=sys.stderr)

            # Decode
            if not args.prefill_only:
                try:
                    dec_tps = bench_decode(
                        lm, bs, pl, args.decode_steps, dtype, args.n_warmup, args.n_iters
                    )
                    row += f" {dec_tps:>{COL}.1f} {args.decode_steps:>{COL}}"
                except Exception as e:
                    row += f" {'ERR':>{COL}} {args.decode_steps:>{COL}}"
                    print(f'  [decode  bs={bs} pl={pl}] {e}', file=sys.stderr)

            print(row)

    print()
    print(f'dtype={args.dtype}, n_warmup={args.n_warmup}, n_iters={args.n_iters}')
    print('Throughput = (batch_size × tokens) / wall_time  [excludes JIT compile]')


if __name__ == '__main__':
    main()
