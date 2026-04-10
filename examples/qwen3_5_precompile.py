# -*- coding: utf-8 -*-
"""True AOT compilation for all power-of-2 shapes.

Uses  jax.jit(fn).lower(*abstract_shapes).compile()  — XLA compilation runs
immediately from shape specifications, no real data required.
The compiled executables are persisted to disk via jax_compilation_cache_dir.

Subsequent inference / benchmark runs call @jax.jit functions whose shapes have
already been compiled; JAX loads the cached executable and skips recompilation.

Usage:
    source ~/.venv/bin/activate && cd ~/tunix
    python examples/qwen3_5_precompile.py \\
        --ckpt_dir ~/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/<hash> \\
        --max_seq_len  2048 \\
        --batch_sizes  1,2,4 \\
        --decode_steps 512 \\
        --jax_cache_dir ~/.cache/jax_compile_cache/qwen3_5
"""

import os
import sys
import argparse
import time

os.environ.setdefault('JAX_PROCESS_COUNT', '1')
os.environ.setdefault('JAX_PROCESS_ID', '0')
os.environ.setdefault('JAX_COORDINATION_SERVICE_ADDR', 'localhost:8888')
os.environ.setdefault('TPU_CHIPS_PER_HOST_BOUNDS', '2,2,1')
os.environ.setdefault('TPU_HOST_BOUNDS', '1,1,1')

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from tunix.models.qwen3_5 import model as qwen_model
from tunix.models.qwen3_5 import params as qwen_params
from tunix.utils import env_utils


# ---------------------------------------------------------------------------
# Shape utilities
# ---------------------------------------------------------------------------

def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


def _pow2_buckets(max_seq_len: int, min_seq_len: int = 32) -> list[int]:
    """All power-of-2 values from min_seq_len up to next_pow2(max_seq_len)."""
    buckets, p = [], min_seq_len
    while p <= _next_pow2(max_seq_len):
        buckets.append(p)
        p <<= 1
    return buckets


def _abstract(shape, dtype):
    """Shorthand for jax.ShapeDtypeStruct."""
    return jax.ShapeDtypeStruct(shape, dtype)


def _abstract_cache(lm_model, batch_size: int, cache_size: int, dtype):
    """Build a pytree of ShapeDtypeStruct mirroring init_cache output."""
    concrete = lm_model.init_cache(
        batch_size=batch_size, cache_size=cache_size, dtype=dtype
    )
    return jax.tree_util.tree_map(
        lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), concrete
    )


# ---------------------------------------------------------------------------
# AOT compilation helpers
# Each returns a jax.stages.Compiled object (callable, but we discard it;
# the side-effect is that the executable is written to the persistent cache).
# ---------------------------------------------------------------------------

def aot_prefill(lm_model, batch_size: int, T_pad: int, cache_size: int, dtype):
    """AOT-compile the prefill kernel for shape (batch_size, T_pad).

    Matches _make_prefill_fn in qwen3_5_tpu_inference.py exactly.
    """
    def prefill(tokens, positions, attn_mask, cache):
        return lm_model(
            input_tokens=tokens, positions=positions,
            cache=cache, attention_mask=attn_mask,
        )

    abstract_args = (
        _abstract((batch_size, T_pad), jnp.int32),           # tokens
        _abstract((batch_size, T_pad), jnp.int32),           # positions
        _abstract((batch_size, T_pad, cache_size), jnp.bool_),  # attn_mask
        _abstract_cache(lm_model, batch_size, cache_size, dtype),  # cache
    )
    return jax.jit(prefill).lower(*abstract_args).compile()


def aot_decode_bs1(lm_model, T_pad: int, n_steps: int, cache_size: int, dtype):
    """AOT-compile the lax.scan decode kernel for batch_size=1.

    Matches _make_scan_decode_fn in qwen3_5_tpu_inference.py exactly:
      carry last_token  : scalar int32
      tok               : last_token[None, None] → [1, 1]
      next_tok          : jnp.argmax(logits[0, 0]) → scalar int32
    """
    def run_decode(first_token, cache, dec_mask):
        def body(carry, step):
            cache, dec_mask, last_token = carry
            pos     = T_pad + step
            tok     = last_token[None, None]                        # [1, 1]
            pos_arr = jnp.full((1, 1), pos, dtype=jnp.int32)       # [1, 1]
            dec_mask = jax.lax.dynamic_update_slice(
                dec_mask, jnp.ones((1, 1, 1), jnp.bool_), (0, 0, pos)
            )
            logits, new_cache = lm_model(
                input_tokens=tok, positions=pos_arr,
                cache=cache, attention_mask=dec_mask,
            )
            next_tok = jnp.argmax(logits[0, 0]).astype(jnp.int32)  # scalar
            return (new_cache, dec_mask, next_tok), next_tok

        steps = jnp.arange(n_steps, dtype=jnp.int32)
        init  = (cache, dec_mask, first_token)      # first_token is scalar
        (_, _, _), all_tokens = jax.lax.scan(body, init, steps)
        return all_tokens

    abstract_args = (
        _abstract((), jnp.int32),                             # first_token (scalar)
        _abstract_cache(lm_model, 1, cache_size, dtype),      # cache (bs=1)
        _abstract((1, 1, cache_size), jnp.bool_),             # dec_mask
    )
    return jax.jit(run_decode).lower(*abstract_args).compile()


def aot_decode_batched(lm_model, batch_size: int, T_pad: int, n_steps: int,
                       cache_size: int, dtype):
    """AOT-compile the lax.scan decode kernel for batch_size >= 1.

    Matches make_scan_decode_fn in qwen3_5_benchmark.py:
      carry last_tokens : [B] int32
      tok              : last_tokens[:, None] → [B, 1]
      next_toks        : jnp.argmax(logits[:, 0, :], axis=-1) → [B]
    """
    B = batch_size

    def run_decode(first_tokens, cache, dec_mask):
        def body(carry, step):
            cache, dec_mask, last_tokens = carry
            pos     = T_pad + step
            toks    = last_tokens[:, None]                           # [B, 1]
            pos_arr = jnp.full((B, 1), pos, dtype=jnp.int32)        # [B, 1]
            dec_mask = jax.lax.dynamic_update_slice(
                dec_mask, jnp.ones((B, 1, 1), jnp.bool_), (0, 0, pos)
            )
            logits, new_cache = lm_model(
                input_tokens=toks, positions=pos_arr,
                cache=cache, attention_mask=dec_mask,
            )
            next_toks = jnp.argmax(logits[:, 0, :], axis=-1).astype(jnp.int32)
            return (new_cache, dec_mask, next_toks), next_toks

        steps = jnp.arange(n_steps, dtype=jnp.int32)
        init  = (cache, dec_mask, first_tokens)    # first_tokens: [B]
        (_, _, _), all_tokens = jax.lax.scan(body, init, steps)
        return all_tokens

    abstract_args = (
        _abstract((B,), jnp.int32),                           # first_tokens [B]
        _abstract_cache(lm_model, B, cache_size, dtype),      # cache
        _abstract((B, 1, cache_size), jnp.bool_),             # dec_mask
    )
    return jax.jit(run_decode).lower(*abstract_args).compile()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='True AOT precompile for all power-of-2 shapes'
    )
    parser.add_argument('--ckpt_dir', required=True)
    parser.add_argument('--dtype', default='bfloat16',
                        choices=['float32', 'bfloat16'])
    parser.add_argument('--batch_sizes', default='1',
                        help='Comma-separated, e.g. "1,2,4"')
    parser.add_argument('--max_seq_len', type=int, default=2048,
                        help='Compiles T_pad buckets from 32 up to next_pow2(max_seq_len)')
    parser.add_argument('--min_seq_len', type=int, default=32)
    parser.add_argument('--decode_steps', type=int, default=512,
                        help='Scan length; must match max_new_tokens-1 at inference time')
    parser.add_argument('--prefill_only', action='store_true')
    parser.add_argument('--jax_cache_dir',
                        default=os.path.expanduser('~/.cache/jax_compile_cache/qwen3_5'),
                        help='Persistent XLA cache directory')
    args = parser.parse_args()

    # ---- Enable persistent cache BEFORE any jax.jit ----
    os.makedirs(args.jax_cache_dir, exist_ok=True)
    jax.config.update('jax_compilation_cache_dir', args.jax_cache_dir)
    # Cache everything (no minimum compile-time threshold during precompile).
    jax.config.update('jax_persistent_cache_min_compile_time_secs', 0.0)
    print(f'[JAX cache] {args.jax_cache_dir}')

    env_utils.setup_sharding_environment()
    dtype       = jnp.bfloat16 if args.dtype == 'bfloat16' else jnp.float32
    batch_sizes = [int(x) for x in args.batch_sizes.split(',')]
    buckets     = _pow2_buckets(args.max_seq_len, args.min_seq_len)

    n_kernels = len(batch_sizes) * len(buckets) * (1 if args.prefill_only else 2)
    print(f'\nAOT compiling {n_kernels} kernels:')
    print(f'  batch_sizes  : {batch_sizes}')
    print(f'  T_pad buckets: {buckets}')
    print(f'  decode_steps : {args.decode_steps}  (skipped: {args.prefill_only})')
    print(f'  dtype        : {args.dtype}\n')

    # ---- Load model ----
    print(f'Loading model from {args.ckpt_dir} ...')
    cfg = qwen_model.ModelConfig.qwen3_5_0p8b()
    cfg.dtype = dtype
    lm = qwen_params.create_model_from_safe_tensors(
        file_dir=args.ckpt_dir, config=cfg, dtype=dtype,
    )
    n_params = sum(x.size for x in jax.tree_util.tree_leaves(nnx.state(lm)))
    print(f'Loaded ({n_params:,} params). Devices: {jax.devices()}\n')

    # ---- AOT compile grid ----
    col = 12
    print(f'{"batch":>{col}} {"T_pad":>{col}} {"stage":>{col}} {"t (s)":>{col}}  note')
    print('-' * (col * 4 + 8))

    t_wall = time.perf_counter()
    n_ok   = 0

    for bs in batch_sizes:
        for T_pad in buckets:
            cache_size = T_pad + args.decode_steps + 4

            # ---- Prefill ----
            stage = 'prefill'
            try:
                t0 = time.perf_counter()
                aot_prefill(lm, bs, T_pad, cache_size, dtype)
                elapsed = time.perf_counter() - t0
                note = 'OK'
                n_ok += 1
            except Exception as e:
                elapsed = 0.0
                note = f'ERR: {e!s:.50}'
            print(f'{bs:>{col}} {T_pad:>{col}} {stage:>{col}} {elapsed:>{col}.2f}  {note}',
                  flush=True)

            if args.prefill_only:
                continue

            # ---- Decode ----
            stage = 'decode'
            try:
                t0 = time.perf_counter()
                if bs == 1:
                    # Use scalar-carry variant matching qwen3_5_tpu_inference.py
                    aot_decode_bs1(lm, T_pad, args.decode_steps, cache_size, dtype)
                else:
                    # Use [B]-carry variant matching qwen3_5_benchmark.py
                    aot_decode_batched(lm, bs, T_pad, args.decode_steps, cache_size, dtype)
                elapsed = time.perf_counter() - t0
                note = 'OK'
                n_ok += 1
            except Exception as e:
                elapsed = 0.0
                note = f'ERR: {e!s:.50}'
            print(f'{bs:>{col}} {T_pad:>{col}} {stage:>{col}} {elapsed:>{col}.2f}  {note}',
                  flush=True)

    t_total = time.perf_counter() - t_wall
    print(f'\n{n_ok}/{n_kernels} kernels compiled in {t_total:.1f} s')
    print(f'Cache: {args.jax_cache_dir}')

    # Report cache size
    try:
        total_bytes = sum(
            f.stat().st_size
            for f in __import__('pathlib').Path(args.jax_cache_dir).rglob('*')
            if f.is_file()
        )
        print(f'Cache size: {total_bytes / 1e6:.1f} MB')
    except Exception:
        pass

    print('\nDone. Future runs load from cache — no recompilation.')


if __name__ == '__main__':
    main()
