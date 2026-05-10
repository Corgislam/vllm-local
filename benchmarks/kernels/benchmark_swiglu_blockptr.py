# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Head-to-head benchmark: raw-pointer vs. block-pointer SwiGLU-Step kernel.

Both kernels run in the same process against the same pre-allocated tensors.
For each shape, N_ROUNDS rounds are executed. Within each round the two
kernels are timed back-to-back; the order alternates across rounds so
temperature / boost drift does not systematically favour either variant.

Standalone script: not collected by pytest. Run with:
    python benchmarks/kernels/benchmark_swiglu_blockptr.py
"""

from __future__ import annotations

import statistics
import sys

import torch
import triton
import triton.testing

from vllm.model_executor.layers.activation import (
    swiglustep_and_mul_triton,
    swiglustep_and_mul_triton_blockptr,
)

SHAPES: list[tuple[int, int]] = [
    (128, 4096),
    (512, 4096),
    (4096, 4096),
    (4096, 14336),
]
DTYPE = torch.bfloat16
LIMIT = 7.0
N_ROUNDS = 7
WARMUP_ITERS = 3             # outside do_bench, to absorb Triton JIT
DO_BENCH_WARMUP = 25         # ms, default do_bench warmup
DO_BENCH_REP = 100           # ms, default do_bench measurement window


def _provenance() -> str:
    dev = torch.cuda.get_device_properties(0)
    return (
        f"GPU={dev.name} sm_{dev.major}{dev.minor} "
        f"mem={dev.total_memory // (1024 ** 3)}GiB | "
        f"torch={torch.__version__} "
        f"cuda={torch.version.cuda} "
        f"triton={triton.__version__}"
    )


def _bench_one_round(
    x: torch.Tensor,
    out_raw: torch.Tensor,
    out_blk: torch.Tensor,
    raw_first: bool,
) -> tuple[float, float]:
    """Return (raw_ms, blk_ms) for one round, running the requested
    kernel first."""

    def run_raw():
        swiglustep_and_mul_triton(out_raw, x, limit=LIMIT)

    def run_blk():
        swiglustep_and_mul_triton_blockptr(out_blk, x, limit=LIMIT)

    if raw_first:
        t_raw = triton.testing.do_bench(
            run_raw, warmup=DO_BENCH_WARMUP, rep=DO_BENCH_REP, quantiles=[0.5]
        )
        t_blk = triton.testing.do_bench(
            run_blk, warmup=DO_BENCH_WARMUP, rep=DO_BENCH_REP, quantiles=[0.5]
        )
    else:
        t_blk = triton.testing.do_bench(
            run_blk, warmup=DO_BENCH_WARMUP, rep=DO_BENCH_REP, quantiles=[0.5]
        )
        t_raw = triton.testing.do_bench(
            run_raw, warmup=DO_BENCH_WARMUP, rep=DO_BENCH_REP, quantiles=[0.5]
        )
    # do_bench returns a list of floats (ms) when quantiles= is passed; pick
    # the scalar median.
    return float(t_raw[0] if hasattr(t_raw, "__len__") else t_raw), \
           float(t_blk[0] if hasattr(t_blk, "__len__") else t_blk)


def bench_shape(B: int, d: int) -> dict:
    torch.manual_seed(0)
    x = torch.randn(B, 2 * d, dtype=DTYPE, device="cuda")
    out_raw = torch.empty(B, d, dtype=DTYPE, device="cuda")
    out_blk = torch.empty(B, d, dtype=DTYPE, device="cuda")

    # Warm both kernels before the first timed round. do_bench has its own
    # warmup, but the first JIT compile happens once per kernel per shape;
    # doing it explicitly keeps round 0 comparable to later rounds.
    for _ in range(WARMUP_ITERS):
        swiglustep_and_mul_triton(out_raw, x, limit=LIMIT)
        swiglustep_and_mul_triton_blockptr(out_blk, x, limit=LIMIT)
    torch.cuda.synchronize()

    raw_ms: list[float] = []
    blk_ms: list[float] = []
    for r in range(N_ROUNDS):
        raw_first = (r % 2 == 0)
        t_raw, t_blk = _bench_one_round(x, out_raw, out_blk, raw_first)
        raw_ms.append(t_raw)
        blk_ms.append(t_blk)

    return {
        "shape": (B, d),
        "raw_median": statistics.median(raw_ms),
        "raw_min": min(raw_ms),
        "raw_max": max(raw_ms),
        "blk_median": statistics.median(blk_ms),
        "blk_min": min(blk_ms),
        "blk_max": max(blk_ms),
        "raw_rounds": raw_ms,
        "blk_rounds": blk_ms,
    }


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA unavailable; aborting.", file=sys.stderr)
        return 1

    print(_provenance())
    print(
        f"dtype={DTYPE} limit={LIMIT} rounds={N_ROUNDS} "
        f"warmup_iters={WARMUP_ITERS} "
        f"do_bench(warmup={DO_BENCH_WARMUP}ms, rep={DO_BENCH_REP}ms)"
    )
    print()
    header = (
        f"{'(B, d)':>14} | "
        f"{'raw med':>9} {'(min..max)':>17} | "
        f"{'blk med':>9} {'(min..max)':>17} | "
        f"{'speedup':>8}"
    )
    print(header)
    print("-" * len(header))

    for (B, d) in SHAPES:
        r = bench_shape(B, d)
        speedup = r["raw_median"] / r["blk_median"]
        print(
            f"{str(r['shape']):>14} | "
            f"{r['raw_median']*1000:>7.2f}us "
            f"({r['raw_min']*1000:>5.2f}..{r['raw_max']*1000:>5.2f}) | "
            f"{r['blk_median']*1000:>7.2f}us "
            f"({r['blk_min']*1000:>5.2f}..{r['blk_max']*1000:>5.2f}) | "
            f"{speedup:>7.3f}x"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
