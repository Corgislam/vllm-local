# RMSNorm block-pointer conversion

Part of vllm-project/vllm#40458 — migrating selected Triton kernels from raw
pointer (`tl.load(ptr + offsets, mask=...)`) to the block-pointer API
(`tl.make_block_ptr` + `tl.advance`). Companion to `feat/swiglu-blockptr`.

## What is in this branch

- `vllm/model_executor/layers/batch_invariant.py`
  - Adds `_rms_norm_kernel_blockptr` and `rms_norm_blockptr` alongside the
    original `_rms_norm_kernel` / `rms_norm`. The original remains the
    default dispatch for `rms_norm_batch_invariant`.
- `tests/kernels/core/test_rmsnorm_blockptr_equivalence.py`
  - 45 parameterized equivalence cases + 1 large-shape case + 1 pure-PyTorch
    reference integration case.
- `benchmarks/kernels/benchmark_rmsnorm_blockptr.py`
  - Standalone head-to-head benchmark with alternating execution order.
- `PLAN-rmsnorm-blockptr.md`
  - Scope / design / acceptance / risks.

## Design notes

The block-pointer kernel keeps the **same grid shape** (`(n_rows,)`), the
**same two-pass structure**, and the **same sequence of FP operations** as
the raw-pointer version. Only address generation changes.

To keep the `tl.sum` reduction shape identical to the raw-pointer kernel,
the row offset is folded into the base pointer (`input_ptr + row_idx *
stride`) and the block pointer is 1D (`shape=(n_cols,)`, `block_shape=
(BLOCK_SIZE,)`). This avoids the `[1, BLOCK_SIZE]` vs `[BLOCK_SIZE]`
reduction-tree mismatch that a naïve 2D block-pointer would introduce.

`offsets` are int32 (Triton requirement). The wrapper asserts last-dim
contiguity on both `input_2d` and `weight` — the block pointer's inner
stride is hard-coded to 1, so callers that violate the invariant get a
clear error instead of silent corruption.

## Validation on RTX 5090 (sm_120, Triton 3.6, CUDA 12.8)

### Equivalence (47/47 pass)

```
47 passed in 3.13s
```

- fp16, bf16: bitwise (`torch.equal`) on every parameterized case including
  the large `(4096, 14336)` shape.
- fp32: `torch.testing.assert_close(atol=2e-6, rtol=0)`. The block-pointer
  load vectorises differently than raw load, which changes the `tl.sum`
  reduction tree and produces 1–2 ulp drift on fp32 (peak observed
  `max|Δ| ≈ 1.43e-6`). fp16/bf16 are unaffected because the fp32 accumulator
  is rounded to half precision at store time.

### Benchmark (bf16, eps=1e-6, 7 rounds, alternating order per round)

```
  (rows, cols) |   raw med        (min..max) |   blk med        (min..max) |  speedup
-------------------------------------------------------------------------------------
   (128, 4096) |    7.74us ( 7.74.. 8.19) |    7.74us ( 7.74.. 8.19) |   1.000x
   (512, 4096) |   10.24us ( 9.79..10.24) |    9.79us ( 9.79..10.24) |   1.046x
  (4096, 4096) |   48.70us (48.70..49.15) |   48.70us (48.70..49.15) |   1.000x
 (4096, 14336) |  225.86us (224.86..226.77) |  225.82us (225.82..225.89) |   1.000x
```

No material regression on any shape.

## Reproduce

```bash
conda activate vllm

# equivalence
PYTHONPATH=. python -m pytest \
    tests/kernels/core/test_rmsnorm_blockptr_equivalence.py \
    --noconftest -v

# benchmark
PYTHONPATH=. python benchmarks/kernels/benchmark_rmsnorm_blockptr.py
```

## Scope boundary

- The active dispatch path (`rms_norm_batch_invariant`) is **unchanged**.
  The block-pointer wrapper is opt-in; callers select it explicitly.
- No upstream PR is opened from this branch. Local validation only.
- The remaining RFC kernels (log-softmax, MRoPE) will follow after this
  branch is reviewed.
