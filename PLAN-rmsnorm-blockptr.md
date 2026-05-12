# Plan: Convert RMSNorm Triton kernel from raw pointer to block pointer

Tracking upstream discussion: vllm-project/vllm#40458 (RFC).

Local-only validation, no upstream PR at this stage. Follows the same
workflow as `feat/swiglu-blockptr`.

## Goal

Provide a drop-in `tl.make_block_ptr` variant of `_rms_norm_kernel`
(together with a matching Python wrapper) and prove, with reproducible
data, that:

1. it is **numerically equivalent** to the existing raw-pointer kernel on
   fp16 / bf16 / fp32 inputs across multiple shapes and `eps` values;
2. it does **not regress performance** on the local GPU (NVIDIA RTX 5090,
   Blackwell sm_120);
3. it integrates cleanly with the existing `rms_norm` Python wrapper
   (module-level integration against a PyTorch reference).

The original raw-pointer kernel is **kept in place**. The block-pointer
version is added alongside it so the two can be compared in the same
process. `rms_norm_batch_invariant` still dispatches to the raw-pointer
wrapper by default.

## Scope

### In scope

- `vllm/model_executor/layers/batch_invariant.py`
  - Add `_rms_norm_kernel_blockptr` (Triton kernel, block-pointer
    load/store with `tl.advance` across the column axis).
  - Add `rms_norm_blockptr` (Python wrapper, same signature as existing
    `rms_norm`).
  - Wrapper asserts last-dim contiguity on input and weight
    (`input_2d.stride(-1) == 1`, `weight.stride(-1) == 1`).
  - Original `_rms_norm_kernel` and `rms_norm` remain unchanged.
    `rms_norm_batch_invariant` continues to call the original.

- `tests/kernels/core/test_rmsnorm_blockptr_equivalence.py` (new)
  - Parameterized bitwise equivalence tests against the raw-pointer
    kernel on fp16 / bf16 / fp32, across block-aligned and non-aligned
    shapes, with multiple `eps` values.
  - One explicit large-shape test.
  - One module-level integration test comparing the block-pointer
    wrapper against a pure-PyTorch RMSNorm reference.

- `benchmarks/kernels/benchmark_rmsnorm_blockptr.py` (new)
  - Standalone script (not collected by pytest).
  - In-process head-to-head timing with alternating order per round.

### Out of scope

- Full `vllm.LLM.generate()` end-to-end test. The active dispatch path is
  unchanged (`rms_norm_batch_invariant` still calls the raw-pointer
  version), so a full generate pass would not exercise the new code.
- Hopper TMA variant using `tl.make_tensor_descriptor`.
- Converting the remaining RFC kernels (log-softmax, MRoPE). Those
  follow after RMSNorm is validated.
- Any upstream PR or change to dispatch defaults.

## Approach

### Block-pointer design

The raw-pointer kernel uses a 1D grid `(n_rows,)`; each program handles
one row via a manually-chunked column loop with two passes:

1. Pass 1 accumulates `sum_sq` in fp32 across all column chunks.
2. Pass 2 re-reads the row, normalises, scales by `weight`, and writes
   the output.

The block-pointer variant keeps the **same grid shape**, the **same two
passes**, and the **same sequence of FP operations** (sigmoid-free here;
just square / sum / sqrt / multiply). Only address generation changes:

- One `input_bp` block pointer for the input row, `shape=(n_rows, n_cols)`,
  `block_shape=(1, BLOCK_SIZE)`, `offsets=(row_idx, 0)`, advanced by
  `(0, BLOCK_SIZE)` each column iteration.
- One `weight_bp` 1D block pointer for the weight row,
  `shape=(n_cols,)`, `block_shape=(BLOCK_SIZE,)`, `offsets=(0,)`,
  advanced by `(BLOCK_SIZE,)`.
- One `output_bp` for the output row, mirror of `input_bp`.

For pass 1 we load through a separate `input_bp_pass1` that advances
independently; pass 2 uses fresh block pointers so the two passes do not
share advance state. Load / store use `boundary_check=(1,)` (input /
output) and `boundary_check=(0,)` (weight) on the column axis. The row
axis is always in-bounds by construction.

Accumulation order is preserved: `sum_sq += tl.sum(tl.where(mask, sq,
0.0))`. With block pointers the mask is implicit in `boundary_check`, so
we load with `padding_option="zero"` on the column axis, square, and
sum. This yields the same partial sums in the same order as the
raw-pointer version.

`BLOCK_SIZE = 1024`, matching the original. Grid shape unchanged.

### `offsets` dtype (important)

`tl.make_block_ptr` requires `offsets` / `block_shape` to be int32.
Raw-pointer `_rms_norm_kernel` casts `row_idx = tl.program_id(0).to(tl.int64)`
as an overflow guard for manual pointer arithmetic. **Do not** carry
that cast into the block-pointer version — `make_block_ptr` uses the
int64 `input_row_stride` passed from Python for address math internally;
`offsets` must stay int32. See
`roles/default/experience/2026-05-10-triton-make-block-ptr-requires-int32-offsets.md`.

### Contiguity assertion

The wrapper calls `.contiguous()` on `input_2d` and `weight` exactly as
the original does, and then asserts `stride(-1) == 1` on both. The
assertion guards against future refactors that remove the
`.contiguous()` call; block pointers with hard-coded column stride 1
would silently corrupt otherwise.

## Validation

### Equivalence test

Location: `tests/kernels/core/test_rmsnorm_blockptr_equivalence.py`.

- dtypes: `{float16, bfloat16, float32}`
- shapes `(n_rows, n_cols)`:
  - `(1, 128)` — tiny, single block
  - `(2, 1024)` — block-aligned
  - `(7, 1000)` — non-aligned, triggers boundary_check
  - `(17, 1536)` — non-aligned across two blocks
  - `(128, 4096)` — medium, multi-block
- eps ∈ `{1e-6, 1e-5, 1e-4}`
- Full Cartesian = 45 cases.

Plus one large-shape test:

- `(n_rows, n_cols) = (4096, 14336)`, `dtype = bfloat16`, `eps = 1e-6`.

Plus one reference integration test:

- Compute against a pure-PyTorch RMSNorm in fp32 (`x * rsqrt(mean(x^2)
  + eps) * weight`, cast back to input dtype). Compare the block-pointer
  output with `allclose(atol=2e-2, rtol=2e-2)` for fp16/bf16 — not
  bitwise because the reference runs the whole expression in fp32 and
  the kernel mixes fp32 accum with the input dtype at load time.

Assertions:

- Primary: `torch.equal(out_raw, out_blockptr)` — bitwise identical.
  Both kernels perform the same sequence of FP ops in the same order;
  only address generation differs.
- Fallback when primary fails (diagnostic only): `allclose(atol=0,
  rtol=0)` with max|Δ| logged.

Fixed seed (`torch.manual_seed(0)`).

### Benchmark

Location: `benchmarks/kernels/benchmark_rmsnorm_blockptr.py`.

- Shapes `(n_rows, n_cols) ∈ {(128, 4096), (512, 4096), (4096, 4096),
  (4096, 14336)}`.
- dtype: `torch.bfloat16` only (kernel is memory-bound).
- eps = `1e-6`.
- Per shape: `N_ROUNDS = 7`. Each round calls
  `triton.testing.do_bench` once for raw and once for block-pointer,
  with alternating execution order (`round % 2 == 0` → raw first;
  odd → block-pointer first).
- `do_bench(warmup=25, rep=100, quantiles=[0.5])` per invocation.
- Same pre-allocated `input`, `weight`, `out_raw`, `out_blockptr`
  tensors throughout a shape's 7 rounds.
- Warm up both kernels (3× each) before the first timed round to absorb
  Triton JIT.
- Report per shape: median of 7 round medians, min and max across
  rounds, speedup. Print GPU name, driver, Triton version for
  provenance.
- Not collected by pytest.

### Acceptance

- All 45 parameterized cases pass `torch.equal`.
- Large-shape case passes `torch.equal`.
- Reference integration case passes `allclose` within the stated
  tolerance.
- Benchmark shows block-pointer median within ±5 % of raw-pointer
  median on every shape (no material regression on 5090). Speedups
  welcome but not required locally.

If bitwise equivalence fails in any case, execution stops and the
failure mode is investigated before further changes.

## Risks

- **Two-pass accumulation order may diverge.** The raw-pointer version
  does `sum_sq += tl.sum(tl.where(mask, sq, 0.0))`; the block-pointer
  version relies on `boundary_check` + zero padding. If Triton lowers
  these to different reduction trees, `torch.equal` will fail.
  Mitigation: diagnostic fallback records max|Δ|; if non-zero, inspect
  and decide whether to accept `allclose(atol=0, rtol=0)` as equivalent.
- **`tl.advance` semantics on 1D block pointer for weight.** Less
  commonly exercised than 2D advance; test coverage on non-aligned
  shapes catches regressions.
- **Triton on Blackwell (sm_120) block-pointer lowering.** Same risk
  class as SwiGLU work; isolated to this worktree.
- **`input_2d = input.reshape(-1, input.shape[-1]).contiguous()` may
  copy.** The block-pointer wrapper inherits this from the raw-pointer
  version — no change in behaviour, but the `stride(-1) == 1` assertion
  is added so the invariant is checked explicitly.

## Execution order

1. Create worktree `feat/rmsnorm-blockptr`. *(done)*
2. Write this plan. *(in progress)*
3. Implement kernel + wrapper in `batch_invariant.py`.
4. Write equivalence test file.
5. Run equivalence test under `conda activate vllm`.
6. Write benchmark script.
7. Run benchmark and record output.
8. Sediment new findings (if any) into `~/.agent-knowledge/`.

Every step is independently verifiable. Failures at any step pause
execution pending a diagnosis rather than a forward-fix.
