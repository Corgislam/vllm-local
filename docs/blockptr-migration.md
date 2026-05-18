# Triton Kernel 改写技术文档：raw pointer → block pointer

## 背景

vLLM 中的 Triton kernel 原本使用 **raw pointer + offset + mask** 的方式访问内存。本次改写将其迁移到 `tl.make_block_ptr` + `tl.advance` 的 **block pointer** 风格。三个 kernel 已完成改写：

1. **SwiGLU** (`vllm/model_executor/layers/activation.py` — `_swiglustep_and_mul_kernel`)
2. **RMSNorm** (`vllm/model_executor/layers/batch_invariant.py` — `_rms_norm_kernel`)
3. **Log-softmax** (`vllm/model_executor/layers/batch_invariant.py` — `_log_softmax_kernel`)

---

## 核心概念对比

### Raw Pointer 风格

```python
# 手动计算偏移量，手动构造 mask
offsets = j * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
mask = offsets < d
vals = tl.load(ptr + row_offset + offsets, mask=mask, other=0.0)
tl.store(ptr + row_offset + offsets, result, mask=mask)
```

特点：
- 地址 = 基址 + 手算偏移
- 越界保护靠 `mask=` 参数
- 每次迭代需要重新计算 `offsets`

### Block Pointer 风格

```python
# 声明一个"带形状和步长信息"的指针对象
bp = tl.make_block_ptr(
    base=ptr,
    shape=(rows, cols),   # 逻辑张量的形状（用于越界检查）
    strides=(row_stride, 1),
    offsets=(row_i, col_start),
    block_shape=(1, BLOCK_SIZE),
    order=(1, 0),         # 内存布局：最后一维连续
)
vals = tl.load(bp, boundary_check=(1,))   # 只检查列维度
tl.store(bp, result, boundary_check=(1,))

# 迭代时用 advance 移动指针，不重算偏移
bp = tl.advance(bp, (0, BLOCK_SIZE))
```

特点：
- 指针携带形状/步长/当前偏移的完整元数据
- 越界保护靠 `boundary_check=` 指定哪些维度需要检查
- 迭代靠 `tl.advance` 增量移动，代码更清晰

---

## SwiGLU Kernel 改写

### 原始版本（raw pointer）

文件：`vllm/model_executor/layers/activation.py` — `_swiglustep_and_mul_kernel`

```python
@triton.jit
def _swiglustep_and_mul_kernel(
    o_ptr, o_stride, x_ptr, x_stride,
    limit: tl.constexpr, d: tl.constexpr, BLOCK_SIZE: tl.constexpr,
) -> None:
    i = tl.program_id(axis=0).to(tl.int64)
    j = tl.program_id(axis=1)
    o_row_ptr = o_ptr + o_stride * i
    x_row_ptr = x_ptr + x_stride * i
    offsets = j * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < d

    gate = tl.load(x_row_ptr + offsets, mask=mask).to(tl.float32)
    up   = tl.load(x_row_ptr + offsets + d, mask=mask).to(tl.float32)
    # ... 计算 ...
    tl.store(o_row_ptr + offsets, result, mask=mask)
```

**关键问题：**
- `x_row_ptr + offsets` 和 `x_row_ptr + offsets + d` 是两次手算偏移
- `mask = offsets < d` 需要在每个 program 里重新计算
- 行索引 `i` 必须转成 `int64` 防止大 batch 溢出，但 `offsets` 是 `int32`，混合类型容易出错

### 改写版本（block pointer）

文件：`vllm/model_executor/layers/activation.py` — `_swiglustep_and_mul_kernel_blockptr`

```python
@triton.jit
def _swiglustep_and_mul_kernel_blockptr(
    o_ptr, o_stride, x_ptr, x_stride,
    B,                                    # 新增：行数，用于 shape 声明
    limit: tl.constexpr, d: tl.constexpr, BLOCK_SIZE: tl.constexpr,
) -> None:
    i = tl.program_id(axis=0)            # int32 即可，make_block_ptr 内部用 int64 strides
    j = tl.program_id(axis=1)
    col_off = j * BLOCK_SIZE

    gate_bp = tl.make_block_ptr(
        base=x_ptr,
        shape=(B, 2 * d),                # 完整张量形状
        strides=(x_stride, 1),
        offsets=(i, col_off),            # gate 从列 0 开始
        block_shape=(1, BLOCK_SIZE),
        order=(1, 0),
    )
    up_bp = tl.make_block_ptr(
        base=x_ptr,
        shape=(B, 2 * d),
        strides=(x_stride, 1),
        offsets=(i, d + col_off),        # up 从列 d 开始，偏移直接写在 offsets 里
        block_shape=(1, BLOCK_SIZE),
        order=(1, 0),
    )
    out_bp = tl.make_block_ptr(
        base=o_ptr,
        shape=(B, d),
        strides=(o_stride, 1),
        offsets=(i, col_off),
        block_shape=(1, BLOCK_SIZE),
        order=(1, 0),
    )

    gate = tl.load(gate_bp, boundary_check=(1,)).to(tl.float32)
    up   = tl.load(up_bp,   boundary_check=(1,)).to(tl.float32)
    # ... 同样的计算逻辑 ...
    tl.store(out_bp, result, boundary_check=(1,))
```

**改写要点：**

| 原始 | 改写后 | 原因 |
|------|--------|------|
| `x_row_ptr + offsets` | `gate_bp`（offsets=(i, col_off)） | 偏移折入指针声明 |
| `x_row_ptr + offsets + d` | `up_bp`（offsets=(i, d+col_off)） | gate/up 的分割点 `d` 直接写在 offsets 里 |
| `mask = offsets < d` | `boundary_check=(1,)` | 列维度越界由 Triton 自动处理 |
| `i.to(tl.int64)` | `i` 保持 int32 | strides 是 int64，make_block_ptr 内部处理大地址 |

**Python wrapper 新增的约束检查：**

```python
assert input.stride(-1) == 1,  "需要最后一维连续"
assert output.stride(-1) == 1, "需要最后一维连续"
```

block pointer 的 inner stride 硬编码为 1，非连续张量会产生错误结果，所以必须在入口处断言。

---

## RMSNorm Kernel 改写

### 原始版本（raw pointer）

文件：`vllm/model_executor/layers/batch_invariant.py` — `_rms_norm_kernel`

```python
@triton.jit
def _rms_norm_kernel(...):
    row_idx = tl.program_id(0).to(tl.int64)
    row_start_ptr = input_ptr + row_idx * input_row_stride

    # Pass 1: 求平方和
    sum_sq = tl.zeros([1], dtype=tl.float32)
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = col_idx < n_cols
        vals = tl.load(row_start_ptr + col_idx, mask=mask, other=0.0)
        vals_f32 = vals.to(tl.float32)
        sum_sq += tl.sum(tl.where(mask, vals_f32 * vals_f32, 0.0))

    inv_rms = 1.0 / tl.sqrt(sum_sq / n_cols + eps)

    # Pass 2: 归一化并写出
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = col_idx < n_cols
        vals   = tl.load(row_start_ptr + col_idx, mask=mask, other=0.0)
        weight = tl.load(weight_ptr + col_idx,    mask=mask, other=1.0)
        output = (vals.to(tl.float32) * inv_rms * weight.to(tl.float32)).to(vals.dtype)
        tl.store(output_row_start_ptr + col_idx, output, mask=mask)
```

### 改写版本（block pointer）

文件：`vllm/model_executor/layers/batch_invariant.py` — `_rms_norm_kernel_blockptr`

```python
@triton.jit
def _rms_norm_kernel_blockptr(...):
    row_idx = tl.program_id(0).to(tl.int64)
    # 行偏移折入基址，block pointer 保持 1D
    in_row_base  = input_ptr  + row_idx * input_row_stride
    out_row_base = output_ptr + row_idx * output_row_stride

    in_bp = tl.make_block_ptr(
        base=in_row_base,
        shape=(n_cols,),
        strides=(1,),
        offsets=(0,),
        block_shape=(BLOCK_SIZE,),
        order=(0,),
    )

    # Pass 1: 求平方和，用 tl.advance 推进指针
    sum_sq = tl.zeros([1], dtype=tl.float32)
    for _ in range(0, n_cols, BLOCK_SIZE):
        vals = tl.load(in_bp, boundary_check=(0,), padding_option="zero")
        vals_f32 = vals.to(tl.float32)
        sum_sq += tl.sum(vals_f32 * vals_f32)
        in_bp = tl.advance(in_bp, (BLOCK_SIZE,))

    inv_rms = 1.0 / tl.sqrt(sum_sq / n_cols + eps)

    # Pass 2: 重新创建指针（从头开始），归一化并写出
    in_bp2 = tl.make_block_ptr(base=in_row_base,  shape=(n_cols,), strides=(1,),
                                offsets=(0,), block_shape=(BLOCK_SIZE,), order=(0,))
    out_bp = tl.make_block_ptr(base=out_row_base, shape=(n_cols,), strides=(1,),
                                offsets=(0,), block_shape=(BLOCK_SIZE,), order=(0,))
    w_bp   = tl.make_block_ptr(base=weight_ptr,   shape=(n_cols,), strides=(1,),
                                offsets=(0,), block_shape=(BLOCK_SIZE,), order=(0,))

    for _ in range(0, n_cols, BLOCK_SIZE):
        vals   = tl.load(in_bp2, boundary_check=(0,), padding_option="zero")
        weight = tl.load(w_bp,   boundary_check=(0,), padding_option="zero")
        output = (vals.to(tl.float32) * inv_rms * weight.to(tl.float32)).to(vals.dtype)
        tl.store(out_bp, output, boundary_check=(0,))
        in_bp2 = tl.advance(in_bp2, (BLOCK_SIZE,))
        w_bp   = tl.advance(w_bp,   (BLOCK_SIZE,))
        out_bp = tl.advance(out_bp,  (BLOCK_SIZE,))
```

**改写要点：**

| 原始 | 改写后 | 原因 |
|------|--------|------|
| `row_start_ptr + col_idx` | 1D block pointer，base 已含行偏移 | 保持 1D 使 reduction 形状与原版一致（`[BLOCK_SIZE]` vs `[1, BLOCK_SIZE]`），确保 fp32 数值等价 |
| `mask = col_idx < n_cols` | `boundary_check=(0,)` | 越界由 Triton 处理 |
| `tl.where(mask, sq_vals, 0.0)` | `padding_option="zero"` | 越界列自动填 0，等价于原版的 `other=0.0` |
| `for col_offset in range(...)` 手算 `col_idx` | `for _ in range(...)`，用 `tl.advance` 推进 | 不再需要在循环体内计算偏移 |
| Pass 2 复用同一个 `row_start_ptr` | Pass 2 重新 `make_block_ptr`（从 offset=0 开始） | block pointer 是有状态对象，Pass 1 结束后已推进到末尾，必须重建 |

---

## Log-softmax Kernel 改写

### 原始版本（raw pointer）

文件：`vllm/model_executor/layers/batch_invariant.py` — `_log_softmax_kernel`

```python
@triton.jit
def _log_softmax_kernel(...):
    row_idx = tl.program_id(0).to(tl.int64)
    row_start_ptr = input_ptr + row_idx * input_row_stride
    output_row_start_ptr = output_ptr + row_idx * output_row_stride

    # Pass 1: 求行最大值
    max_val = -float("inf")
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = col_idx < n_cols
        vals = tl.load(row_start_ptr + col_idx, mask=mask, other=-float("inf"))
        max_val = tl.max(tl.maximum(vals, max_val))

    # Pass 2: 求 exp(x - max_val) 的和
    sum_exp = 0.0
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = col_idx < n_cols
        vals = tl.load(row_start_ptr + col_idx, mask=mask, other=0.0)
        exp_vals = tl.exp(vals - max_val)
        sum_exp += tl.sum(tl.where(mask, exp_vals, 0.0))

    log_sum_exp = tl.log(sum_exp)

    # Pass 3: 写出 x - max_val - log_sum_exp
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = col_idx < n_cols
        vals = tl.load(row_start_ptr + col_idx, mask=mask)
        output = vals - max_val - log_sum_exp
        tl.store(output_row_start_ptr + col_idx, output, mask=mask)
```

**关键问题：**
- 这是三 pass kernel，每个 pass 都会从行首重新扫描一次，因此 block pointer 版本每个 pass 都必须重新创建指针。
- Pass 1 的 `other=-inf` 不能用 `padding_option="zero"` 直接替代，否则全负数行的 ragged block 会错误地把 `max_val` 抬到 0。
- Pass 2 即使读取越界位置时 zero padding，也必须保留逻辑 mask，避免 padded lane 贡献 `exp(0 - max_val)`。
- block pointer store 对 value dtype 更严格，写 fp16/bf16 输出前必须显式 cast。

### 改写版本（block pointer）

文件：`vllm/model_executor/layers/batch_invariant.py` — `_log_softmax_kernel_blockptr`

```python
@triton.jit
def _log_softmax_kernel_blockptr(...):
    row_idx = tl.program_id(0).to(tl.int64)
    row_start_ptr = input_ptr + row_idx * input_row_stride
    output_row_start_ptr = output_ptr + row_idx * output_row_stride

    in_bp = tl.make_block_ptr(
        base=row_start_ptr,
        shape=(n_cols,),
        strides=(1,),
        offsets=(0,),
        block_shape=(BLOCK_SIZE,),
        order=(0,),
    )

    # Pass 1: boundary_check 只能安全读取，-inf 语义需要手动恢复
    max_val = -float("inf")
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = col_idx < n_cols
        vals = tl.load(in_bp, boundary_check=(0,), padding_option="zero")
        vals = tl.where(mask, vals, -float("inf"))
        max_val = tl.max(tl.maximum(vals, max_val))
        in_bp = tl.advance(in_bp, (BLOCK_SIZE,))

    # Pass 2: 重新从行首开始，masked lane 不参与 sum_exp
    in_bp2 = tl.make_block_ptr(base=row_start_ptr, shape=(n_cols,), strides=(1,),
                               offsets=(0,), block_shape=(BLOCK_SIZE,), order=(0,))
    sum_exp = 0.0
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = col_idx < n_cols
        vals = tl.load(in_bp2, boundary_check=(0,), padding_option="zero")
        exp_vals = tl.exp(vals - max_val)
        sum_exp += tl.sum(tl.where(mask, exp_vals, 0.0))
        in_bp2 = tl.advance(in_bp2, (BLOCK_SIZE,))

    log_sum_exp = tl.log(sum_exp)

    # Pass 3: 重新从行首开始，计算并写出
    in_bp3 = tl.make_block_ptr(base=row_start_ptr, shape=(n_cols,), strides=(1,),
                               offsets=(0,), block_shape=(BLOCK_SIZE,), order=(0,))
    out_bp = tl.make_block_ptr(base=output_row_start_ptr, shape=(n_cols,), strides=(1,),
                               offsets=(0,), block_shape=(BLOCK_SIZE,), order=(0,))
    for _ in range(0, n_cols, BLOCK_SIZE):
        vals = tl.load(in_bp3, boundary_check=(0,), padding_option="zero")
        output = vals - max_val - log_sum_exp
        output = output.to(input_ptr.dtype.element_ty)
        tl.store(out_bp, output, boundary_check=(0,))
        in_bp3 = tl.advance(in_bp3, (BLOCK_SIZE,))
        out_bp = tl.advance(out_bp, (BLOCK_SIZE,))
```

**改写要点：**

| 原始 | 改写后 | 原因 |
|------|--------|------|
| `row_start_ptr + col_idx` | 1D block pointer，base 已含行偏移 | 保持 reduction 形状为 `[BLOCK_SIZE]`，和 raw kernel 对齐 |
| `tl.load(..., other=-inf)` | `padding_option="zero"` + `tl.where(mask, vals, -inf)` | block pointer padding 不支持 `-inf`，必须手动恢复 max pass 语义 |
| `tl.where(mask, exp_vals, 0.0)` | 保留 `tl.where(mask, exp_vals, 0.0)` | padded lane 不能贡献 `exp(0 - max_val)` |
| 三个 pass 都从行首开始 | 每个 pass 重新 `make_block_ptr` | `tl.advance` 后指针已推进到末尾，下一 pass 必须重建 |
| raw store 隐式处理 dtype | `output.to(input_ptr.dtype.element_ty)` 后 `tl.store` | block pointer store 要求 value dtype 与目标元素类型匹配 |

**验证结果：**
- 测试文件：`tests/kernels/core/test_logsoftmax_blockptr_equivalence.py`
- 覆盖：`fp16` / `bf16` / `fp32`，5 组随机 shape，4 组全负数 ragged shape，大 shape，以及 PyTorch sanity check
- 运行结果：`29 passed`
- raw pointer 与 block pointer 在覆盖用例中 `max|Δ| = 0.0`

---

## 三次改写的共同规律

### 1. 行偏移的处理策略

SwiGLU、RMSNorm 和 Log-softmax 采用了不同策略，各有适用场景：

**SwiGLU：行偏移写入 `offsets` 参数（2D block pointer）**
```python
tl.make_block_ptr(base=x_ptr, shape=(B, 2*d), strides=(x_stride, 1),
                  offsets=(i, col_off), block_shape=(1, BLOCK_SIZE), order=(1,0))
```
适合：需要同时访问同一张量的不同列段（gate 和 up），用 2D 指针可以在声明时就区分两个区域。

**RMSNorm：行偏移折入 `base`（1D block pointer）**
```python
in_row_base = input_ptr + row_idx * input_row_stride
tl.make_block_ptr(base=in_row_base, shape=(n_cols,), strides=(1,), ...)
```
适合：每个 program 只访问一行，1D 指针更简洁，且 reduction 的 tensor 形状保持 `[BLOCK_SIZE]`，与原版数值完全等价。

**Log-softmax：同样采用 1D block pointer，但保留逻辑 mask**
```python
vals = tl.load(in_bp, boundary_check=(0,), padding_option="zero")
vals = tl.where(mask, vals, -float("inf"))
```
适合：需要 `other=-inf` 或 masked sum 的 reduction kernel。block pointer 负责地址生成和越界安全读取，mask 仍负责恢复数学语义。

### 2. `boundary_check` vs `mask`

```
mask=offsets < n   →   boundary_check=(dim,)
other=0.0          →   padding_option="zero"
other=1.0          →   padding_option="zero" 后手动处理（或保持 mask 方式）
other=-inf         →   padding_option="zero" 后用 tl.where(mask, vals, -inf)
```

`boundary_check` 只能填 0（zero padding），如果 `other` 不是 0 则不能直接替换，需要额外处理。

### 3. `tl.advance` 的使用

`tl.advance` 返回一个新的 block pointer，原指针不变（Triton 的 block pointer 是不可变值）：

```python
bp = tl.advance(bp, (BLOCK_SIZE,))   # 正确：重新赋值
tl.advance(bp, (BLOCK_SIZE,))        # 错误：返回值被丢弃
```

### 4. 数值等价的保证

改写只改变地址生成方式，不改变：
- 浮点运算顺序
- dtype 转换时机（`vals.to(tl.float32)` 的位置）
- reduction 的 tensor 形状

这三点保证了 fp16/bf16/fp32 下的数值等价。

### 5. block pointer store 的 dtype 更严格

raw pointer store 可以在一些场景下隐式处理 dtype 转换，但 block pointer store 会检查 value dtype 与目标元素类型是否一致。若计算表达式提升到了 fp32，写回 fp16/bf16 前必须显式 cast：

```python
output = output.to(input_ptr.dtype.element_ty)
tl.store(out_bp, output, boundary_check=(0,))
```

缺少这个 cast 会在 Triton 编译阶段报 block element type 与 value element type mismatch。

---

## 约束总结

使用 block pointer 有以下硬性前提：

1. **最后一维必须连续**（`stride(-1) == 1`）——inner stride 硬编码为 1
2. **`order` 参数必须与实际内存布局一致**——`order=(1, 0)` 表示行优先（C-contiguous）
3. **多 pass 的 kernel 需要重建指针**——block pointer 是有状态的，pass 结束后偏移已到末尾
4. **`B`（行数）需要作为运行时参数传入**——`shape` 参数不能是 `constexpr`，需要从 Python 侧传递
5. **非零 padding 语义必须手动恢复**——`boundary_check` 不能直接表达 `other=-inf` 或 `other=1.0`
6. **store 前显式 cast**——当计算结果 dtype 与目标 tensor dtype 不一致时，block pointer store 不会替你兜底
