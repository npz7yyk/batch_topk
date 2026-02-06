# batch_topk

**A single-file, JIT-compiled CUDA batch top-k with variable k and variable sequence length per row.**

Just drop [`batch_topk.py`](batch_topk.py) into your project — no build step, no extra dependencies beyond PyTorch.

## Features

- **All-in-one file** — CUDA kernels, JIT compilation, and Python API live in a single `batch_topk.py`. Import and go.
- **Variable k per row** — each row in the batch can request a different k.
- **Variable sequence length** — each row can have a different valid length, avoiding unnecessary computation on padding.
- **Adaptive algorithm** — automatically picks single-block (one block/row, best for large batch) or multi-block (multiple blocks/row, best for long sequences) based on problem shape.
- **fp16 / bf16 / fp32** — supports all common floating-point types.

## Quick Start

```python
from batch_topk import batch_topk, get_buffer

batch_size, max_len, k = 64, 131072, 1024

metric    = torch.randn(batch_size, max_len, device="cuda", dtype=torch.float16)
topks     = torch.full((batch_size,), k, device="cuda", dtype=torch.int32)
valid_lens = torch.full((batch_size,), max_len, device="cuda", dtype=torch.int32)
out_idxs  = torch.empty(batch_size, k, device="cuda", dtype=torch.int32)

# optional: pre-allocate workspace to avoid repeated allocation
buf = get_buffer(batch_size, max_len, k, metric.device)

batch_topk(metric, topks, valid_lens, out_idxs, buf)
# out_idxs now holds the column indices of the top-k elements per row
```

## API

### `batch_topk(metric, topks, valid_lens, out_idxs, buf=None, select_min=False)`

| Arg | Shape | Dtype | Description |
|---|---|---|---|
| `metric` | `(B, L)` | fp16 / bf16 / fp32 | Input scores |
| `topks` | `(B,)` | int32 | Per-row k |
| `valid_lens` | `(B,)` | int32 | Per-row valid length |
| `out_idxs` | `(B, max_k)` | int32 | Output indices (filled in-place) |
| `buf` | flat 1-D | uint8 | Optional workspace buffer |
| `select_min` | — | bool | Select smallest instead of largest |

### `get_buffer(batch_size, max_len, max_k, device) -> Tensor`

Pre-allocates the workspace buffer so you can reuse it across calls.

## How It Works

The kernel uses **radix select** — a linear-time selection algorithm that processes one radix digit (8 bits) per pass:

1. **Histogram** — count elements per bucket at the current radix digit.
2. **Choose bucket** — prefix-sum the histogram to find which bucket contains the k-th element; narrow k.
3. **Repeat** for each radix pass (2 passes for fp16, 4 for fp32).
4. **Filter** — emit all indices whose radix prefix is above the k-th threshold, then fill remaining slots with ties.

Two execution strategies are selected at runtime:

| Strategy | Blocks | Best when |
|---|---|---|
| **Single-block** | 1 per row | `batch_size` is large or `max_len <= 32768` |
| **Multi-block** | many per row | `batch_size` is small and `max_len` is large |

## License

MIT License

Copyright (c) 2026 Yikang Yue
