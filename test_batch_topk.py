"""
Test script for batch_topk module.
"""

import torch
import time


def test_batch_topk_basic():
    """Test basic functionality."""
    from batch_topk import batch_topk

    print("Testing batch_topk basic...")

    batch_size = 4
    max_len = 100
    k = 10

    metric = torch.randn(batch_size, max_len, device='cuda',
                         dtype=torch.float16)
    topks = torch.full((batch_size,), k, device='cuda', dtype=torch.int32)
    valid_lens = torch.full((batch_size,), max_len, device='cuda',
                            dtype=torch.int32)
    out_idxs = torch.empty(batch_size, k, device='cuda', dtype=torch.int32)

    batch_topk(metric, topks, valid_lens, out_idxs, select_min=False)

    # Verify against PyTorch's topk
    ref_values, _ = torch.topk(metric, k, dim=1, largest=True, sorted=False)

    # Gather values using our indices
    our_values = torch.gather(metric, 1, out_idxs.long())

    # Sort both for comparison
    our_sorted, _ = torch.sort(our_values, dim=1, descending=True)
    ref_sorted, _ = torch.sort(ref_values, dim=1, descending=True)

    assert torch.allclose(our_sorted, ref_sorted, rtol=1e-3, atol=1e-3), \
        "Values mismatch!"

    print(f"  batch_size={batch_size}, max_len={max_len}, k={k}")
    print("  Values match: PASS")


def test_batch_topk_variable_k():
    """Test with variable k per row."""
    from batch_topk import batch_topk

    print("\nTesting batch_topk with variable k...")

    batch_size = 4
    max_len = 100

    metric = torch.randn(batch_size, max_len, device='cuda',
                         dtype=torch.float16)

    topks = torch.tensor([5, 10, 3, 7], device='cuda', dtype=torch.int32)
    valid_lens = torch.tensor([100, 80, 50, 100], device='cuda',
                              dtype=torch.int32)
    max_k = int(topks.max().item())

    out_idxs = torch.empty(batch_size, max_k, device='cuda', dtype=torch.int32)

    batch_topk(metric, topks, valid_lens, out_idxs, select_min=False)

    # Verify each row
    for i in range(batch_size):
        k = topks[i].item()
        vl = valid_lens[i].item()

        row_metric = metric[i, :vl]
        ref_values, _ = torch.topk(row_metric, k, largest=True, sorted=False)

        our_indices = out_idxs[i, :k]
        our_values = metric[i, our_indices.long()]

        our_sorted, _ = torch.sort(our_values, descending=True)
        ref_sorted, _ = torch.sort(ref_values, descending=True)

        assert torch.allclose(our_sorted, ref_sorted, rtol=1e-3, atol=1e-3), \
            f"Row {i}: Values mismatch!"

    print(f"  batch_size={batch_size}, topks={topks.tolist()}")
    print(f"  valid_lens={valid_lens.tolist()}")
    print("  All rows match: PASS")


def test_batch_topk_select_min():
    """Test selecting minimum values."""
    from batch_topk import batch_topk

    print("\nTesting batch_topk with select_min=True...")

    batch_size = 4
    max_len = 100
    k = 10

    metric = torch.randn(batch_size, max_len, device='cuda',
                         dtype=torch.float16)
    topks = torch.full((batch_size,), k, device='cuda', dtype=torch.int32)
    valid_lens = torch.full((batch_size,), max_len, device='cuda',
                            dtype=torch.int32)
    out_idxs = torch.empty(batch_size, k, device='cuda', dtype=torch.int32)

    batch_topk(metric, topks, valid_lens, out_idxs, select_min=True)

    ref_values, _ = torch.topk(metric, k, dim=1, largest=False, sorted=False)

    our_values = torch.gather(metric, 1, out_idxs.long())

    our_sorted, _ = torch.sort(our_values, dim=1)
    ref_sorted, _ = torch.sort(ref_values, dim=1)

    assert torch.allclose(our_sorted, ref_sorted, rtol=1e-3, atol=1e-3), \
        "Min values mismatch!"

    print(f"  batch_size={batch_size}, max_len={max_len}, k={k}")
    print("  Min values match: PASS")


def test_batch_topk_dtypes():
    """Test different data types."""
    from batch_topk import batch_topk

    print("\nTesting different dtypes...")

    batch_size = 4
    max_len = 100
    k = 10

    for dtype in [torch.float16, torch.bfloat16, torch.float32]:
        metric = torch.randn(batch_size, max_len, device='cuda', dtype=dtype)
        topks = torch.full((batch_size,), k, device='cuda', dtype=torch.int32)
        valid_lens = torch.full((batch_size,), max_len, device='cuda',
                                dtype=torch.int32)
        out_idxs = torch.empty(batch_size, k, device='cuda', dtype=torch.int32)

        batch_topk(metric, topks, valid_lens, out_idxs)

        ref_values, _ = torch.topk(
            metric, k, dim=1, largest=True, sorted=False
        )
        our_values = torch.gather(metric, 1, out_idxs.long())

        our_sorted, _ = torch.sort(our_values, dim=1, descending=True)
        ref_sorted, _ = torch.sort(ref_values, dim=1, descending=True)

        rtol = 1e-2 if dtype == torch.bfloat16 else 1e-3
        assert torch.allclose(our_sorted, ref_sorted, rtol=rtol, atol=1e-3), \
            f"Values mismatch for dtype {dtype}!"

        print(f"  dtype={dtype}: PASS")


def benchmark_batch_topk():
    """Benchmark batch_topk performance."""
    from batch_topk import batch_topk, get_buffer

    print("\nBenchmarking batch_topk...")

    configs = [
        (4, 128 * 1024, 0.05),    # Small batch, large seq
        (16, 128 * 1024, 0.05),   # Medium batch, large seq
        (128, 32 * 1024, 0.05),   # Large batch, medium seq
    ]

    for batch_size, max_len, k_ratio in configs:
        k = int(max_len * k_ratio)

        metric = torch.randn(batch_size, max_len, device='cuda',
                             dtype=torch.float16)
        topks = torch.full((batch_size,), k, device='cuda', dtype=torch.int32)
        valid_lens = torch.full((batch_size,), max_len, device='cuda',
                                dtype=torch.int32)
        out_idxs = torch.empty(batch_size, k, device='cuda', dtype=torch.int32)
        buf = get_buffer(batch_size, max_len, k, metric.device)

        # Warmup
        for _ in range(10):
            batch_topk(metric, topks, valid_lens, out_idxs, buf)
        torch.cuda.synchronize()

        # Benchmark batch_topk
        n_iters = 100
        start = time.perf_counter()
        for _ in range(n_iters):
            batch_topk(metric, topks, valid_lens, out_idxs, buf)
        torch.cuda.synchronize()
        batch_topk_time = (time.perf_counter() - start) / n_iters * 1000

        # Benchmark PyTorch topk
        for _ in range(10):
            torch.topk(metric, k, dim=1)
        torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(n_iters):
            torch.topk(metric, k, dim=1)
        torch.cuda.synchronize()
        pytorch_time = (time.perf_counter() - start) / n_iters * 1000

        speedup = pytorch_time / batch_topk_time
        print(f"  ({batch_size}, {max_len}, {k}): "
              f"ours={batch_topk_time:.3f}ms, "
              f"torch={pytorch_time:.3f}ms, "
              f"speedup={speedup:.2f}x")


if __name__ == "__main__":
    print("=" * 60)
    print("Batch Top-K Tests")
    print("=" * 60)

    test_batch_topk_basic()
    test_batch_topk_variable_k()
    test_batch_topk_select_min()
    test_batch_topk_dtypes()
    benchmark_batch_topk()

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
