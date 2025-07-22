#!/usr/bin/env python3
"""
FlashAttention2 Performance Benchmark Script

This script benchmarks FlashAttention2 implementations comparing Triton vs PyTorch performance.


# option 1: print autotuning info
CUDA_VISIBLE_DEVICES=0 TRITON_PRINT_AUTOTUNING=1 python benchmark_script.py
# option 2: or just run the script
python benchmark_script.py
"""

import torch
import triton
import triton.language as tl
import sys
import os

# # Add the forward module to the path
# sys.path.append("/data/rbg/users/anugrah/triton/Flash-Triton/triton_src/forward")

from flash_forward_triton import attention_triton_launch
from forward_torch import naive_forward_aux_wrapper


def setup_benchmark_configs():
    """Setup benchmark configurations."""
    configs = []
    
    HEAD_DIM = 128
    H = 2048 // HEAD_DIM 
    SEQ_LENS = [512, 1024, 2048, 4096, 8192]
    
    configs.append(
        triton.testing.Benchmark(
            x_names=["N"],
            x_vals=SEQ_LENS,
            line_arg="provider",
            line_vals=["triton", "torch"],
            line_names=["Triton", "Torch"],
            styles=[("green", "-"), ("blue", "-")],
            ylabel="TFLOPS",
            plot_name="FlashAttention2 Performance",
            args={
                "H": H,
                "D": HEAD_DIM,
            },
        )
    )
    
    return configs


@triton.testing.perf_report(setup_benchmark_configs())
def benchmark(N, H, D, provider):
    """Benchmark function for FlashAttention2 performance comparison."""
    B = 8192 // N
    print(f"Benchmarking with B={B}, N={N}, H={H}, D={D}")
    
    torch.manual_seed(42)
    QKV = torch.randn(B, N, 3, H, D, device='cuda', dtype=torch.float32)

    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: naive_forward_aux_wrapper(QKV), 
            rep=200,
            quantiles=quantiles
        )
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: attention_triton_launch(QKV), 
            rep=200,
            quantiles=quantiles
        )

    flops_per_matmul = 2.0 * B * H * N * N * D
    total_flops = 2 * flops_per_matmul
    perf = lambda ms: total_flops * 1e-12 / (ms * 1e-3)
    return perf(ms), perf(max_ms), perf(min_ms)


def print_device_info():
    """Print CUDA device information."""
    device = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(device)
    print(f"Device: {props.name}")
    print(f"Compute Capability: {props.major}.{props.minor}")
    print(f"Total Memory: {props.total_memory // 1024 // 1024}MB")
    print(f"Multi-Processor Count: {props.multi_processor_count}")
    print(f"L2 Cache Size: {props.L2_cache_size // 1024 // 1024}MB")


def main():
    """Main function to run the benchmark."""
    print("FlashAttention2 Performance Benchmark")
    print("=" * 50)
    
    # Print device information
    print_device_info()
    print()
    
    # Run the benchmark
    print("Running benchmark...")
    benchmark.run(save_path="./logs", print_data=True)
    
    print("\nBenchmark completed!")


if __name__ == "__main__":
    main() 