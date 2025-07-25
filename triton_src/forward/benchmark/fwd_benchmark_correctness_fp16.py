#!/usr/bin/env python3
"""
FlashAttention2 Performance Benchmark CorrectnessScript

This script benchmarks FlashAttention2 implementations comparing Triton vs PyTorch performance WHILE ALSO CHECKING FOR CORRECTNESS.


# option 1: print autotuning info
CUDA_VISIBLE_DEVICES=0 TRITON_PRINT_AUTOTUNING=1 python fwd_benchmark_correctness.py
# option 2: or just run the script
python fwd_benchmark_correctness.py
"""

import torch
import triton
import math


import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from flash_forward_triton_fp16 import attention_triton_launch
from forward_torch import naive_forward_aux_wrapper

def compare_outputs(O_naive, O_triton, L_naive, L_triton):
        max_diff_ref = torch.max(torch.abs(O_naive - O_triton.float())).item()
        max_diff_ref_normed = torch.max(torch.abs(O_naive - O_triton.float()) / O_naive.abs()).item()

        max_diff_L = torch.max(torch.abs(L_naive - L_triton.float())).item()
        max_diff_L_normed = torch.max(torch.abs(L_naive - L_triton.float()) / L_naive.abs()).item()
        O_ratio = O_triton / O_naive
        L_ratio = L_triton / L_naive

        print(f"O statistics")
        O_matches = torch.allclose(O_naive, O_triton.float(), atol=5e-3, rtol=1e-3)
        print(f"  Naive vs Triton (main): {torch.allclose(O_naive, O_triton.float(), atol=5e-3, rtol=1e-3)}")
        print(f"  Max diff (naive vs ref):  {max_diff_ref:.6f}")
        print(f"  Max diff (naive vs ref) normalized:  {max_diff_ref_normed:.6f}")
        if not O_matches:
            print(f"O_triton")
            print(O_triton[0, :4, 0, :4])
            print(f"O_naive")
            print(O_naive[0, :4, 0, :4])
            # print(f"O_ratio")
            # print(O_ratio[0, :4, 0, :4])

        print(f"L statistics")
        L_matches = torch.allclose(L_naive, L_triton.float(), atol=5e-3, rtol=1e-3)
        print(f"  Naive vs Triton (main): {torch.allclose(L_naive, L_triton.float(), atol=5e-3, rtol=1e-3)}")
        print(f"  Max diff (naive vs ref):  {max_diff_L:.6f}")
        print(f"  Max diff (naive vs ref) normalized:  {max_diff_L_normed:.6f}")
        if not L_matches:
            print(f"L_triton")
            print(L_triton[0, 0, :4])
            print(f"L_naive")
            print(L_naive[0, 0, :4])
            # print(f"L_ratio")
            # print(L_ratio[0, 0, :4])

        # print(f"  O_ratio: {O_ratio[0, :, 0, :4]}")


def benchmark_custom_wrapper():

    configs = []
    HEAD_DIM = 64
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
    @triton.testing.perf_report(configs)
    def benchmark_correctness(N, H, D, provider):
        B = 8192 // N
        print(f"\n================================================")
        print(f"Benchmarking {provider} with B={B}, N={N}, H={H}, D={D}")

        torch.manual_seed(42)
        QKV_base = torch.randn(B, N, 3, H, D, device='cuda', dtype=torch.bfloat16)
        QKV = QKV_base.clone().detach()
        QKV_eval = QKV_base.clone().detach()
        _, _, O_naive, L_naive, _ = naive_forward_aux_wrapper(QKV_base)


        quantiles = [0.5, 0.2, 0.8]
        if provider == 'torch':
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: naive_forward_aux_wrapper(QKV),
                                                        rep = 200,
                                                        quantiles=quantiles)
        if provider == 'triton':
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: attention_triton_launch(QKV),
                                                        rep = 200,
                                                        quantiles=quantiles)
            O_triton, L_triton = attention_triton_launch(QKV_eval)
            compare_outputs(O_naive, O_triton, L_naive, L_triton)


        flops_per_matmul = 2.0 * B * H * N * N * D
        total_flops = 2 * flops_per_matmul
        perf = lambda ms: total_flops * 1e-12 / (ms * 1e-3)

        print(f"ms = {ms:.3f}, total_flops = {total_flops:.3e}, TFLOPS = {perf(ms):.3f}")
        return perf(ms), perf(max_ms), perf(min_ms)


    benchmark_correctness.run(save_path="./logs", print_data=True)


if __name__ == "__main__":
    benchmark_custom_wrapper()
