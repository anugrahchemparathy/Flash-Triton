# Flash2-Triton
This repository contains an in-progress rewrite of my toy implementation of FlashAttention2 using Triton.

## Performance Benchmark Results
As per the [FlashAttention2](https://arxiv.org/pdf/2307.08691) paper we compute the TFLOPs of our implementation for various sequence lengths with $16384$ total tokens.
- **Model Parameters**: $H=32$ (num heads), $D=64$ (head dimension), $B * N = 16384$ (total tokens)


### Reported Results
The evaluations were carried out on a NVIDIA A100-PCIE-40GB GPU, with peak throughput of 19.5 TFLOP/s (fp32), 156 TFLOP/s (tf32), 312 TFLOP/s (fp16/bf16) ([reference](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf)).


#### Forward Pass (without causal mask)
| Sequence Length (N) | Batch Size (B) | Triton TFLOP/s | PyTorch TFLOP/s |
|---------------------|----------------|---------------|----------------|
| 512                 | 16             | 17.52         | 4.87          |
| 1024                | 8              | 26.55         | 5.20          |
| 2048                | 4              | 30.67         | 5.35          |
| 4096                | 2              | 34.40         | 5.34          |
| 8192                | 1              | 37.04         | 5.14          |

It seems like `tl.dot` uses `tf32` for its matmul (to take advantage of the tensorcore) which really messes up my benchmarking! The kernel is much faster than torch, but quite a bit slower than the theoretical tf32 peak using the tensorcore.


### Theoretical Peak Calculation
We can lower bound the TFLOPs of the attention computation with the two matrix multiplies:
```
TFLOPs (QK^T) = B * H * (N * N) * (D * 2) * 1e-12
TFLOPs (PV) = B * H * (N * D) * (N * 2) * 1e-12
TFLOPs (total) > 4 * B * H * N * N * D * 1e-12
```



## TODOs

- [ ] Forward Pass
    - [ ] Add BF16/FP16 versions
- [ ] Backward Pass
    - [ ] Add FP32 version
    - [ ] Add BF16/FP16 versions
- [ ] General
    - [ ] Train a small model with an MHA module derived from these kernels