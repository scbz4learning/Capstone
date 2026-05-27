# CPU Inference

## Overview

CPU inference serves as a universal **fallback** mechanism in the inference stack. While it lacks the massive parallel processing power of dedicated accelerators like GPUs or NPUs, it provides the highest level of compatibility and reliability across almost any hardware environment.

## Role as a Fallback

In production environments, CPU inference is typically reserved for scenarios where specialized hardware is unavailable, mismatched, or exhausted. It ensures that the model remains functional even if the primary acceleration path fails.

### Performance Limitations

Compared to hardware-accelerated paths, CPU inference is significantly slower. Based on our profiling data for models like `smolvlm`:

- **CPU (Bfloat16)**: TTFT ~128s, TPOT ~288ms
- **integrated GPU (CUDA/ROCm)**: TTFT ~7s, TPOT ~105ms

In practice, CPU inference can be **10x to 20x slower** than integrated GPU paths, and even further behind discrete GPU solutions.

!!! info "Intel OpenVINO Acceleration"
    For CPU-based inference, Intel's **OpenVINO** toolkit may provide additional acceleration through highly optimized kernels and graph optimizations. While it is an Intel-led project, it also offers support for various cross-vendor hardware and can be a powerful tool to squeeze more performance out of the CPU path.


## Modern Optimizations (AVX-512 and BF16)

While CPU inference is slower by nature, performance on modern processors—especially **AMD APUs**—has seen substantial improvements. Recent architectures now natively support:

- **AVX-512**: Enables wider vector operations, allowing the processor to handle more data per clock cycle during matrix multiplications.
- **BF16 (Bfloat16) Instructions**: Provides native hardware acceleration for the brain-float16 data format commonly used in modern LLMs.

These instructions significantly narrow the performance gap compared to older processors, making the CPU fallback far more viable for light workloads or emergency scenarios.
