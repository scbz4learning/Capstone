# CPU Inference

## Overview

CPU inference is the universal fallback when no GPU or NPU path is
available or practical. It works on almost any hardware, but is
significantly slower than accelerated paths.

## When to Use

- No GPU acceleration path is available
- Model operators or frameworks lack GPU backend support
- Running in a constrained environment where GPU drivers cannot be installed

## General Idea

Use a lightweight inference framework optimized for CPU execution.
Quantization and graph optimizations can help, but expect latency and
throughput to be much lower than GPU or NPU.
