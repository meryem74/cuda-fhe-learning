# CUDA & FHE Learning — PURE 2026

Hands-on exercises from my first week in the PURE summer research program
(GPU-accelerated homomorphic encryption, supervised by Prof. Erkay Savaş,
Sabancı University).

## Contents

**CUDA fundamentals** (`01_cuda_basics.ipynb`, `*.cu`)
- `hello.cu` — kernel launch, thread/block indexing
- `vector_add.cu` — host/device memory management, cudaMemcpy
- `matmul.cu` — 2D grids, CPU vs GPU benchmark with cudaEvent

**Result:** 1024×1024 matrix multiplication on a Tesla T4:
CPU 3587 ms → GPU 5.24 ms (**684.6× speedup**)

**Orion FHE framework** (`02_orion_setup.ipynb`)
- Built the [Orion](https://github.com/baahl-nyu/orion) framework from source
  (Go 1.22 + Lattigo backend) and ran encrypted MNIST inference (LoLA CNN).

**Result:** FHE output matched cleartext PyTorch output with
MAE = 0.0045 (~7.8 bits of precision), single encrypted inference in 0.62 s.

## Environment
Google Colab — Tesla T4 (CUDA 12.8) for CUDA exercises, CPU runtime for Orion.
