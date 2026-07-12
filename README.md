# CUDA & FHE Learning — PURE 2026

Hands-on exercises from my first weeks in the PURE summer research program
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

**OpenFHE hands-on** (`03_openfhe.ipynb`, `my_first_fhe.cpp`, `notes_openfhe_vs_orion.md`)

- Built [OpenFHE](https://github.com/openfheorg/openfhe-development) from source
  (CMake, Release) and ran the PKE examples: simple-integers (BFV),
  simple-real-numbers (CKKS), depth-bfvrns, rotation, simple-ckks-bootstrapping.
- Wrote my first hand-written FHE program in CKKS: batching,
  EvalAdd/EvalMult/EvalRotate with per-operation timing, and live
  multiplicative-depth tracking (levels 0 → 1 → 2 with depth = 2).
- Read through the encrypted SVM inference examples
  ([python-svm-examples](https://github.com/openfheorg/python-svm-examples)):
  plaintext model + encrypted input, inner products via Mult + rotate-and-add,
  slot selection by masking, and the depth cost of a polynomial kernel
  (multDepth 2 → 6).

**Results:**
- EvalMult ≈ 54× and EvalRotate ≈ 39× slower than EvalAdd
  (CKKS, N = 16384; 26.6 ms / 19.2 ms / 0.50 ms) — key-switching dominates
  FHE cost.
- CKKS multiplication error ≈ 1.2e-12 (~39.5 bits of precision) after a
  single multiplication.
- Bootstrapping observed live: remaining levels 1 → 10 after bootstrap,
  while precision dropped from 59 to 17 bits — bootstrapping refreshes the
  multiplicative budget but is itself approximate and costly.

## Environment

Google Colab — Tesla T4 (CUDA 12.8) for CUDA exercises, CPU runtime for
Orion and OpenFHE.

