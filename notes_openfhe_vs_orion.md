# OpenFHE vs. Orion — What I Learned

Notes from working through OpenFHE's PKE examples and writing my first
hand-written FHE program (`my_first_fhe.cpp`), compared against my earlier
experience running encrypted inference with the Orion framework.

## 1. Level of abstraction

Orion is an "autopilot": you write a normal PyTorch model, and range
estimation, level assignment, bootstrap placement, encoding, and key
generation all happen automatically behind `fit()` / `compile()`. OpenFHE is
"manual transmission": I create the CryptoContext myself, declare the
multiplicative depth up front (`SetMultiplicativeDepth`), generate every key
explicitly (public/secret, relinearization via `EvalMultKeyGen`, and a
separate rotation key for *each* rotation amount via `EvalRotateKeyGen`), and
call each homomorphic operation by hand (`EvalAdd`, `EvalMult`, `EvalRotate`).

## 2. Multiplicative depth = Orion's levels

In Orion I saw layers compiled at decreasing levels (`conv1 @ level=5` →
`fc2 @ level=1`). In OpenFHE the same concept appears as a budget I must
declare in advance. I verified it directly: with depth = 2, a fresh
ciphertext reports level 0, one multiplication moves it to 1, a second to 2,
and a third multiplication is no longer possible. Rescaling after each
multiplication is what consumes the level. The SVM examples showed the
practical consequence: a linear kernel needs multDepth = 2, while a degree-3
polynomial kernel needs multDepth = 6 — model complexity directly drives
crypto parameters.

## 3. Batching = Orion's slot/SIMD packing

`MakeCKKSPackedPlaintext` packs a whole vector into one ciphertext, and every
operation applies to all slots at once — the same SIMD property Orion
exploits for its packing strategies. With BatchSize = 8 and 8 values, I also
observed true cyclic rotation: rot(x,1) wrapped the first element around to
the last slot, i.e. (1,...,8) → (2,...,8,1). The polynomial-kernel SVM used
batching aggressively: the input vector is cloned once per support vector so
that a single Mult computes all inner products simultaneously.

## 4. Measured cost of operations (the key takeaway)

On my build (CKKS, N = 16384, auto-selected for 128-bit security):

| Operation  | Time     | Relative to EvalAdd |
|------------|----------|---------------------|
| EvalAdd    | 0.50 ms  | 1×                  |
| EvalRotate | 19.2 ms  | ~39×                |
| EvalMult   | 26.6 ms  | ~54×                |

Measured EvalMult ≈ 54× and EvalRotate ≈ 39× slower than EvalAdd, confirming
that key-switching operations dominate FHE cost. Across two independent runs
the absolute times varied with the Colab CPU, but the ratios stayed stable
(53×/37× vs 54×/39×). This matches what the Orion paper reports (HMult and
HRot require expensive key-switching, which is full of NTTs and RNS basis
conversions) and explains why Orion's main optimizations target rotation
counts, and why GPU FHE work focuses on accelerating NTT and key-switching.

## 5. Bootstrapping, observed live

Running simple-ckks-bootstrapping: a ciphertext with only 1 level remaining
was refreshed to 10 levels — but its precision dropped from 59 bits to 17
bits. Bootstrapping restores the multiplicative budget, yet it is itself an
approximate, expensive homomorphic computation. This is why deep networks
(ResNet-50 needs 351 bootstraps in Orion) are so slow on CPU, and why GPU
libraries like HEonGPU invest in multiple bootstrapping variants
(Regular/Slim/Bit/Gate).

## 6. Approximation error grows with circuit depth

CKKS is approximate. In my single-multiplication program the maximum error
was ~1.2e-12 (≈ 39.5 bits of precision), while Orion's LoLA network — dozens
of chained operations — ended at ≈ 7.8 bits (MAE 0.0045). Same scheme, same
error source; depth is what accumulates it.

## 7. BFV vs CKKS

`simple-integers` (BFV) computed on integers exactly, with no error; CKKS
returns real numbers with small noise. BFV = exact integer arithmetic,
CKKS = approximate real arithmetic — which is why CKKS is the natural choice
for machine learning.

## Next

The group's development happens on HEonGPU. Every Eval* call I made here on
the CPU exists there as CUDA kernels — the intersection of the two things I
studied these past weeks (CUDA + FHE semantics).
