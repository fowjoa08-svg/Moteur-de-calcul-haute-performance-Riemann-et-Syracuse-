# High-Performance Computing (HPC) Simulation Engine

An asynchronous numerical simulation engine built to audit the boundaries of computational verification regarding the Riemann Hypothesis and the Collatz Conjecture.

## 🛠️ Technical Architecture & Optimization
- **Memory Footprint:** Operates strictly under a continuous data-stream pipeline with an O(1) memory footprint, maintaining a Resident Set Size (RSS) bounded at **1.6 MB** to prevent memory leaks or out-of-memory crashes on large datasets.
- **Asynchronous Safepoints:** Implements periodic local JSON state serialization checkpoints every 10 wall-clock seconds to handle system timeouts natively.
- **Hardware Acceleration:** C-based verification modules optimized using OpenMP multithreading directives (`-O3 -march=native -fopenmp`) and native __uint128_t precision handling to mitigate mathematical overflow limits.

## 🔬 Mathematical Protocols Implemented
1. **Davenport-Heilbronn Counter-Example Proof:** A custom Python module utilizing high-precision `mpmath` (up to 40 decimal places) to isolate non-trivial zeroes escaping the critical line (\(\mathcal{Re}(s) = 1/2\)), mathematically validating why finite computer brute-forcing is inherently blind to infinite functional proofs.
2. **Collatz Strong Induction:** A highly parallelized bitwise culling engine evaluating trajectory convergence, mapping peak excursions, and logging stopping times (σ(n)).

---
*Developed as an independent research project exploring structural invariants and high-performance system design.*
# High-Performance Computing (HPC) Simulation Engine

An asynchronous numerical simulation engine built to audit the boundaries of computational verification regarding the Riemann Hypothesis and the Collatz Conjecture.

## 🛠️ Technical Architecture & Optimization
- **Memory Footprint:** Operates strictly under a continuous data-stream pipeline with an O(1) memory footprint, maintaining a Resident Set Size (RSS) bounded at **1.6 MB** to prevent memory leaks or out-of-memory crashes on large datasets.
- **Asynchronous Safepoints:** Implements periodic local JSON state serialization checkpoints every 10 wall-clock seconds to handle system timeouts natively.
- **Hardware Acceleration:** C-based verification modules optimized using OpenMP multithreading directives (`-O3 -march=native -fopenmp`) and native __uint128_t precision handling to mitigate mathematical overflow limits.

## 🔬 Mathematical Protocols Implemented
1. **Davenport-Heilbronn Counter-Example Proof:** A custom Python module utilizing high-precision `mpmath` (up to 40 decimal places) to isolate non-trivial zeroes escaping the critical line (\(\mathcal{Re}(s) = 1/2\)), mathematically validating why finite computer brute-forcing is inherently blind to infinite functional proofs.
2. **Collatz Strong Induction:** A highly parallelized bitwise culling engine evaluating trajectory convergence, mapping peak excursions, and logging stopping times (σ(n)).

---
*Developed as an independent research project exploring structural invariants and high-performance system design.*
