# Cpp_ML

An experimental machine-learning micro-framework written **entirely in modern C++**—no third-party dependencies, just the standard library, a C++17-capable compiler, and (optionally) OpenMP for parallelism.  
Tensors, layers, and training loops from first principles.

---

## Features

* **`Tensor` class** supporting arbitrary rank, broadcasting helpers, and basic arithmetic operators.
* **Matrix/tensor ops** (mat-mul, soft-max, arg-max, cross-entropy, etc.) with optional OpenMP parallel sections.
* **Core layers**: `Linear`, `Conv2D`, `MaxPool2D`, `ReLU`, `Sigmoid`, `ReduceSum`; each implements `forward_pass` and `backward_pass`.
* **Simple sequential `Model` container** with a minimal `fit()` training loop and `predict()` helper.
* **MNIST data loader** (binary format) for quick CNN demos.
* Header-only **interface**, `.cpp` sources kept lean; everything compiles to a single executable.

---

## Getting Started

### Prerequisites

* C++17 compiler (g++ 10+, clang 12+, or MSVC ≥19.29).  
* OpenMP (optional but recommended for speed).  
* MNIST binary files in `mnist/`.

### Build & Run

```bash
git clone https://github.com/gideon116/Cpp_ML.git
cd Cpp_ML

# Simple build and run
g++ train.cpp tensor.cpp matrix_operations.cpp layers.cpp && ./a.out

# Parallel build
g++ -std=c++17 -O3 -fopenmp \
    train.cpp tensor.cpp matrix_operations.cpp layers.cpp \
    -o mnist_demo
