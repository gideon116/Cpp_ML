# Cpp_ML

A custom machine learning micro-framework written **entirely in C++** with no third party dependencies, just the standard library, a C++14 capable compiler, and (optionally) OpenMP for parallelism.  
Tensors, layers, and training loops from first principles.

---

## Features

* **`Tensor` class** supporting arbitrary rank, broadcasting helpers, and arithmetic operators.
* **Matrix/tensor ops** (matmul, softmax, argmax, crossentropy, etc.) with optional OpenMP parallel sections.
* **Core layers**: `Linear`, `Conv2D`, `MaxPool2D`, `ReLU`, `Sigmoid`, `ReduceSum`; each implements `forward_pass` and `backward_pass`.
* **Sequential `Model` container** with a `fit()` for training loop and `predict()` for inference.
* **MNIST data loader** (binary format) for CNN demo.
---

## Getting Started

### Prerequisites

* C++14 or higher compiler (g++ 10+, clang 12+, or MSVC ≥19.29).  
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
