# Custom C++ Machine Learning Library and Model

A custom machine learning micro-framework written entirely in C++ with no third party dependencies, just the standard library, a C++14 capable compiler, and (optionally) OpenMP for parallelism.  

---

## Features

* **`Tensor`** class supporting arbitrary rank, broadcasting helpers, and arithmetic operators.
* **Tensor Operations** (matmul, softmax, argmax, crossentropy, etc.) with optional OpenMP parallel sections.
* **Core Layers**: `Linear`, `Conv2D`, `MaxPool2D`, `ReLU`, `Sigmoid`, `ReduceSum`; each implements `forward_pass` and `backward_pass`.
* Sequential **`Model`** container with a `fit()` for training loop and `predict()` for inference.
* **MNIST Data Loader** (binary format) for CNN demo.
---

## Getting Started

### Prerequisites

* C++14 or higher compiler (g++, clang++, or MSVC).  
* OpenMP (optional but recommended for speed).  

### Build and Run

```bash
git clone https://github.com/gideon116/Cpp_ML.git
cd Cpp_ML

# simple build and run
g++ train.cpp src/*.cpp -pthread && ./a.out && rm a.out

# parallel build
g++ train.cpp src/*.cpp -pthread -O3 -fopenmp  && ./a.out && rm a.out