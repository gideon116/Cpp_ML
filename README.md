# Custom C++ Machine Learning Library and Model

A custom machine learning library written entirely in C++ with no third party dependencies. You can use the custom CPU parallelism or OpenMP. Some contents include:

* **`Tensor`** class supporting arbitrary rank, broadcasting helpers, and arithmetic operators.
* **Tensor Operations** (matmul, softmax, argmax, crossentropy, etc.) with optional multithreaded versions.
* **Core Layers**: `Linear`, `Conv2D`, `MaxPool2D`, `ReLU`, `Sigmoid`, `ReduceSum`; each implements `forward_pass` and `backward_pass`.
* Sequential **`Model`** container with a `fit()` for training loop and `predict()` for inference.
* **MNIST Data Loader** (binary format) for CNN demo.
---

### Prerequisites

* C++14 or higher compiler (g++, clang++, or MSVC).  
* OpenMP (optional).  

### Build and Run

```bash
git clone https://github.com/gideon116/Cpp_ML.git
cd Cpp_ML

# Build and run
g++ train.cpp src/*.cpp -pthread -O3 && ./a.out && rm a.out

# OpenMP build
g++ train.cpp src/*.cpp -pthread -O3 -fopenmp && ./a.out && rm a.out