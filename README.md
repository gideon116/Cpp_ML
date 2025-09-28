# Custom C++ Machine Learning Library and Model

A custom machine learning library written in C++.
If you have a GPU you can select between Vulkan or CUDA implementations (Vulkan = platform agnorstic; CUDA = NVIDIA GPUs)
If you'd rather run on a CPU, then you can use the custom CPU parallelization or OpenMP.

Some of the contents include:

* **`Tensor`** class that uses float32 datatype (this is generally good for models).
* **Tensor Math Operations** (matmul, softmax, argmax, crossentropy loss, etc.) each with CPU multithreaded or GPU versions.
* **Layers**: `Linear`, `Conv2D`, `MaxPool2D`, `ReLU`, `MHA` (Multi-Head Attention), `Embedding`, and more; each implements `forward_pass` and `backward_pass` (meaning it lets you do backpropagation).
* **`Sequential Model`** container with for simple neural networks with `fit()`, `predict()` and other specializations. There is a MNIST image classifier made using this as a demo.
* **`Functional Model`** for complex architectures that is massively customizable. There is an **English to Spanish transfomer** to show how this works and a tempelate file that serves as a backbone for you to make a functional model of your own.
---

## Usage

```cpp
// Sequential model
Model model(true); // enable GPU
model.add(new Linear_GPU(128));
model.add(new ReLU());
model.add(new Linear_GPU(10));
model.fit(labels, images, epochs=100, lr=0.01f);
```

## Build

### Prerequisites

* C++11 or higher compiler (g++, clang++, or MSVC).
* OpenMP (optional).
* (ONLY if you are using Vulkan GPU version) Vulkan.
* (ONLY if you are using CUDA GPU version) CUDA.

### Make

```bash
git clone https://github.com/gideon116/Cpp_ML.git
cd Cpp_ML
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=release .. && make && ./wef
```

## Architecture

- `tensor.h` - Core tensor operations with CPU/GPU variants
- `layers.h` - Neural network layers (CPU: `Linear`, GPU: `Linear_GPU`, `Conv2D_GPU`, etc.)  
- `model.h` - Sequential model container with training loop
- `use_GPU.h` - Vulkan compute pipeline wrapper
- `transformer.cpp` - English→Spanish transformer example
- `functional_model_template.cpp` - Template for custom architectures