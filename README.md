# Custom C++ Machine Learning Library and Model

A custom machine learning library written entirely in C++ with no third party dependencies.
If you have a GPU you can select between Vulkan or CUDA implementations (Vulkan = platform agnorstic; CUDA = NVIDIA GPUs)
If you'd rather run on a CPU, then uou can use the custom CPU parallelization or OpenMP.

Some contents include:

* **`Tensor`** class for float32 datatype (this is generally good for models).
* **Tensor Operations** (matmul, softmax, argmax, crossentropy loss, etc.) each with CPU multithreaded or GPU versions.
* **Core Layers**: `Linear`, `Conv2D`, `MaxPool2D`, `ReLU`, `Sigmoid`, `ReduceSum`, and more; each implements `forward_pass` and `backward_pass` (meaning it lets you do backpropagation).
* Sequential **`Model`** container with a `fit()` for training loop and `predict()` for inference among other specializations. There is a MNIST image classifier made using this as a demo.
* Functional **`Model`** tempelate that is massively custimizable. There is an **English to Spanish transfomer** to show how this works and a tempelate file that serves as a backbone for you to make a functional model of your own.
---

### Prerequisites

* C++11 or higher compiler (g++, clang++, or MSVC).
* OpenMP (optional).
* (ONLY if you are using Vulkan GPU version) Vulkan.
* (ONLY if you are using CUDA GPU version) CUDA.

### Build and Run

```bash
git clone https://github.com/gideon116/Cpp_ML.git
cd Cpp_ML
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=release .. && make && ./wef