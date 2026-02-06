"""
wefml — A C++ deep learning library with optional GPU support (CUDA / Vulkan).

Usage:
    import wefml

    # Check GPU backend
    print(wefml.gpu_backend())     # 'cuda', 'vulkan', or 'none'
    print(wefml.is_gpu_available()) # True if any GPU backend compiled in

    # Enable/disable GPU globally
    wefml.use_gpu(True)   # or False for CPU-only

    # Create tensors
    t = wefml.Tensor.from_numpy(my_numpy_array)
    result = wefml.ops.matmul(t1, t2)

    # Build a model
    model = wefml.Model([
        wefml.Linear(128, use_bias=True),
        wefml.ReLU(),
        wefml.Linear(10, use_bias=True),
    ])
    model.fit(labels, inputs, epochs=10, lr=0.01)
"""

from wefml._api import (
    # Config
    use_gpu,
    is_gpu_available,
    is_cuda_available,
    is_vulkan_available,
    gpu_backend,
    # Tensor
    Tensor,
    # Ops
    ops,
    # Layers
    Linear,
    LinearFast,
    ReLU,
    Sigmoid,
    Conv2D,
    Conv2DFast,
    MaxPool2D,
    Flatten,
    ReduceSum,
    LayerNorm,
    Embedding,
    MHA,
    # GPU-specific layers (None when GPU unavailable)
    LinearGPU,
    Conv2DGPU,
    MaxPool2DGPU,
    # Model
    Model,
    # Data
    Tokenizer,
    load_mnist_images,
    load_mnist_labels,
    prepare_translation_data,
    # Transformer
    Transformer,
)

__version__ = "0.1.6"

__all__ = [
    "use_gpu",
    "is_gpu_available",
    "is_cuda_available",
    "is_vulkan_available",
    "gpu_backend",
    "Tensor",
    "ops",
    "Linear",
    "LinearFast",
    "ReLU",
    "Sigmoid",
    "Conv2D",
    "Conv2DFast",
    "MaxPool2D",
    "Flatten",
    "ReduceSum",
    "LayerNorm",
    "Embedding",
    "MHA",
    "LinearGPU",
    "Conv2DGPU",
    "MaxPool2DGPU",
    "Model",
    "Tokenizer",
    "load_mnist_images",
    "load_mnist_labels",
    "prepare_translation_data",
    "Transformer",
]
