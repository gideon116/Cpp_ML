"""
wefml._api  — Pythonic wrapper around the C++ _learnn_core extension.

The key idea: a global `_USE_GPU` flag that automatically picks
GPU layers vs CPU layers, so the user just writes:

    wefml.use_gpu(True)
    layer = wefml.Linear(128, use_bias=True)   # → picks LinearGPU under the hood

GPU backend priority: CUDA > Vulkan > CPU-only.
"""

from __future__ import annotations

import os as _os
import numpy as np

_DATA_DIR = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "data")
from typing import List, Optional, Sequence

# The compiled C++ module
import wefml._learnn_core as _C

# ─── Global GPU toggle ────────────────────────────────────
_USE_GPU: bool = False


def use_gpu(enable: bool = True) -> None:
    """Enable or disable GPU acceleration globally.

    When enabled, layer constructors that have GPU variants (Linear, Conv2D,
    MaxPool2D) will automatically use the GPU implementation.

    The build-time backend is selected automatically:
    CUDA (if available) > Vulkan (if available) > CPU-only.
    """
    global _USE_GPU
    if enable and not (_C.HAS_VULKAN or _C.HAS_CUDA):
        raise RuntimeError(
            "This build of wefml was compiled without GPU support. "
            "Rebuild with CUDA toolkit or Vulkan SDK to use GPU acceleration."
        )
    _USE_GPU = enable


def is_gpu_available() -> bool:
    """Return True if the library was compiled with any GPU support (CUDA or Vulkan)."""
    return bool(_C.HAS_VULKAN or _C.HAS_CUDA)


def is_cuda_available() -> bool:
    """Return True if the library was compiled with CUDA GPU support."""
    return bool(_C.HAS_CUDA)


def is_vulkan_available() -> bool:
    """Return True if the library was compiled with Vulkan GPU support."""
    return bool(_C.HAS_VULKAN)


def gpu_backend() -> str:
    """Return the name of the active GPU backend ('cuda', 'vulkan', or 'none')."""
    if _C.HAS_CUDA:
        return "cuda"
    if _C.HAS_VULKAN:
        return "vulkan"
    return "none"


# ─── Tensor ───────────────────────────────────────────────
class Tensor:
    """Pythonic wrapper around the C++ Tensor.

    Create from numpy:
        t = Tensor.from_numpy(arr)
    Create zeros:
        t = Tensor.zeros([3, 4, 5])
    Convert back:
        arr = t.numpy()
    """

    def __init__(self, _cpp: Optional[_C.Tensor] = None):
        self._t = _cpp if _cpp is not None else _C.Tensor()

    # ── Constructors ──────────────────────────────────────
    @staticmethod
    def from_numpy(arr: np.ndarray) -> "Tensor":
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        return Tensor(_C.Tensor.from_numpy(arr))

    @staticmethod
    def zeros(shape: Sequence[int]) -> "Tensor":
        return Tensor(_C.Tensor.create(list(shape)))

    @staticmethod
    def create(shape: Sequence[int]) -> "Tensor":
        return Tensor(_C.Tensor.create(list(shape)))

    # ── Properties ────────────────────────────────────────
    @property
    def shape(self) -> List[int]:
        return self._t.shape

    @property
    def rank(self) -> int:
        return self._t.rank

    @property
    def size(self) -> int:
        return self._t.size

    # ── Conversion ────────────────────────────────────────
    def numpy(self) -> np.ndarray:
        return self._t.numpy()

    # ── Arithmetic ────────────────────────────────────────
    def __add__(self, other):
        if isinstance(other, Tensor):
            return Tensor(self._t + other._t)
        return Tensor(self._t + float(other))

    def __radd__(self, other):
        return Tensor(float(other) + self._t) if not isinstance(other, Tensor) else NotImplemented

    def __sub__(self, other):
        if isinstance(other, Tensor):
            return Tensor(self._t - other._t)
        return Tensor(self._t - float(other))

    def __rsub__(self, other):
        return Tensor(float(other) - self._t) if not isinstance(other, Tensor) else NotImplemented

    def __mul__(self, other):
        if isinstance(other, Tensor):
            return Tensor(self._t * other._t)
        return Tensor(self._t * float(other))

    def __rmul__(self, other):
        return Tensor(float(other) * self._t) if not isinstance(other, Tensor) else NotImplemented

    def __truediv__(self, other):
        if isinstance(other, Tensor):
            return Tensor(self._t / other._t)
        return Tensor(self._t / float(other))

    def __repr__(self):
        return f"wefml.Tensor(shape={self.shape}, size={self.size})"


# ─── Ops (wef namespace) ─────────────────────────────────
class _Ops:
    """Namespace for matrix / tensor operations."""

    @staticmethod
    def matmul(a: Tensor, b: Tensor) -> Tensor:
        return Tensor(_C.ops.matmul(a._t, b._t))

    @staticmethod
    def matmul_mt(a: Tensor, b: Tensor, threads: int = 0) -> Tensor:
        return Tensor(_C.ops.matmul_mt(a._t, b._t, threads))

    @staticmethod
    def transpose(a: Tensor) -> Tensor:
        return Tensor(_C.ops.transpose(a._t))

    @staticmethod
    def argmax(a: Tensor) -> Tensor:
        return Tensor(_C.ops.argmax(a._t))

    @staticmethod
    def softmax(a: Tensor) -> Tensor:
        return Tensor(_C.ops.softmax(a._t))

    @staticmethod
    def relu(a: Tensor) -> Tensor:
        return Tensor(_C.ops.relu(a._t))

    @staticmethod
    def sigmoid(a: Tensor) -> Tensor:
        return Tensor(_C.ops.sigmoid(a._t))

    @staticmethod
    def reducesum(a: Tensor, axis: int, keepdims: bool = True) -> Tensor:
        return Tensor(_C.ops.reducesum(a._t, axis, keepdims))

    @staticmethod
    def positional_encoding(length: int, depth: int) -> Tensor:
        return Tensor(_C.ops.positional_encoding(length, depth))

    @staticmethod
    def l2(a: Tensor, b: Tensor) -> float:
        return _C.ops.l2(a._t, b._t)

    @staticmethod
    def binarycrossentropy(a: Tensor, b: Tensor) -> float:
        return _C.ops.binarycrossentropy(a._t, b._t)

    @staticmethod
    def print(t: Tensor) -> None:
        _C.ops.print(t._t)


ops = _Ops()


# ─── Smart layer factories ───────────────────────────────
# These pick GPU or CPU variant based on the global _USE_GPU flag.


def Linear(units: int, use_bias: bool = False, seed: int = 3):
    """Create a Linear (dense) layer. Uses GPU variant when `use_gpu(True)`."""
    if _USE_GPU and hasattr(_C, "LinearGPU"):
        return _C.LinearGPU(units, use_bias, seed)
    return _C.LinearFast(units, use_bias, seed)


def LinearFast(units: int, use_bias: bool = False, seed: int = 3):
    """CPU-only multithreaded Linear layer."""
    return _C.LinearFast(units, use_bias, seed)


def ReLU():
    return _C.ReLU()


def Sigmoid():
    return _C.Sigmoid()


def Conv2D(kernel_h: int, kernel_w: int, units: int, use_bias: bool = False, seed: int = 3):
    """Create a Conv2D layer. Uses GPU variant when `use_gpu(True)`."""
    if _USE_GPU and hasattr(_C, "Conv2DGPU"):
        return _C.Conv2DGPU(kernel_h, kernel_w, units, use_bias, seed)
    return _C.Conv2DFast(kernel_h, kernel_w, units, use_bias, seed)


def Conv2DFast(kernel_h: int, kernel_w: int, units: int, use_bias: bool = False, seed: int = 3):
    return _C.Conv2DFast(kernel_h, kernel_w, units, use_bias, seed)


def MaxPool2D(kernel_h: int, kernel_w: int):
    """Create a MaxPool2D layer. Uses GPU variant when `use_gpu(True)`."""
    if _USE_GPU and hasattr(_C, "MaxPool2DGPU"):
        return _C.MaxPool2DGPU(kernel_h, kernel_w)
    return _C.MaxPool2D(kernel_h, kernel_w)


def Flatten():
    return _C.Flatten()


def ReduceSum(axis: int, keepdims: bool = False):
    return _C.ReduceSum(axis, keepdims)


def LayerNorm(axis: int, eps: float = 1e-5):
    return _C.LayerNorm(axis, eps)


def Embedding(vocab_size: int, d_model: int, seed: int = 3):
    return _C.Embedding(vocab_size, d_model, seed)


def MHA(
    d_model: int,
    self_attention: bool = False,
    num_heads: int = 1,
    use_bias: bool = False,
    use_mask: bool = False,
    use_gpu: bool = False,
):
    """Multi-Head Attention layer. Respects global GPU flag or explicit use_gpu param."""
    gpu = use_gpu or _USE_GPU
    return _C.MHA(d_model, self_attention, num_heads, use_bias, use_mask, gpu)


# GPU-specific layers (None when unavailable)
LinearGPU = getattr(_C, "LinearGPU", None)
Conv2DGPU = getattr(_C, "Conv2DGPU", None)
MaxPool2DGPU = getattr(_C, "MaxPool2DGPU", None)


# ─── Model ────────────────────────────────────────────────
class Model:
    """High-level Model wrapper.

    Usage:
        model = Model([
            wefml.Linear(128, use_bias=True),
            wefml.ReLU(),
            wefml.Linear(10, use_bias=True),
        ])
        model.fit(labels, inputs, epochs=10, lr=0.01)
        pred = model.predict(test_inputs)
    """

    def __init__(self, layers: Optional[List] = None, use_gpu: Optional[bool] = None):
        gpu = use_gpu if use_gpu is not None else _USE_GPU
        if layers:
            self._model = _C.Model(layers, gpu)
        else:
            self._model = _C.Model(gpu)

    def add(self, layer) -> None:
        self._model.add(layer)

    def fit(
        self,
        labels,
        inputs,
        val_labels=None,
        val_inputs=None,
        epochs: int = 10,
        lr: float = 0.01,
        batch_size: int = 0,
        loss_fn: Optional[str] = None,
        verbose: bool = True,
    ) -> None:
        """Train the model.

        If val_labels / val_inputs are provided, trains with validation.
        If loss_fn is provided (e.g. 'mse', 'categoricalcrossentropy'), uses that loss.
        Otherwise uses the default categorical cross-entropy.
        """
        lab = labels._t if isinstance(labels, Tensor) else labels
        inp = inputs._t if isinstance(inputs, Tensor) else inputs

        if val_labels is not None and val_inputs is not None:
            vl = val_labels._t if isinstance(val_labels, Tensor) else val_labels
            vi = val_inputs._t if isinstance(val_inputs, Tensor) else val_inputs
            self._model.fit(lab, inp, vl, vi, epochs, lr, batch_size)
        elif loss_fn is not None:
            self._model.fit_loss(lab, inp, epochs, lr, loss_fn, verbose)
        else:
            self._model.fit_simple(lab, inp, epochs, lr)

    def predict(self, inputs) -> Tensor:
        inp = inputs._t if isinstance(inputs, Tensor) else inputs
        return Tensor(self._model.predict(inp))

    def summary(self) -> None:
        self._model.summary()


# ─── Data utilities ───────────────────────────────────────
class Tokenizer:
    """English-Spanish tab-separated tokenizer."""

    def __init__(self):
        self._tok = _C.Tokenizer()

    def process(self, filepath: str = "english_spanish_tab.txt", early_stop: int = 0) -> None:
        # If the file doesn't exist at the given path, try the bundled copy
        if not _os.path.isfile(filepath):
            bundled = _os.path.join(_DATA_DIR, _os.path.basename(filepath))
            if _os.path.isfile(bundled):
                filepath = bundled
        self._tok.process(filepath, early_stop)

    def vocab_sample(self) -> None:
        self._tok.vocab_sample()

    @property
    def english_vsize(self) -> int:
        return self._tok.english_vsize

    @property
    def spanish_vsize(self) -> int:
        return self._tok.spanish_vsize

    @property
    def maxlen(self) -> int:
        return self._tok.maxlen

    @property
    def english_sen(self):
        return self._tok.english_sen

    @property
    def spanish_sen(self):
        return self._tok.spanish_sen


def load_mnist_images(path: str, max_items: int = int(1e9)) -> Tensor:
    return Tensor(_C.load_mnist_images(path, max_items))


def load_mnist_labels(path: str, max_items: int = int(1e9)) -> Tensor:
    return Tensor(_C.load_mnist_labels(path, max_items))


# ─── Transformer (seq2seq) ────────────────────────────────
class Transformer:
    """Encoder-decoder transformer for sequence-to-sequence tasks.

    Wraps the C++ TransformerModel which implements:
      - Token + positional embeddings
      - Encoder: self-attention → add & norm → FFN → add & norm
      - Decoder: masked self-attention → add & norm → cross-attention
                 → add & norm → FFN → add & norm → linear projection

    Works on both CPU and GPU. When use_gpu is None, uses the global
    wefml.use_gpu() setting.

    Usage:
        tok = wefml.Tokenizer()
        tok.process("english_spanish_tab.txt", early_stop=4000)

        model = wefml.Transformer(
            vocab_size=max(tok.english_vsize, tok.spanish_vsize),
            d_model=128,
            num_heads=4,
            batch_size=50,
        )
        model.train(enc, dec, target, val_enc, val_dec, val_target,
                     epochs=10, lr=0.05)
        output = model.generate(test_input, start_token, end_token)
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        num_heads: int = 4,
        batch_size: int = 50,
        use_gpu: Optional[bool] = None,
    ):
        gpu = use_gpu if use_gpu is not None else _USE_GPU
        self._model = _C.TransformerModel(vocab_size, d_model, num_heads, batch_size, gpu)

    def train(
        self,
        enc_input,
        dec_input,
        dec_target,
        val_enc_input,
        val_dec_input,
        val_dec_target,
        epochs: int = 10,
        lr: float = 0.05,
    ) -> None:
        """Train on encoder/decoder pairs with validation."""
        self._model.train(
            enc_input._t  if isinstance(enc_input, Tensor)  else enc_input,
            dec_input._t  if isinstance(dec_input, Tensor)  else dec_input,
            dec_target._t if isinstance(dec_target, Tensor) else dec_target,
            val_enc_input._t  if isinstance(val_enc_input, Tensor)  else val_enc_input,
            val_dec_input._t  if isinstance(val_dec_input, Tensor)  else val_dec_input,
            val_dec_target._t if isinstance(val_dec_target, Tensor) else val_dec_target,
            epochs, lr,
        )

    def generate(
        self,
        enc_input,
        start_token: int,
        end_token: int,
        max_tokens: int = 10,
    ) -> Tensor:
        """Auto-regressively generate from a single encoder input [1, max_len]."""
        inp = enc_input._t if isinstance(enc_input, Tensor) else enc_input
        return Tensor(self._model.generate(inp, start_token, end_token, max_tokens))

    @property
    def loss(self) -> float:
        """Return the last training loss."""
        return self._model.loss()
