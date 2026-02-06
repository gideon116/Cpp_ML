# wefml

C++ deep learning library with optional CUDA/Vulkan GPU acceleration, packaged for Python.

## Install

```
pip install wefml
```

Requires Python >= 3.8, CMake >= 3.15, C++17 compiler. CUDA or Vulkan are auto-detected at build time; CPU-only mode works out of the box.

## Quick Start

```python
import wefml
import numpy as np

a = wefml.Tensor.from_numpy(np.random.randn(2, 3).astype(np.float32))
b = wefml.Tensor.from_numpy(np.random.randn(3, 4).astype(np.float32))
c = wefml.ops.matmul(a, b)
print(c.numpy())
```

## Sequential Model

```python
model = wefml.Model([
    wefml.Linear(128, use_bias=True),
    wefml.ReLU(),
    wefml.Linear(10, use_bias=True),
])
model.fit(labels, inputs, epochs=10, lr=0.01)
pred = model.predict(test_inputs)
```

## Transformer (English-to-Spanish)

A built-in encoder-decoder transformer for seq2seq tasks. The bundled `english_spanish_tab.txt` is resolved automatically.

```python
import wefml

tok = wefml.Tokenizer()
tok.process("english_spanish_tab.txt", early_stop=200)   # use more for better results
data = wefml.prepare_translation_data(tok, val_size=50)

model = wefml.Transformer(
    vocab_size=data["vocab_size"],
    d_model=64,
    num_heads=4,
    batch_size=50,
)

model.train(
    data["enc"], data["dec"], data["target"],
    data["val_enc"], data["val_dec"], data["val_target"],
    epochs=1, lr=0.05,   # increase epochs for better results
)
print(model.loss)

test_input = data["val_enc"]  # single [1, maxlen] sample
output = model.generate(test_input, data["start_token"], data["end_token"], max_tokens=10)
print(output.numpy())
```

## GPU Mode

```python
print(wefml.gpu_backend())   # 'cuda', 'vulkan', or 'none'
wefml.use_gpu(True)          # Linear/Conv2D/MaxPool2D auto-pick GPU variants

model = wefml.Model([...], use_gpu=True)
# or
model = wefml.Transformer(vocab_size=vocab, use_gpu=True)
```

## API Reference

**GPU** -- `use_gpu(bool)`, `is_gpu_available()`, `is_cuda_available()`, `is_vulkan_available()`, `gpu_backend()`

**Tensor** -- `Tensor.from_numpy(arr)`, `Tensor.zeros(shape)`, `.numpy()`, `.shape`, `.rank`, `.size`, arithmetic (`+`, `-`, `*`, `/`)

**Ops** (`wefml.ops`) -- `matmul`, `matmul_mt`, `transpose`, `argmax`, `softmax`, `relu`, `sigmoid`, `reducesum`, `positional_encoding`, `l2`, `binarycrossentropy`

**Layers** -- `Linear`, `Conv2D`, `MaxPool2D`, `ReLU`, `Sigmoid`, `Flatten`, `LayerNorm`, `ReduceSum`, `Embedding`, `MHA`

**Model** -- `Model(layers, use_gpu)` with `.add()`, `.fit()`, `.predict()`, `.summary()`

**Transformer** -- `Transformer(vocab_size, d_model=128, num_heads=4, batch_size=50, use_gpu=False)` with `.train()`, `.generate()`, `.loss`

**Data** -- `Tokenizer().process(path, early_stop)`, `prepare_translation_data(tok, val_size)`, `load_mnist_images(path)`, `load_mnist_labels(path)`

## License

MIT
