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
import numpy as np

tok = wefml.Tokenizer()
tok.process("english_spanish_tab.txt", early_stop=4000)

vocab = max(tok.english_vsize, tok.spanish_vsize)
start_token = vocab + 1
end_token = start_token + 1
ml = tok.maxlen + 1

# -- build training tensors (enc, dec, target, val splits) --
n_train = len(tok.english_sen) - 50
n_val = 50

enc_np = np.zeros((n_train, ml), dtype=np.float32)
dec_np = np.zeros((n_train, ml), dtype=np.float32)
tar_np = np.zeros((n_train, ml), dtype=np.float32)

for i in range(n_train):
    es, ss = tok.english_sen[i], tok.spanish_sen[i]
    for j, t in enumerate(es): enc_np[i, j] = float(t)
    enc_np[i, len(es)] = end_token
    dec_np[i, 0] = start_token
    for j, t in enumerate(ss):
        dec_np[i, j + 1] = float(t)
        tar_np[i, j] = float(t)
    tar_np[i, len(ss)] = end_token

val_enc_np = np.zeros((n_val, ml), dtype=np.float32)
val_dec_np = np.zeros((n_val, ml), dtype=np.float32)
val_tar_np = np.zeros((n_val, ml), dtype=np.float32)

for i in range(n_val):
    idx = n_train + i
    es, ss = tok.english_sen[idx], tok.spanish_sen[idx]
    for j, t in enumerate(es): val_enc_np[i, j] = float(t)
    val_enc_np[i, len(es)] = end_token
    val_dec_np[i, 0] = start_token
    for j, t in enumerate(ss):
        val_dec_np[i, j + 1] = float(t)
        val_tar_np[i, j] = float(t)
    val_tar_np[i, len(ss)] = end_token

enc     = wefml.Tensor.from_numpy(enc_np)
dec     = wefml.Tensor.from_numpy(dec_np)
target  = wefml.Tensor.from_numpy(tar_np)
val_enc = wefml.Tensor.from_numpy(val_enc_np)
val_dec = wefml.Tensor.from_numpy(val_dec_np)
val_tar = wefml.Tensor.from_numpy(val_tar_np)

# -- train --
model = wefml.Transformer(
    vocab_size=vocab,
    d_model=128,
    num_heads=4,
    batch_size=50,
    use_gpu=False,
)
model.train(enc, dec, target, val_enc, val_dec, val_tar, epochs=10, lr=0.05)
print(model.loss)

# -- generate --
test_input = wefml.Tensor.from_numpy(val_enc_np[:1])
output = model.generate(test_input, start_token, end_token, max_tokens=10)
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

**Data** -- `Tokenizer().process(path, early_stop)`, `load_mnist_images(path)`, `load_mnist_labels(path)`

## License

MIT
