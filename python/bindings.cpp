// python/bindings.cpp  –  pybind11 bridge for the learnn library
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>

#include "tensor.h"
#include "matrix_operations.h"
#include "layers.h"
#include "model.h"
#include "tokenizer.h"
#include "mnist.h"
#include "transformer_model.h"

#ifdef LEARNN_HAS_VULKAN
#include "use_GPU.h"
#endif

namespace py = pybind11;

// ─── Helpers: numpy ↔ Tensor ─────────────────────────────

// Create a Tensor from a numpy array (copies data)
static Tensor tensor_from_numpy(py::array_t<float, py::array::c_style | py::array::forcecast> arr)
{
    py::buffer_info buf = arr.request();
    size_t rank = buf.ndim;
    std::vector<size_t> shape(rank);
    for (size_t i = 0; i < rank; i++)
        shape[i] = static_cast<size_t>(buf.shape[i]);

    Tensor t = Tensor::create(shape.data(), rank);
    std::memcpy(t.m_tensor, buf.ptr, t.m_size * sizeof(float));
    return t;
}

// Return a numpy array that **copies** the tensor data (safe)
static py::array_t<float> tensor_to_numpy(const Tensor& t)
{
    std::vector<py::ssize_t> shape(t.m_rank);
    for (size_t i = 0; i < t.m_rank; i++)
        shape[i] = static_cast<py::ssize_t>(t.m_shape[i]);
    
    py::array_t<float> arr(shape);
    std::memcpy(arr.mutable_data(), t.m_tensor, t.m_size * sizeof(float));
    return arr;
}

// ─── MODULE ──────────────────────────────────────────────
PYBIND11_MODULE(_learnn_core, m)
{
    m.doc() = "learnn – C++ deep learning library with optional Vulkan/CUDA GPU acceleration";

    // ── GPU availability flags ───────────────────────────
#ifdef LEARNN_HAS_VULKAN
    m.attr("HAS_VULKAN") = true;
#else
    m.attr("HAS_VULKAN") = false;
#endif

#ifdef LEARNN_HAS_CUDA
    m.attr("HAS_CUDA") = true;
#else
    m.attr("HAS_CUDA") = false;
#endif

    // ─── Tensor ──────────────────────────────────────────
    py::class_<Tensor>(m, "Tensor")
        .def(py::init<>())
        // create from shape
        .def_static("create", [](std::vector<size_t> shape) {
            return Tensor::create(shape.data(), shape.size());
        }, py::arg("shape"), "Create a zero-filled tensor with the given shape")
        // create from numpy
        .def_static("from_numpy", &tensor_from_numpy, py::arg("array"),
            "Create a Tensor from a numpy array")
        // to numpy
        .def("numpy", &tensor_to_numpy, "Return a copy of the tensor data as a numpy array")
        // shape / size / rank
        .def_property_readonly("rank", [](const Tensor& t){ return t.m_rank; })
        .def_property_readonly("size", [](const Tensor& t){ return t.m_size; })
        .def_property_readonly("shape", [](const Tensor& t){
            std::vector<size_t> s(t.m_shape, t.m_shape + t.m_rank);
            return s;
        })
        .def("print_shape", &Tensor::print_shape)
        // arithmetic
        .def("__add__", [](const Tensor& a, const Tensor& b){ return a + b; })
        .def("__sub__", [](const Tensor& a, const Tensor& b){ return a - b; })
        .def("__mul__", [](const Tensor& a, const Tensor& b){ return a * b; })
        .def("__truediv__", [](const Tensor& a, const Tensor& b){ return a / b; })
        .def("__add__", [](const Tensor& a, float s){ return a + s; })
        .def("__sub__", [](const Tensor& a, float s){ return a - s; })
        .def("__mul__", [](const Tensor& a, float s){ return a * s; })
        .def("__truediv__", [](const Tensor& a, float s){ return a / s; })
        .def("__radd__", [](const Tensor& a, float s){ return s + a; })
        .def("__rsub__", [](const Tensor& a, float s){ return s - a; })
        .def("__rmul__", [](const Tensor& a, float s){ return s * a; })
        .def("__repr__", [](const Tensor& t){
            std::string s = "Tensor(shape=[";
            for (size_t i = 0; i < t.m_rank; i++){
                if (i) s += ", ";
                s += std::to_string(t.m_shape[i]);
            }
            s += "], size=" + std::to_string(t.m_size) + ")";
            return s;
        })
    ;

    // ─── Matrix operations  (wef namespace) ──────────────
    auto wef_mod = m.def_submodule("ops", "Matrix / tensor operations");

    wef_mod.def("matmul", [](const Tensor& a, const Tensor& b){ return wef::matmul(a, b); },
        py::arg("a"), py::arg("b"), "Matrix multiply two tensors");
    wef_mod.def("matmul_mt", [](const Tensor& a, const Tensor& b, size_t threads){
        return wef::matmul(a, b, true, threads);
    }, py::arg("a"), py::arg("b"), py::arg("threads") = 0,
        "Multithreaded matrix multiply");
    wef_mod.def("transpose", [](const Tensor& a){ return wef::transpose(a); },
        py::arg("a"));
    wef_mod.def("argmax", &wef::argmax, py::arg("a"));
    wef_mod.def("softmax", &wef::softmax, py::arg("a"));
    wef_mod.def("relu", &wef::relu, py::arg("a"));
    wef_mod.def("sigmoid", &wef::sigmoid, py::arg("a"));
    wef_mod.def("reducesum", &wef::reducesum,
        py::arg("a"), py::arg("axis"), py::arg("keepdims") = true);
    wef_mod.def("positional_encoding", &wef::positional_encoding,
        py::arg("length"), py::arg("depth"));
    wef_mod.def("print", [](const Tensor& t){ wef::print(t); }, py::arg("tensor"));

    // loss functions
    wef_mod.def("l2", &wef::l2, py::arg("a"), py::arg("b"));
    wef_mod.def("binarycrossentropy", &wef::binarycrossentropy, py::arg("a"), py::arg("b"));
    wef_mod.def("categoricalcrossentropy",
        py::overload_cast<const Tensor&, const Tensor&, Tensor*>(&wef::categoricalcrossentropy),
        py::arg("real"), py::arg("pred"), py::arg("mask") = nullptr);

    // ─── Layer base (trampoline not needed – users don't subclass in Python) ──
    py::class_<Layer>(m, "Layer")
        .def_readonly("name", &Layer::m_name)
        .def_readonly("num_param", &Layer::m_num_param)
    ;

    // ─── Concrete layers ─────────────────────────────────
    py::class_<Linear, Layer>(m, "Linear")
        .def(py::init<size_t, bool, size_t>(),
            py::arg("units"), py::arg("use_bias") = false, py::arg("seed") = 3);

    py::class_<Linear_Fast, Layer>(m, "LinearFast")
        .def(py::init<size_t, bool, size_t>(),
            py::arg("units"), py::arg("use_bias") = false, py::arg("seed") = 3);

    py::class_<ReLU, Layer>(m, "ReLU")
        .def(py::init<>());

    py::class_<Sigmoid, Layer>(m, "Sigmoid")
        .def(py::init<>());

    py::class_<Conv2D, Layer>(m, "Conv2D")
        .def(py::init<size_t, size_t, size_t, bool, size_t>(),
            py::arg("kernel_h"), py::arg("kernel_w"), py::arg("units"),
            py::arg("use_bias") = false, py::arg("seed") = 3);

    py::class_<Conv2D_Fast, Layer>(m, "Conv2DFast")
        .def(py::init<size_t, size_t, size_t, bool, size_t>(),
            py::arg("kernel_h"), py::arg("kernel_w"), py::arg("units"),
            py::arg("use_bias") = false, py::arg("seed") = 3);

    py::class_<MaxPool2D, Layer>(m, "MaxPool2D")
        .def(py::init<size_t, size_t>(), py::arg("kernel_h"), py::arg("kernel_w"));

    py::class_<Flatten, Layer>(m, "Flatten")
        .def(py::init<>());

    py::class_<ReduceSum, Layer>(m, "ReduceSum")
        .def(py::init<int, bool>(), py::arg("axis"), py::arg("keepdims") = false);

    py::class_<LayerNorm, Layer>(m, "LayerNorm")
        .def(py::init<int, float>(), py::arg("axis"), py::arg("eps") = 1e-5f);

    py::class_<Embedding, Layer>(m, "Embedding")
        .def(py::init<size_t, size_t, size_t>(),
            py::arg("vocab_size"), py::arg("d_model"), py::arg("seed") = 3);

    // GPU layers (available when Vulkan or CUDA is compiled in)
#if defined(LEARNN_HAS_VULKAN) || defined(LEARNN_HAS_CUDA)
    py::class_<Linear_GPU, Layer>(m, "LinearGPU")
        .def(py::init<size_t, bool, size_t>(),
            py::arg("units"), py::arg("use_bias") = false, py::arg("seed") = 3);

    py::class_<Conv2D_GPU, Layer>(m, "Conv2DGPU")
        .def(py::init<size_t, size_t, size_t, bool, size_t>(),
            py::arg("kernel_h"), py::arg("kernel_w"), py::arg("units"),
            py::arg("use_bias") = false, py::arg("seed") = 3);

    py::class_<MaxPool2D_GPU, Layer>(m, "MaxPool2DGPU")
        .def(py::init<size_t, size_t>(), py::arg("kernel_h"), py::arg("kernel_w"));

    py::class_<MHA, Layer>(m, "MHA")
        .def(py::init<size_t, bool, size_t, bool, bool, bool>(),
            py::arg("d_model"),
            py::arg("self_attention") = false,
            py::arg("num_heads") = 1,
            py::arg("use_bias") = false,
            py::arg("use_mask") = false,
            py::arg("use_gpu") = false);
#else
    py::class_<MHA, Layer>(m, "MHA")
        .def(py::init<size_t, bool, size_t, bool, bool, bool>(),
            py::arg("d_model"),
            py::arg("self_attention") = false,
            py::arg("num_heads") = 1,
            py::arg("use_bias") = false,
            py::arg("use_mask") = false,
            py::arg("use_gpu") = false);
#endif

    // ─── Model ───────────────────────────────────────────
    py::class_<Model>(m, "Model")
        .def(py::init<bool>(), py::arg("use_gpu") = false)
        .def(py::init<std::vector<Layer*>, bool>(),
            py::arg("layers"), py::arg("use_gpu") = false,
            py::keep_alive<1, 2>())  // prevent layers from being GC'd
        .def("add", &Model::add, py::arg("layer"), py::keep_alive<1, 2>())
        .def("fit",
            py::overload_cast<const Tensor&, const Tensor&,
                              const Tensor&, const Tensor&,
                              int, float, size_t,
                              std::vector<float>*, std::vector<float>*, std::mutex*>(&Model::fit),
            py::arg("labels"), py::arg("inputs"),
            py::arg("val_labels"), py::arg("val_inputs"),
            py::arg("epochs") = 10, py::arg("lr") = 0.01f,
            py::arg("batch_size") = 0,
            py::arg("logging") = nullptr, py::arg("val_logging") = nullptr,
            py::arg("mutex") = nullptr,
            "Train with validation")
        .def("fit_simple",
            py::overload_cast<const Tensor&, const Tensor&, int, float>(&Model::fit),
            py::arg("labels"), py::arg("inputs"),
            py::arg("epochs") = 10, py::arg("lr") = 0.01f,
            "Train without validation")
        .def("fit_loss",
            py::overload_cast<const Tensor&, const Tensor&, int, float, const char*, bool>(&Model::fit),
            py::arg("labels"), py::arg("inputs"),
            py::arg("epochs") = 10, py::arg("lr") = 0.01f,
            py::arg("loss_fn") = "categoricalcrossentropy",
            py::arg("verbose") = true,
            "Train with a specified loss function")
        .def("predict", &Model::predict, py::arg("inputs"))
        .def("summary", &Model::summary)
    ;

    // ─── Tokenizer ───────────────────────────────────────
    py::class_<Tokenizer>(m, "Tokenizer")
        .def(py::init<>())
        .def("process", &Tokenizer::process,
            py::arg("filepath") = "english_spanish_tab.txt",
            py::arg("early_stop") = 0)
        .def("vocab_sample", &Tokenizer::vocab_sample)
        .def_readonly("english_vsize", &Tokenizer::english_vsize)
        .def_readonly("spanish_vsize", &Tokenizer::spanish_vsize)
        .def_readonly("maxlen", &Tokenizer::maxlen)
        .def_readonly("english_sen", &Tokenizer::english_sen)
        .def_readonly("spanish_sen", &Tokenizer::spanish_sen)
    ;

    // ─── MNIST loader ────────────────────────────────────
    m.def("load_mnist_images", &load_mnist_images,
        py::arg("path"), py::arg("max_items") = (size_t)1e19,
        "Load MNIST images from IDX file");
    m.def("load_mnist_labels", &load_mnist_labels,
        py::arg("path"), py::arg("max_items") = (size_t)1e19,
        "Load MNIST labels from IDX file");

    // ─── TransformerModel ────────────────────────────────
    py::class_<TransformerModel>(m, "TransformerModel")
        .def(py::init<size_t, size_t, size_t, size_t, bool>(),
            py::arg("vocab_size"),
            py::arg("d_model")    = 128,
            py::arg("num_heads")  = 4,
            py::arg("batch_size") = 50,
            py::arg("use_gpu")    = false)
        .def("train", &TransformerModel::train,
            py::arg("enc_input"),     py::arg("dec_input"),     py::arg("dec_target"),
            py::arg("val_enc_input"), py::arg("val_dec_input"), py::arg("val_dec_target"),
            py::arg("epochs"), py::arg("lr"),
            "Train the transformer on encoder/decoder pairs with validation")
        .def("generate", &TransformerModel::generate,
            py::arg("enc_input"),
            py::arg("start_token"),
            py::arg("end_token"),
            py::arg("max_tokens") = 10,
            "Auto-regressively generate from a single encoder input")
        .def("loss", &TransformerModel::loss,
            "Return the last training loss")
    ;
}
