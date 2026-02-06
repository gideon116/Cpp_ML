"""Basic smoke tests for the wefml package."""
import numpy as np


def test_import():
    import wefml
    assert hasattr(wefml, "Tensor")
    assert hasattr(wefml, "Model")
    assert hasattr(wefml, "ops")


def test_gpu_flag():
    import wefml
    # is_gpu_available returns bool
    result = wefml.is_gpu_available()
    assert isinstance(result, bool)

    # CUDA/Vulkan specific checks
    assert isinstance(wefml.is_cuda_available(), bool)
    assert isinstance(wefml.is_vulkan_available(), bool)
    assert wefml.gpu_backend() in ("cuda", "vulkan", "none")

    # Consistency: if either is available, is_gpu_available should be True
    if wefml.is_cuda_available() or wefml.is_vulkan_available():
        assert wefml.is_gpu_available()
    # If CUDA, backend should be 'cuda'
    if wefml.is_cuda_available():
        assert wefml.gpu_backend() == "cuda"


def test_tensor_from_numpy():
    import wefml
    arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    t = wefml.Tensor.from_numpy(arr)
    assert t.shape == [2, 3]
    assert t.rank == 2
    assert t.size == 6
    out = t.numpy()
    np.testing.assert_array_equal(out, arr)


def test_tensor_arithmetic():
    import wefml
    a = wefml.Tensor.from_numpy(np.ones((2, 3), dtype=np.float32))
    b = wefml.Tensor.from_numpy(np.ones((2, 3), dtype=np.float32) * 2)
    c = a + b
    np.testing.assert_array_almost_equal(c.numpy(), np.full((2, 3), 3.0))


def test_tensor_scalar():
    import wefml
    a = wefml.Tensor.from_numpy(np.ones((2, 3), dtype=np.float32))
    c = a * 5.0
    np.testing.assert_array_almost_equal(c.numpy(), np.full((2, 3), 5.0))


def test_matmul():
    import wefml
    a = wefml.Tensor.from_numpy(np.eye(3, dtype=np.float32))
    b = wefml.Tensor.from_numpy(np.array([[1], [2], [3]], dtype=np.float32))
    c = wefml.ops.matmul(a, b)
    np.testing.assert_array_almost_equal(c.numpy(), [[1], [2], [3]])


def test_transpose():
    import wefml
    a = wefml.Tensor.from_numpy(np.array([[1, 2], [3, 4]], dtype=np.float32))
    t = wefml.ops.transpose(a)
    np.testing.assert_array_almost_equal(t.numpy(), [[1, 3], [2, 4]])


def test_relu():
    import wefml
    a = wefml.Tensor.from_numpy(np.array([[-1, 2], [-3, 4]], dtype=np.float32))
    r = wefml.ops.relu(a)
    np.testing.assert_array_almost_equal(r.numpy(), [[0, 2], [0, 4]])


def test_model_creation():
    import wefml
    model = wefml.Model()
    model.summary()


if __name__ == "__main__":
    test_import()
    test_gpu_flag()
    test_tensor_from_numpy()
    test_tensor_arithmetic()
    test_tensor_scalar()
    test_matmul()
    test_transpose()
    test_relu()
    test_model_creation()
    print("All tests passed!")
