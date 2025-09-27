#include "matrix_operations.h"

Tensor wef::elemwise_GPU(const void* gpu, const Tensor& m1, const Tensor& m2, const int operation/* 0 add, 1 sub, 2 mul, 3 div*/, float*, float*, float*)
{
    // TODO : add broadcast ability
    if (m1.m_rank != m2.m_rank)
        throw std::invalid_argument("tensor 1 and tensor 2 must have the same shape");
    if (memcmp(m1.m_shape, m2.m_shape, sizeof(size_t) * m1.m_rank)) // compare shapes
        throw std::invalid_argument("matrix size mismatch [4]");

    struct PC
    {
        uint32_t operation; // 0 add, 1 sub, 2 mul, 3 div
        uint32_t size;
    } push_constant;

    const char* spv_path =  "../shaders/binaries/elemwise.spv";
    VkDeviceSize bytes = sizeof(float) * m1.m_size;

    Tensor m = m1;

    push_constant.operation = operation;
    push_constant.size = m1.m_size;

    const uint32_t WG = 256;
    uint32_t gx = UseGPU::ceil_div(m1.m_size, WG);
    uint32_t gy = 1;
    uint32_t gz = 1;

    ((UseGPU*)gpu)->program({bytes, bytes}, {bytes}, {m1.m_tensor, m2.m_tensor}, {m.m_tensor}, spv_path, (void*)&push_constant, sizeof(push_constant), gx, gy, gz);
    
    return m;
}

Tensor wef::c_elemwise_GPU(const void* gpu, const Tensor& m1, const float& constant, const int operation/* 0 add, 1 sub, 2 mul, 3 div, 4 pow*/, float*, float*)
{
    
    // TODO : add += capability
    struct PC
    {
        uint32_t operation; // 0 add, 1 sub, 2 mul, 3 div, 4 pow
        uint32_t size;
        float constant;
    } push_constant;

    const char* spv_path =  "../shaders/binaries/c_elemwise.spv";
    VkDeviceSize bytes = sizeof(float) * m1.m_size;

    Tensor m = m1;

    push_constant.operation = operation;
    push_constant.size = m1.m_size;

    const uint32_t WG = 256;
    uint32_t gx = UseGPU::ceil_div(m1.m_size, WG);
    uint32_t gy = 1;
    uint32_t gz = 1;

    ((UseGPU*)gpu)->program({bytes, bytes}, {bytes}, {m1.m_tensor}, {m.m_tensor}, spv_path, (void*)&push_constant, sizeof(push_constant), gx, gy, gz);
    
    return m;
}

Tensor wef::matmul_GPU(const void* gpu, const Tensor& m1, const Tensor& m2, float*, float*, float*)
{
    if (m1.m_rank < 2 || m2.m_rank < 2)
        throw std::invalid_argument("tensor 1 and tensor 2 rank must be > 1");

    size_t M = m1.m_shape[m1.m_rank - 2];
    size_t N = m1.m_shape[m1.m_rank - 1];
    size_t K = m2.m_shape[m2.m_rank - 1];
    if (m2.m_shape[m2.m_rank - 2] != N)
        throw std::invalid_argument("matrix size mismatch [3]");

    bool bcast = false;
    if (m2.m_rank == 2) // check broadcast
        bcast = true;
    else // if m2 is not a simple matrix like a 2d weight then compare the whole tensors up to the last 2 elements
        if (memcmp(m1.m_shape, m2.m_shape, sizeof(size_t) * m1.m_rank - 2 * sizeof(size_t))) // compare everything but the last 2
            throw std::invalid_argument("matrix size mismatch [4]");

    std::unique_ptr<size_t[]> temp_shape = std::make_unique<size_t[]>(m1.m_rank);
    memcpy(temp_shape.get(), m1.m_shape, sizeof(size_t) * m1.m_rank);
    temp_shape[m1.m_rank - 1] = K;

    Tensor m = Tensor::create(temp_shape.get(), m1.m_rank);

    const char* spv_path =  "../shaders/binaries/matmul.spv";

    VkDeviceSize sizeA = sizeof(float) * m1.m_size;
    VkDeviceSize sizeB = sizeof(float) * m2.m_size;
    VkDeviceSize sizeC = sizeof(float) * m.m_size;

    struct PC
    {
        uint32_t    m1_r, m1_c, m2_c,
                    batch, 
                    m1_stride, m2_stride, m_stride;
    } push_constant;

    push_constant.m1_r = M;
    push_constant.m1_c = N;
    push_constant.m2_c = K;
    push_constant.batch = m1.m_size/(M*N); // assume no m1 bcast
    push_constant.m1_stride = M * N;
    push_constant.m2_stride = N * K * !bcast; // set to 0 to broadcast B across batches
    push_constant.m_stride = M * K;

    const uint32_t WGX = 16;
    const uint32_t WGY = 16;
    uint32_t gx = UseGPU::ceil_div(K, WGX);
    uint32_t gy = UseGPU::ceil_div(M, WGY);
    uint32_t gz = m1.m_size/(M*N);

    ((UseGPU*)gpu)->program({sizeA, sizeB}, {sizeC}, {m1.m_tensor, m2.m_tensor}, {m.m_tensor}, spv_path, (void*)&push_constant, sizeof(push_constant), gx, gy, gz);
    return m;
}