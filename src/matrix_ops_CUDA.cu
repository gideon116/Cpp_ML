#include "matrix_operations.h"

static uint32_t s_ceilDiv(uint32_t a, uint32_t b) { return (a + b - 1) / b; }

struct PC_E
{
    uint32_t operation; // 0 add, 1 sub, 2 mul, 3 div, 4 pow
    uint32_t size;
};

__global__ void k_elem(float* m1, float* m2, float* m, PC_E pc)
{
    uint xi = blockIdx.x * blockDim.x + threadIdx.x;

    if (xi < pc.size)
    {
        switch (pc.operation)
        {
            case 0:
                m[xi] = m1[xi] + m2[xi];
                break;
            case 1:
                m[xi] = m1[xi] - m2[xi];
                break;
            case 2:
                m[xi] = m1[xi] * m2[xi];
                break;
            case 3:
                m[xi] = m1[xi] / m2[xi];
                break;
            default:
                break;
        }
    }
}

__global__ void k_c_elem(float* m1, float constant, float* m, PC_E pc)
{
    uint xi = blockIdx.x * blockDim.x + threadIdx.x;

    if (xi < pc.size)
    {
        switch (pc.operation)
        {
            case 0:
                m[xi] = m1[xi] + constant;
                break;
            case 1:
                m[xi] = m1[xi] - constant;
                break;
            case 2:
                m[xi] = m1[xi] * constant;
                break;
            case 3:
                m[xi] = m1[xi] / constant;
                break;
            case 4:
                m[xi] = pow(m1[xi], constant);
                break;
            default:
                break;
        }
    }
}

struct PC_M
{
    uint32_t    m1_r, m1_c, m2_c,
                batch, 
                m1_stride, m2_stride, m_stride;
};

__global__ void k_matmul(float* m1, float* m2, float* m, PC_M pc)
{

    uint ci = blockIdx.x * blockDim.x + threadIdx.x;
    uint ri = blockIdx.y * blockDim.y + threadIdx.y;
    uint bi = blockIdx.z * blockDim.z + threadIdx.z;

    if (ri < pc.m1_r && ci < pc.m2_c && bi < pc.batch)
    {
        uint m1_i = bi * pc.m1_stride + ri * pc.m1_c;
        uint m2_i = bi * pc.m2_stride + ci;
        uint m_i = bi * pc.m_stride + ri * pc.m2_c + ci;

        float sum = 0.0;
        for (uint j = 0; j < pc.m1_c; j++)
            sum += m1[m1_i + j] * m2[m2_i + j * pc.m2_c];

        m[m_i] = sum;
    }
}

Tensor wef::elemwise_GPU(const void* gpu, const Tensor& m1, const Tensor& m2, const int operation/* 0 add, 1 sub, 2 mul, 3 div*/)
{
    // TODO : add broadcast ability
    if (m1.m_rank != m2.m_rank)
        throw std::invalid_argument("tensor 1 and tensor 2 must have the same shape");
    if (memcmp(m1.m_shape, m2.m_shape, sizeof(size_t) * m1.m_rank)) // compare shapes
        throw std::invalid_argument("matrix size mismatch [4]");

    PC_E push_constant;
    push_constant.operation = operation;
    push_constant.size = m1.m_size;

    Tensor m = m1;

    const uint32_t WGX = 256;
    const uint32_t WGY = 1;
    const uint32_t WGZ = 1;

    {
        float* a = m1.m_tensor;
        float* b = m2.m_tensor;
        float* c = m.m_tensor;

        float* a_gpu = nullptr;
        float* b_gpu = nullptr;
        float* c_gpu = nullptr;

        uint32_t gx = s_ceilDiv(m1.m_size, WGX);
        uint32_t gy = 1;
        uint32_t gz = 1;

        dim3 dimBlock(WGX, WGY, WGZ);
        dim3 dimGrid(gx, gy, gz);

        size_t bytes = sizeof(float) * m1.m_size;
        
        cudaMalloc(&a_gpu, bytes);
        cudaMalloc(&b_gpu, bytes);
        cudaMalloc(&c_gpu, bytes);

        cudaMemcpy(a_gpu, a, bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(b_gpu, b, bytes, cudaMemcpyHostToDevice);
        
        k_elem<<<dimGrid, dimBlock>>>(a_gpu, b_gpu, c_gpu, push_constant);
        
        cudaDeviceSynchronize();
        cudaMemcpy(c, c_gpu, bytes, cudaMemcpyDeviceToHost);

        cudaFree(a_gpu);
        cudaFree(b_gpu);
        cudaFree(c_gpu);
    }
    return m;
}

Tensor wef::c_elemwise_GPU(const void* gpu, const Tensor& m1, const float& constant, const int operation/* 0 add, 1 sub, 2 mul, 3 div, 4 pow*/)
{
    
    // TODO : add += capability
    PC_E push_constant;
    
    Tensor m = m1;

    push_constant.operation = operation;
    push_constant.size = m1.m_size;

    const uint32_t WGX = 256;
    const uint32_t WGY = 1;
    const uint32_t WGZ = 1;

    {
        float* a = m1.m_tensor;
        float* c = m.m_tensor;

        float* a_gpu = nullptr;
        float* c_gpu = nullptr;

        uint32_t gx = s_ceilDiv(m1.m_size, WGX);
        uint32_t gy = 1;
        uint32_t gz = 1;

        dim3 dimBlock(WGX, WGY, WGZ);
        dim3 dimGrid(gx, gy, gz);

        size_t bytes = sizeof(float) * m1.m_size;

        cudaMalloc(&a_gpu, bytes);
        cudaMalloc(&c_gpu, bytes);

        cudaMemcpy(a_gpu, a, bytes, cudaMemcpyHostToDevice);
        
        k_c_elem<<<dimGrid, dimBlock>>>(a_gpu, constant, c_gpu, push_constant);
        
        cudaDeviceSynchronize();
        cudaMemcpy(c, c_gpu, bytes, cudaMemcpyDeviceToHost);

        cudaFree(a_gpu);
        cudaFree(c_gpu);
    }
    return m;
}

Tensor wef::matmul_GPU(const void* gpu, const Tensor& m1, const Tensor& m2)
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

    PC_M push_constant;
    push_constant.m1_r = M;
    push_constant.m1_c = N;
    push_constant.m2_c = K;
    push_constant.batch = m1.m_size/(M*N); // assume no m1 bcast
    push_constant.m1_stride = M * N;
    push_constant.m2_stride = N * K * !bcast; // set to 0 to broadcast B across batches
    push_constant.m_stride = M * K;

    const uint32_t WGX = 16;
    const uint32_t WGY = 16;
    const uint32_t WGZ = 1;

    {
        float* a = m1.m_tensor;
        float* b = m2.m_tensor;
        float* c = m.m_tensor;

        float* a_gpu = nullptr;
        float* b_gpu = nullptr;
        float* c_gpu = nullptr;

        uint32_t gx = s_ceilDiv(K, WGX);
        uint32_t gy = s_ceilDiv(M, WGY);
        uint32_t gz = m1.m_size/(M*N);

        dim3 dimBlock(WGX, WGY, WGZ);
        dim3 dimGrid(gx, gy, gz);

        size_t sizeA = sizeof(float) * m1.m_size;
        size_t sizeB = sizeof(float) * m2.m_size;
        size_t sizeC = sizeof(float) * m.m_size;
        
        cudaMalloc(&a_gpu, sizeA);
        cudaMalloc(&b_gpu, sizeB);
        cudaMalloc(&c_gpu, sizeC);

        cudaMemcpy(a_gpu, a, sizeA, cudaMemcpyHostToDevice);
        cudaMemcpy(b_gpu, b, sizeB, cudaMemcpyHostToDevice);
        
        k_matmul<<<dimGrid, dimBlock>>>(a_gpu, b_gpu, c_gpu, push_constant);
        
        cudaDeviceSynchronize();
        cudaMemcpy(c, c_gpu, sizeC, cudaMemcpyDeviceToHost);

        cudaFree(a_gpu);
        cudaFree(b_gpu);
        cudaFree(c_gpu);
    }
    
    return m;
}