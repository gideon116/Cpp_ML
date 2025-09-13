#include "tensor.h"
#include "matrix_operations.h"
#include "kernel.h"
#include <iostream>

static uint32_t ceilDiv(uint32_t a, uint32_t b) { return (a + b - 1) / b; }

struct PC
{
    int inH, inW, inC;
};

__global__ void kernel(float* a_gpu, float* b_gpu, float* c_gpu, size_t N, PC pc)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < 2 && col < 3)
    {
        size_t elem = row * 3 + col;
        c_gpu[elem] = pc.inH + pc.inW + pc.inC;
    }
}

int k()
{

    Tensor m1 = {{1, 2, 3}, {4, 5, 6}};
    Tensor m2 = {{10, 11, 3}, {20, 21, 3}};

    Tensor m = m1;
    memset(m.m_tensor, 0, m.m_size * sizeof(float));

    size_t nums = m.m_size;
    dim3 dimBlock(16, 16, 1);
    dim3 dimGrid(ceilDiv(2, dimBlock.x), ceilDiv(3, dimBlock.y), 1);


    float* a = m1.m_tensor;
    float* b = m2.m_tensor;
    float* c = m.m_tensor;

    float* a_gpu = nullptr;
    float* b_gpu = nullptr;
    float* c_gpu = nullptr;

    PC push_constant;
    push_constant.inH = 1;
    push_constant.inW = 3;
    push_constant.inC = 9;

    // allocate gpu memory (essentially doing new for the cuda pointers)
    cudaMalloc(&a_gpu, m1.m_size * sizeof(float));
    cudaMalloc(&b_gpu, m2.m_size * sizeof(float));
    cudaMalloc(&c_gpu, m.m_size * sizeof(float));

    // copy inputs to gpu
    cudaMemcpy(a_gpu, a, m1.m_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b_gpu, b, m2.m_size * sizeof(float), cudaMemcpyHostToDevice);

    // operation on gpu
    kernel<<<dimGrid, dimBlock>>>(a_gpu, b_gpu, c_gpu, nums, push_constant);

    // cudaDeviceSynchronize();

    // copy res to cpu
    cudaMemcpy(c, c_gpu, m.m_size * sizeof(float), cudaMemcpyDeviceToHost);
    
    wef::print(m);
    
    std::cout << "sucess" << std::endl;

    cudaFree(a_gpu);
    cudaFree(b_gpu);
    cudaFree(c_gpu);

    return 0;
}