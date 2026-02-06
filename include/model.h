#pragma once

#include <iostream>
#include <chrono>
#include "layers.h"
#include "tensor.h"
#include "matrix_operations.h"
#ifdef LEARNN_HAS_VULKAN
#include "use_GPU.h"
#endif
#include <mutex>

class Timer
{

public:
    Timer() { m_start_point = std::chrono::high_resolution_clock::now(); }
    ~Timer()
    {
        m_end_point = std::chrono::high_resolution_clock::now();

        auto start = std::chrono::time_point_cast<std::chrono::milliseconds>(m_start_point);
        auto end = std::chrono::time_point_cast<std::chrono::milliseconds>(m_end_point);
        auto duration = end - start;
        float sec = duration.count() * 0.001f;
        std::cout << sec << " sec" << "\n";
    }
    
private:
    std::chrono::time_point<std::chrono::high_resolution_clock> m_start_point;
    std::chrono::time_point<std::chrono::high_resolution_clock> m_end_point;
};

class Model
{
public:
    Model(bool use_gpu=false)
        : m_use_gpu(use_gpu)
        {
#ifdef LEARNN_HAS_VULKAN
            if (use_gpu) m_gpu = new UseGPU;
#elif defined(LEARNN_HAS_CUDA)
            // CUDA layers manage their own device memory; m_gpu stays nullptr
#endif
        }

    Model(std::vector<Layer*> inputNetwork, bool use_gpu=false)
        : m_network(inputNetwork), m_use_gpu(use_gpu)
        {
#ifdef LEARNN_HAS_VULKAN
            if (use_gpu) m_gpu = new UseGPU;
#elif defined(LEARNN_HAS_CUDA)
            // CUDA layers manage their own device memory; m_gpu stays nullptr
#endif
        }
        
    ~Model()
        {
#ifdef LEARNN_HAS_VULKAN
            if (m_use_gpu) delete (UseGPU*)m_gpu;
#endif
        }

    void add(Layer* i)
        { m_network.push_back(i); }

    // validation + training 
    void fit(
            const Tensor& real, const Tensor& input,
            const Tensor& valid_real, const Tensor& valid_input,
            const int epochs=10, const float lr=0.01f, size_t batch_size=0, std::vector<float>* logging=nullptr, std::vector<float>* val_logging=nullptr, std::mutex* m_=nullptr);
    
    // no validation
    void fit(const Tensor& real, const Tensor& input, const int epochs=10, const float lr=0.01f);

    // choose loss fn
    void fit(const Tensor& real, const Tensor& input, const int epochs, const float lr, const char* lossfn, const bool print=true);

    Tensor predict(const Tensor& input);
    void summary();

private:
    std::vector<Layer*> m_network;
    void* m_gpu = nullptr;
    bool m_use_gpu;
};

