#pragma once

#include <iostream>
#include <random>
#include <cstring>
#include <unordered_map>
#include <vector>
#include "matrix_operations.h"

struct ValLayer
{
    void* layer = nullptr;
    const Tensor* val = nullptr;
};

class Layer
{

public: // TODO : make these private and have getters
    size_t m_num_param = 0;
    size_t m_out_rank = 0;
    const char* m_name;
    
    bool m_init= false;
    std::unique_ptr<size_t[]> m_out_shape;

public:
    virtual Tensor* forward_pass(const Tensor* px, const bool training=true, void* gpu=nullptr) = 0;
    virtual Tensor* backward_pass(const Tensor* dy, const float lr, void* gpu=nullptr) = 0;
    virtual ~Layer() = default;
    
    // Tensor* call(std::unordered_map<Layer*, std::vector<Layer*>>& layers, const std::vector<S>& s, const bool training=true, void* gpu=nullptr)
    // {
    //     for (const auto& input : s)
    //         layers[this].push_back(input.parent);

    //     return forward_pass(px, training, gpu);
    // }

    Layer* m_past;
    virtual ValLayer call(ValLayer px_l, const bool training=true, void* gpu=nullptr) // take px_l by value
    {

        // std::cout << "FORWARD: " << ((px_l->layer) ? ((Layer*)(px_l)->layer)->m_name : "None") << std::endl;
        m_past = (Layer*)(px_l).layer;
        px_l.val = forward_pass(px_l.val, training, gpu);
        px_l.layer = this;
        return px_l;
    }
    virtual void rev(Tensor* dy, const float lr, void* gpu=nullptr)
    {
        // std::cout << "BACKWARD: " << (m_past ? m_past->m_name : "None") << std::endl;

        if (m_past)
            m_past->rev(backward_pass(dy, lr, gpu), lr, gpu);
        else
            backward_pass(dy, lr, gpu);
        
    }
};

class Linear : public Layer
{

private:
    size_t m_units;
    std::normal_distribution<float> m_dist;
    std::mt19937 m_g;
    bool m_use_bias = false;
    Tensor m_out, m_W, m_B, m_X, m_dx, m_dw, m_db;

public:
    // initilize weights
    Linear(size_t m_units, bool use_bias=false, size_t rand=3) 
        : m_units(m_units), m_use_bias(use_bias), m_g(rand)
        { m_name = "linear"; }
    
    Tensor* forward_pass(const Tensor* px, const bool training, void*) override;
    Tensor* backward_pass(const Tensor* dy, const float lr, void*) override;
};

class ReLU : public Layer
{

private:
    Tensor m_dx, m_X;

public:

    ReLU() { m_name = "RelU"; }
    Tensor* forward_pass(const Tensor* px, const bool training, void*) override;
    Tensor* backward_pass(const Tensor* dy, float, void*) override;
};

class Sigmoid : public Layer
{

private:
    Tensor m_dx, m_X;

public:
    Sigmoid() { m_name = "sigmoid"; }

    Tensor* forward_pass(const Tensor* px, const bool training, void*) override;
    Tensor* backward_pass(const Tensor* dy, float, void*) override;
};

class Conv2D : public Layer
{

private:
    std::mt19937 m_g;
    size_t m_k_height, m_k_width, m_units;
    size_t m_height, m_width, m_ch;
    std::normal_distribution<float> m_dist;
    Tensor m_W, m_X, m_out, m_dx, m_dw, m_B, m_db;
    bool m_use_bias = false;

public:
    // initilize weights
    Conv2D(size_t w_h, size_t w_w, size_t u, bool use_bias=false, size_t rand=3)
        : m_g(rand), m_k_height(w_h), m_k_width(w_w), m_units(u), m_use_bias(use_bias)
        { m_name = "Conv2D"; }
    
    Tensor* forward_pass(const Tensor* px, const bool training, void*) override;
    Tensor* backward_pass(const Tensor* dy, const float lr, void*) override;
};

class MaxPool2D : public Layer
{
    
private:
    size_t k_height, k_width;
    size_t m_height, m_width, m_ch;
    std::unique_ptr<size_t[]> argmax;
    Tensor m_X, m_out, m_dx;

public:
    // initilize weights
    MaxPool2D(size_t k_h, size_t k_w)
        : k_height(k_h), k_width(k_w) 
        { 
            if (k_h < 1 || k_w < 1) throw std::invalid_argument("kernel must be greater than 0");
            m_name = "MaxPool2D";
        }
    
    Tensor* forward_pass(const Tensor* px, const bool training, void*) override;
    Tensor* backward_pass(const Tensor* dy, const float lr, void*) override;
};

class ReduceSum : public Layer
{
private:
    Tensor m_X, m_out, m_dx;
    int m_ax;
    bool keepdims = false;

public:
    ReduceSum(int a, bool kd=false) 
        : m_ax(a), keepdims(kd)
        {
            if (a < 0) throw std::invalid_argument("axis connot be negative");
            m_name = "ReduceSum";
        }

    Tensor* forward_pass(const Tensor* px, const bool training, void*) override;
    Tensor* backward_pass(const Tensor* dy, float, void*)  override;
};

class LayerNorm : public Layer
{

private:

    Tensor m_X;
    int m_axis;
    float m_ax_val;
    float m_eps =  1e-5f;
    Tensor m_beta, m_gamma, m_mu, m_x_mu, m_var, m_inv_std, m_x_i_hat, m_y_i, m_d_gamma, m_d_beta, m_dx;

public:
    // initilize weights
    LayerNorm(const int ax, const float ep=1e-5f) 
        : m_axis(ax), m_eps(ep)
        {
            if (m_axis < 0)
                throw std::invalid_argument("axis connot be negative");
            m_name = "LayerNorm";
        }
    
    Tensor* forward_pass(const Tensor* px, const bool training, void*) override;
    Tensor* backward_pass(const Tensor* dy, const float lr, void*) override;
};

class Flatten : public Layer
{
private:
    Tensor m_X, m_out, m_dx;

public:
    Flatten() { m_name = "Flatten"; }

    Tensor* forward_pass(const Tensor* px, const bool training, void*) override;
    Tensor* backward_pass(const Tensor* dy, float, void*) override;
};

class Linear_Fast : public Layer
{
private:
    size_t m_units;
    std::normal_distribution<float> m_dist;
    std::mt19937 m_g;
    Tensor m_out, m_W, m_B, m_X, m_dx, m_dw, m_db;
    bool m_use_bias = false;

public:  
    // initilize weights
    Linear_Fast(size_t unit, bool use_bias=false, size_t rand=3) 
        : m_units(unit), m_use_bias(use_bias), m_g(rand)
        { m_name = "Linear"; }
    
    Tensor* forward_pass(const Tensor* px, const bool training, void*) override;
    Tensor* backward_pass(const Tensor* dy, const float lr, void*) override;
};

class Conv2D_Fast : public Layer
{

private:
    std::mt19937 m_g;
    size_t m_k_height, m_k_width, m_units;
    size_t m_height, m_width, m_ch;
    std::normal_distribution<float> m_dist;
    Tensor m_W, m_X, m_out, m_dx, m_dw, m_B, m_db;
    bool m_use_bias = false;

public:  
    // initilize weights
    Conv2D_Fast(size_t w_h, size_t w_w, size_t u, bool use_bias=false, size_t rand=3)
        : m_g(rand), m_k_height(w_h), m_k_width(w_w), m_units(u), m_use_bias(use_bias)
        {
            m_name = "Conv2D";
        }
    
    Tensor* forward_pass(const Tensor* px, const bool training, void*) override;
    Tensor* backward_pass(const Tensor* dy, const float lr, void*) override;
};

class Conv2D_legacy : public Layer
{

private:
    std::mt19937 m_g;
    size_t m_k_height, m_k_width, m_units;
    size_t m_height, m_width, m_ch;
    std::normal_distribution<float> m_dist;
    Tensor m_W, m_X, m_out, m_dx, m_dw, m_B, m_db;
    bool m_use_bias = false;

public:  
    // initilize weights
    Conv2D_legacy(size_t w_h, size_t w_w, size_t u, bool use_bias=false, size_t rand=3)
        : m_g(rand), m_k_height(w_h), m_k_width(w_w), m_units(u), m_use_bias(use_bias)
        { m_name = "Conv2D"; }
    
    Tensor* forward_pass(const Tensor* px, const bool training, void*);
    Tensor* backward_pass(const Tensor* dy, const float lr, void*);
};

class Conv2D_NR : public Layer
{

private:
    std::mt19937 m_g;
    size_t m_k_height, m_k_width, m_units;
    size_t m_height, m_width, m_ch;
    std::normal_distribution<float> m_dist;
    Tensor m_W, m_X, m_out, m_dx, m_dw, m_B, m_db;
    bool m_use_bias = false;

public:    
    // initilize weights
    Conv2D_NR(size_t w_h, size_t w_w, size_t u, bool use_bias=false, size_t rand=3)
        : m_g(rand), m_k_height(w_h), m_k_width(w_w), m_units(u), m_use_bias(use_bias)
        { m_name = "Conv2D"; }
    
    Tensor* forward_pass(const Tensor* px, const bool training, void*);
    Tensor* backward_pass(const Tensor* dy, const float lr, void*);
};

class Conv2D_GPU : public Layer
{

private:
    std::mt19937 m_g;
    size_t m_k_height, m_k_width, m_units;
    size_t m_height, m_width, m_ch;
    size_t m_WB_size;
    std::normal_distribution<float> m_dist;
    Tensor m_W, m_X, m_out, m_dx, m_dw, m_B, m_db;
    std::unique_ptr<float[]> m_WB;
    bool m_use_bias = false;

public:  
    // initilize weights
    Conv2D_GPU(size_t w_h, size_t w_w, size_t u, bool use_bias=false, size_t rand=3)
        : m_g(rand), m_k_height(w_h), m_k_width(w_w), m_units(u), m_use_bias(use_bias)
        {
            m_name = "Conv2D";
        }
    
    Tensor* forward_pass(const Tensor* px, const bool training, void* gpu) override;
    Tensor* backward_pass(const Tensor* dy, const float lr, void* gpu) override;
};

class Linear_GPU : public Layer
{

private:
    size_t m_units;
    std::normal_distribution<float> m_dist;
    std::mt19937 m_g;
    Tensor out, m_W, m_B, m_X, m_dx, m_dw, m_db;
    bool m_use_bias = false;

public:
    // initilize weights
    Linear_GPU(size_t unit, bool use_bias=false, size_t rand=3) 
        : m_units(unit), m_use_bias(use_bias), m_g(rand)
        { m_name = "linear"; }
    
    Tensor* forward_pass(const Tensor* px, const bool training, void* gpu) override;
    Tensor* backward_pass(const Tensor* dy, const float lr, void* gpu) override;
};

class MaxPool2D_GPU : public Layer
{
    
private:
    size_t k_height, k_width;
    size_t m_height, m_width, m_ch;
    std::unique_ptr<uint32_t[]> argmax;
    size_t m_argmax_len;
    Tensor m_X, m_out, m_dx;

public: 
    // initilize weights
    MaxPool2D_GPU(size_t k_h, size_t k_w)
        : k_height(k_h), k_width(k_w) 
        { 
            if (k_h < 1 || k_w < 1) throw std::invalid_argument("kernel must be greater than 0");
            m_name = "MaxPool2D";
        }
    
    Tensor* forward_pass(const Tensor* px, const bool training, void* gpu) override;
    Tensor* backward_pass(const Tensor* dy, const float lr, void* gpu) override;
};

class MHA : public Layer 
{

private:
    bool m_use_bias = false;
    bool m_self_attention = false;
    bool m_use_gpu = false;
    bool m_use_mask = false;
    float m_keys_dim;

    size_t m_d_model, m_num_heads, m_depth, m_seq_len_q, m_seq_len_k, m_seq_len_v, m_batch;

    Tensor m_aij, m_q, m_k, m_v, m_mask, m_output;
    Tensor m_daij, m_de, m_dq, m_dk, m_dv;
    
    Tensor *m_dq_in, *m_dk_in, *m_dv_in, *m_temp;

    Layer *m_past_k, *m_past_v;
    std::unique_ptr<Layer> m_wq, m_wk, m_wv, m_out_layer;

public:
    // initilize weights
    MHA(size_t d_model, bool self_attention=false, size_t num_heads=1, bool use_bias=false, bool use_mask=false, bool use_gpu=false) 
        : m_d_model(d_model), m_self_attention(self_attention), m_num_heads(num_heads), m_use_bias(use_bias), m_use_mask(use_mask), m_use_gpu(use_gpu)
    {
        m_name = "MHA"; 
        if (m_d_model % m_num_heads != 0)
            throw std::invalid_argument("tensor rank must be > 1");

        m_depth = m_d_model / m_num_heads;
        if (use_gpu)
        {
            m_wq = std::make_unique<Linear_GPU>(m_d_model, m_use_bias, 3);
            m_wk = std::make_unique<Linear_GPU>(m_d_model, m_use_bias, 3);
            m_wv = std::make_unique<Linear_GPU>(m_d_model, m_use_bias, 3);
            m_out_layer = std::make_unique<Linear_GPU>(m_d_model, m_use_bias, 3);
        }
        else
        {
            m_wq = std::make_unique<Linear_Fast>(m_d_model, m_use_bias, 3);
            m_wk = std::make_unique<Linear_Fast>(m_d_model, m_use_bias, 3);
            m_wv = std::make_unique<Linear_Fast>(m_d_model, m_use_bias, 3);
            m_out_layer = std::make_unique<Linear_Fast>(m_d_model, m_use_bias, 3);
        }
    }
    
    Tensor scaled_dot_product_attention(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor* mask, void* gpu);
    void split_heads(Tensor& x, size_t seq_len);
    void merge_heads(Tensor& x, size_t seq_len);

    Tensor* forward_pass(const Tensor* qkv_mask, const bool training=true, void* gpu=nullptr) override;
    Tensor* backward_pass(const Tensor* dy, const float lr, void*) override;

    ValLayer call(ValLayer px_l, const bool training=true, void* gpu=nullptr) override
    { return px_l; }; // TODO : print error?
    
    ValLayer call(ValLayer px_lq, ValLayer px_lk, ValLayer px_lv, const bool training=true, void* gpu=nullptr, ValLayer px_lm={})
    {
        m_past = (Layer*)px_lq.layer;
        m_past_k = (Layer*)px_lk.layer;
        m_past_v = (Layer*)px_lv.layer;

        px_lq.val = forward_pass((Tensor[4]){*px_lq.val, *px_lk.val, *px_lv.val, *px_lm.val}, training, gpu);
        px_lq.layer = this;
        return px_lq;
    }

    void rev(Tensor* dy, const float lr, void* gpu=nullptr) override
    {
        backward_pass(dy, lr, gpu);

        if (m_self_attention)
        {
            if (m_past)
                m_past->rev(&m_output, lr, gpu);
        }
        else
        {
            if (m_past)
            {
                if (m_past == m_past_k && m_past == m_past_v)
                {
                    m_output = *m_dq_in + *m_dk_in + *m_dv_in;
                    m_past->rev(&m_output, lr, gpu);
                }
                else if (m_past == m_past_k)
                {
                    if (m_past_v) m_past_v->rev(m_dv_in, lr, gpu);

                    m_output = *m_dq_in + *m_dk_in;
                    m_past->rev(&m_output, lr, gpu);
                    
                }
                else if (m_past == m_past_v)
                {
                    if (m_past_k) m_past_k->rev(m_dk_in, lr, gpu);

                    m_output = *m_dq_in + *m_dv_in;
                    m_past->rev(&m_output, lr, gpu);
                    
                }
                else if  (m_past_k == m_past_v)
                {
                    m_past->rev(m_dq_in, lr, gpu);
                    
                    m_output = *m_dk_in + *m_dv_in;
                    if (m_past_k) m_past_k->rev(&m_output, lr, gpu);
                }
                else
                {
                    m_past->rev(m_dq_in, lr, gpu);
                    if (m_past_k) m_past_k->rev(m_dk_in, lr, gpu);
                    if (m_past_v) m_past_v->rev(m_dv_in, lr, gpu);
                }
            }
        }
            
        
    }

};

class Embedding : public Layer
{

private:
    size_t m_vocab_size;
    size_t m_d_model;
    std::normal_distribution<float> m_dist;
    std::mt19937 m_g;
    Tensor m_out, m_W, m_X, m_dx, m_dw;

public:
    // initilize weights
    Embedding(size_t vocab_size, size_t d_model, size_t rand=3) 
        : m_vocab_size(vocab_size), m_d_model(d_model), m_g(rand)
        { m_name = "Embedding"; }
    
    Tensor* forward_pass(const Tensor* px, const bool training, void*) override;
    Tensor* backward_pass(const Tensor* dy, const float lr, void*) override;
};

