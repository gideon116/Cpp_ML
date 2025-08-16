#ifndef LAYERS_H
#define LAYERS_H

#include <iostream>
#include <random>
#include <cstring>
#include "matrix_operations.h"

class Layer {

    public:
        size_t m_num_param = 0;
        const char* m_name;
        std::unique_ptr<size_t[]> m_out_shape;
        size_t m_out_rank=0;

        bool init = false;

        virtual Tensor* forward_pass(const Tensor& px, const bool training=true, void* gpu=nullptr) = 0;
        virtual Tensor* backward_pass(const Tensor& dy, const float lr, void* gpu=nullptr) = 0;
        virtual ~Layer() = default;
};

class Linear : public Layer {

    public:
        size_t units;
        std::normal_distribution<float> dist;
        std::mt19937 g;
        Tensor out, W, B, X, dx, dw, db;
        bool usebias = false;

        
        // initilize weights
        Linear(size_t unit, bool use_bias=false, size_t rand=3) 
            : units(unit), usebias(use_bias), g(rand)
            { m_name = "linear"; }
        
        Tensor* forward_pass(const Tensor& px, const bool training, void*) override;
        Tensor* backward_pass(const Tensor& dy, const float lr, void*) override;
};

class ReLU : public Layer {

    public:
        Tensor dx, X;
        ReLU() { m_name = "RelU"; }

        Tensor* forward_pass(const Tensor& px, const bool training, void*) override {

            if (!init)
            {
                m_out_rank = px.rank;
                m_out_shape = std::make_unique<size_t[]>(m_out_rank);
                std::memcpy(m_out_shape.get(), px.shape.get(), m_out_rank * sizeof(size_t));
                init = true;
            }

            X = wef::relu(px);
            return &X;
        }

        Tensor* backward_pass(const Tensor& dy, float, void*) override
        {
            dx = wef::d_relu(X) * dy;
            return &dx;
        }
};

class sigmoid : public Layer {

    public:
        Tensor dx, X;

        sigmoid() { m_name = "sigmoid"; }

        Tensor* forward_pass(const Tensor& px, const bool training, void*) 
        override { 
            if (!init)
            {
                m_out_rank = px.rank;
                m_out_shape = std::make_unique<size_t[]>(m_out_rank);
                std::memcpy(m_out_shape.get(), px.shape.get(), m_out_rank * sizeof(size_t));
                init = true;
            }

            X = wef::sigmoid(px);
            return &X;
        }

        Tensor* backward_pass(const Tensor& dy, float, void*) override
        {
            dx = wef::d_sigmoid(X) * dy;
            return &dx;
        }
};

class Conv2D : public Layer {

    public:
        
        std::mt19937 g;
        
        size_t w_height, w_width, units;
        size_t height, width, ch;

        std::normal_distribution<float> dist;
        
        Tensor W, X, out, dx, dw, B, db;
        bool usebias = false;
        
        // initilize weights
        Conv2D(size_t w_h, size_t w_w, size_t u, bool use_bias=false, size_t rand=3)
            : g(rand), w_height(w_h), w_width(w_w), units(u), usebias(use_bias)
            { m_name = "Conv2D"; }
        
        Tensor* forward_pass(const Tensor& px, const bool training, void*) override;
        Tensor* backward_pass(const Tensor& dy, const float lr, void*) override;
    };

class MaxPool2D : public Layer {
    
    public:
        
        size_t k_height, k_width;
        size_t height, width, ch;

        std::unique_ptr<size_t[]> argmax;
        
        Tensor X, out, dx;
        
        // initilize weights
        MaxPool2D(size_t k_h, size_t k_w)
            : k_height(k_h), k_width(k_w) 
            { 
                if (k_h < 1 || k_w < 1) throw std::invalid_argument("kernel must be greater than 0");
                m_name = "MaxPool2D";
            }
        
        Tensor* forward_pass(const Tensor& px, const bool training, void*) override;
        Tensor* backward_pass(const Tensor& dy, const float lr, void*) override;
    };

class ReduceSum : public Layer {
    public:
        Tensor X, out, dx;
        int ax;
        bool keepdims = false;

        ReduceSum(int a, bool kd=false) 
            : ax(a), keepdims(kd)
            {
                if (a < 0) throw std::invalid_argument("axis connot be negative");
                m_name = "ReduceSum";
            }

        Tensor* forward_pass(const Tensor& px, const bool training, void*) override;
        Tensor* backward_pass(const Tensor& dy, float, void*)  override;
    };

class LayerNorm : public Layer {
    
    public:

        Tensor X;
        int axis;
        float ax_val;
        float eps = 0.01f;
        Tensor beta, gamma, mu, x_mu, var, inv_std, x_i_hat, y_i, d_gamma, d_beta, dx;

        // initilize weights
        LayerNorm(const int ax, const float ep=0.01f) 
            : axis(ax), eps(ep)
            {
                if (axis < 0) throw std::invalid_argument("axis connot be negative");
                m_name = "LayerNorm";
            }
        
        Tensor* forward_pass(const Tensor& px, const bool training, void*) override;
        Tensor* backward_pass(const Tensor& dy, const float lr, void*) override;
};

class Flatten : public Layer {
    
    public:
        Tensor X, out, dx;

        Flatten() { m_name = "Flatten"; }

        Tensor* forward_pass(const Tensor& px, const bool training, void*) override 
        {
            if (!init)
            {
                dx = Tensor::create(px.shape.get(), px.rank); // TODO : set shape only once, locked in after
                
                size_t flat = 1;
                for (size_t i = 1; i < px.rank; i++) flat *= px.shape[i];
                
                m_out_rank = 2; // TODO : hard code??
                m_out_shape = std::make_unique<size_t[]>(m_out_rank);
                m_out_shape[1] = flat;
                init = true;
            }
            else
            {
                // if trying to use (reuse) the layer on a different tensor
                if (training)
                    if (dx.tot_size != px.tot_size) // TODO : better to check shape but this is faster
                        throw std::invalid_argument("cannot reuse layer");
            }

            m_out_shape[0] = px.shape[0];
            out = Tensor::create(m_out_shape.get(), 2);
            memcpy(out.tensor.get(), px.tensor.get(), out.tot_size * sizeof(float));

            return &out;
        }
        Tensor* backward_pass(const Tensor& dy, float, void*) override 
        {
            memcpy(dx.tensor.get(), dy.tensor.get(), dx.tot_size * sizeof(float));
            return &dx;
        }
};

class Linear_Fast : public Layer {

    public:
        size_t units;
        std::normal_distribution<float> dist;
        std::mt19937 g;
        Tensor out, W, B, X, dx, dw, db;
        bool usebias = false;
        
        // initilize weights
        Linear_Fast(size_t unit, bool use_bias=false, size_t rand=3) 
            : units(unit), usebias(use_bias), g(rand)
            { m_name = "Linear"; }
        
        Tensor* forward_pass(const Tensor& px, const bool training, void*) override;
        Tensor* backward_pass(const Tensor& dy, const float lr, void*) override;
};

class Conv2D_Fast : public Layer {

    public:
        
        std::mt19937 g;
        
        size_t w_height, w_width, units;
        size_t height, width, ch;

        std::normal_distribution<float> dist;
        
        Tensor W, X, out, dx, dw, B, db;
        bool usebias = false;
        
        // initilize weights
        Conv2D_Fast(size_t w_h, size_t w_w, size_t u, bool use_bias=false, size_t rand=3)
            : g(rand), w_height(w_h), w_width(w_w), units(u), usebias(use_bias)
            {
                m_name = "Conv2D";
            }
        
        Tensor* forward_pass(const Tensor& px, const bool training, void*) override;
        Tensor* backward_pass(const Tensor& dy, const float lr, void*) override;
};

class Conv2D_legacy : public Layer {

    public:
        
        std::mt19937 g;
        
        size_t w_height, w_width, units;
        size_t height, width, ch;

        std::normal_distribution<float> dist;
        
        Tensor W, X, out, dx, dw, B, db;
        bool usebias = false;
        
        // initilize weights
        Conv2D_legacy(size_t w_h, size_t w_w, size_t u, bool use_bias=false, size_t rand=3)
            : g(rand), w_height(w_h), w_width(w_w), units(u), usebias(use_bias)
            { m_name = "Conv2D"; }
        
        Tensor* forward_pass(const Tensor& px, const bool training, void*);
        Tensor* backward_pass(const Tensor& dy, const float lr, void*);
    };

class Conv2D_NR : public Layer {

    public:
        
        std::mt19937 g;
        
        size_t w_height, w_width, units;
        size_t height, width, ch;

        std::normal_distribution<float> dist;
        
        Tensor W, X, out, dx, dw, B, db;
        bool usebias = false;
        
        // initilize weights
        Conv2D_NR(size_t w_h, size_t w_w, size_t u, bool use_bias=false, size_t rand=3)
            : g(rand), w_height(w_h), w_width(w_w), units(u), usebias(use_bias)
            { m_name = "Conv2D"; }
        
        Tensor* forward_pass(const Tensor& px, const bool training, void*);
        Tensor* backward_pass(const Tensor& dy, const float lr, void*);
    };

class Conv2D_GPU : public Layer {

    public:
        
        std::mt19937 g;
        
        size_t w_height, w_width, units;
        size_t height, width, ch;
        size_t WB_size;

        std::normal_distribution<float> dist;
        
        Tensor W, X, out, dx, dw, B, db;
        std::unique_ptr<float[]> WB;
        bool usebias = false;
        
        // initilize weights
        Conv2D_GPU(size_t w_h, size_t w_w, size_t u, bool use_bias=false, size_t rand=3)
            : g(rand), w_height(w_h), w_width(w_w), units(u), usebias(use_bias)
            {
                m_name = "Conv2D";
            }
        
        Tensor* forward_pass(const Tensor& px, const bool training, void* gpu) override;
        Tensor* backward_pass(const Tensor& dy, const float lr, void* gpu) override;
};

class Linear_GPU : public Layer {

    public:
        size_t units;
        std::normal_distribution<float> dist;
        std::mt19937 g;
        Tensor out, W, B, X, dx, dw, db;
        bool usebias = false;

        
        // initilize weights
        Linear_GPU(size_t unit, bool use_bias=false, size_t rand=3) 
            : units(unit), usebias(use_bias), g(rand)
            { m_name = "linear"; }
        
        Tensor* forward_pass(const Tensor& px, const bool training, void* gpu) override;
        Tensor* backward_pass(const Tensor& dy, const float lr, void* gpu) override;
};

class MaxPool2D_GPU : public Layer {
    
    public:
        
        size_t k_height, k_width;
        size_t height, width, ch;

        std::unique_ptr<uint32_t[]> argmax;
        size_t m_argmax_len;
        
        Tensor X, out, dx;
        
        // initilize weights
        MaxPool2D_GPU(size_t k_h, size_t k_w)
            : k_height(k_h), k_width(k_w) 
            { 
                if (k_h < 1 || k_w < 1) throw std::invalid_argument("kernel must be greater than 0");
                m_name = "MaxPool2D";
            }
        
        Tensor* forward_pass(const Tensor& px, const bool training, void* gpu) override;
        Tensor* backward_pass(const Tensor& dy, const float lr, void* gpu) override;
    };


#endif