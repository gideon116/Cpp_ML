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

        virtual Tensor forward_pass(const Tensor& px, const bool training=true) = 0;
        virtual Tensor backward_pass(const Tensor& dy, const float lr) = 0;
        virtual ~Layer() = default;
};

class Linear : public Layer {

    public:
        size_t units;
        std::normal_distribution<float> dist;
        std::mt19937 g;
        Tensor W, B, X, dx, dw, db;
        bool init = false;
        bool usebias = false;

        
        // initilize weights
        Linear(size_t unit, bool use_bias=false, size_t rand=3) 
            : units(unit), usebias(use_bias), dist(0.0f, std::sqrt(2.0f/units)), g(rand)
            { m_name = "linear"; }
        
        Tensor forward_pass(const Tensor& px, const bool training) override;
        Tensor backward_pass(const Tensor& dy, const float lr) override;
};

class ReLU : public Layer {

    public:
        Tensor X;

        ReLU() { m_name = "RelU"; }

        Tensor forward_pass(const Tensor& px, const bool training)
        override {
            if (training) X = wef::relu(px);
            return X;
        }

        Tensor backward_pass(const Tensor& dy, float)
        override {
            return wef::d_relu(X) * dy;
        }
};

class sigmoid : public Layer {

    public:
        Tensor X;

        sigmoid() { m_name = "sigmoid"; }

        Tensor forward_pass(const Tensor& px, const bool training) 
        override { 
            if (training) X = wef::sigmoid(px);
            return X;
        }

        Tensor backward_pass(const Tensor& dy, float) 
        override {
            return wef::d_sigmoid(X) * dy;
        }
};

class Conv2D : public Layer {

    public:
        
        std::mt19937 g;
        
        size_t w_height, w_width, units;
        size_t height, width, ch;

        std::normal_distribution<float> dist;
        
        Tensor W, X, out, dx, dw, B, db;
        bool init = false;
        bool usebias = false;
        
        // initilize weights
        Conv2D(size_t w_h, size_t w_w, size_t u, bool use_bias=false, size_t rand=3)
            : g(rand), w_height(w_h), w_width(w_w), units(u), usebias(use_bias)
            { m_name = "Conv2D"; }
        
        Tensor forward_pass(const Tensor& px, const bool training) override;
        Tensor forward_pass_legacy(const Tensor& px, const bool training);
        Tensor backward_pass(const Tensor& dy, const float lr) override;
        Tensor backward_pass_legacy(const Tensor& dy, const float lr);
    };

class MaxPool2D : public Layer {
    
    public:
        
        size_t k_height, k_width;
        size_t height, width, ch;

        std::unique_ptr<size_t[]> argmax;
        
        Tensor X, out, dx;
        bool init = false;
        
        // initilize weights
        MaxPool2D(size_t k_h, size_t k_w)
            : k_height(k_h), k_width(k_w) 
            { 
                if (k_h < 1 || k_w < 1) throw std::invalid_argument("kernel must be greater than 0");
                m_name = "MaxPool2D";
            }
        
        Tensor forward_pass(const Tensor& px, const bool training) override;
        Tensor backward_pass(const Tensor& dy, const float lr) override;
    };

class ReduceSum : public Layer {
    public:
        Tensor X, out_keepdims, out, dx;
        size_t ax, keepdims_rank;
        bool keepdims = false;
        bool init = false;

        std::unique_ptr<size_t[]> keepdims_shape;
        std::unique_ptr<size_t[]> reshape_shape;

        ReduceSum(size_t a, bool kd=false) 
            : ax(a), keepdims(kd)
            {
                if (ax < 0) throw std::invalid_argument("axis connot be negative");
                m_name = "ReduceSum";
            }

        Tensor forward_pass(const Tensor& px, const bool training) override;
        Tensor backward_pass(const Tensor& dy, float)  override;
    };

class LayerNorm : public Layer {
    
    public:

        Tensor X;
        bool init = false;

        size_t axis;
        float ax_val;
        float eps = 0.01f;
        Tensor beta, gamma, mu, x_mu, var, inv_std, x_i_hat, y_i, d_gamma, d_beta, dx;

        // initilize weights
        LayerNorm(const size_t ax, const float ep=0.01f) 
            : axis(ax), eps(ep)
            {
                m_name = "LayerNorm";
            }
        
        Tensor forward_pass(const Tensor& px, const bool training) override;
        Tensor backward_pass(const Tensor& dy, const float lr) override;
};

class Flatten : public Layer {
    
    public:
        Tensor X, out, dx;
        bool init = false;

        Flatten() { m_name = "Flatten"; }

        Tensor forward_pass(const Tensor& px, const bool training) override 
        {
            X = Tensor(px);
            size_t flat = 1;
            for (size_t i = 1; i < px.rank; i++) flat *= px.shape[i];

            size_t out_shape[2] = {px.shape[0], flat};
            out = Tensor::create(out_shape, 2);
            float* out_ptr = out.tensor.get();
            float* px_ptr = px.tensor.get();
            for (size_t i = 0; i < out.tot_size; i++) out_ptr[i] = px_ptr[i];

            return out;
        }
        Tensor backward_pass(const Tensor& dy, float) override 
        {
            dx = Tensor::create(X.shape.get(), X.rank);
            float* dx_ptr = dx.tensor.get();
            float* dy_ptr = dy.tensor.get();
            for (size_t i = 0; i < dx.tot_size; i++) dx_ptr[i] = dy_ptr[i];

            return dx;
        }
};

class Linear_Fast : public Layer {

    public:
        size_t units;
        std::normal_distribution<float> dist;
        std::mt19937 g;
        Tensor W, B, X, dx, dw, db;
        bool init = false;
        bool usebias = false;
        
        // initilize weights
        Linear_Fast(size_t unit, bool use_bias=false, size_t rand=3) 
            : units(unit), usebias(use_bias), dist(0.0f, std::sqrt(2.0f/units)), g(rand)
            { m_name = "Linear_Fast"; }
        
        Tensor forward_pass(const Tensor& px, const bool training) override;
        Tensor backward_pass(const Tensor& dy, const float lr) override;
};

class Conv2D_Fast : public Layer {

    public:
        
        std::mt19937 g;
        
        size_t w_height, w_width, units;
        size_t height, width, ch;

        std::normal_distribution<float> dist;
        
        Tensor W, X, out, dx, dw, B, db;
        bool init = false;
        bool usebias = false;
        
        // initilize weights
        Conv2D_Fast(size_t w_h, size_t w_w, size_t u, bool use_bias=false, size_t rand=3)
            : g(rand), w_height(w_h), w_width(w_w), units(u), usebias(use_bias)
            {
                m_name = "Conv2D_Fast";
            }
        
        Tensor forward_pass(const Tensor& px, const bool training) override;
        Tensor forward_pass_legacy(const Tensor& px, const bool training);
        Tensor backward_pass(const Tensor& dy, const float lr) override;
        Tensor backward_pass_legacy(const Tensor& dy, const float lr);
};


#endif