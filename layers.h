#ifndef LAYERS_H
#define LAYERS_H

#include <iostream>
#include <random>
#include <cstring>
#include "matrix_operations.h"

class Layer {
    public:
        virtual Tensor forward_pass(const Tensor& px) = 0;
        virtual Tensor backward_pass(const Tensor& dy, const double lr) = 0;
        virtual ~Layer() = default;
};

class Linear : public Layer {
    

    public:
        int units;
        std::normal_distribution<double> dist;
        std::mt19937 g;
        Tensor W, B, X, dx, dw, db;
        bool init = false;
        
        // initilize weights
        Linear(int unit, int rand=3) : units(unit), dist(0.0, 1.0/std::sqrt(units)), g(rand) {}
        
        Tensor forward_pass(const Tensor& px) override;
        Tensor backward_pass(const Tensor& dy, const double lr) override;
};

class ReLU : public Layer {
    public:
        Tensor X;

        Tensor forward_pass(const Tensor& px) 
        override { 
            X = wef::relu(px);
            return X;
        }

        Tensor backward_pass(const Tensor& dy, double)
        override {
            return wef::d_relu(X) * dy;
        }
};

class sigmoid : public Layer {
    public:
        Tensor X;

        Tensor forward_pass(const Tensor& px) 
        override { 
            X = wef::sigmoid(px);
            return X;
        }

        Tensor backward_pass(const Tensor& dy, double) 
        override {
            return wef::d_sigmoid(X) * dy;
        }
};


class Conv2D : public Layer {
    
    public:
        
        std::mt19937 g;
        
        int w_height, w_width, units;
        int height, width, ch;

        std::normal_distribution<double> dist;
        
        Tensor W, X, out, dx, dw;
        bool init = false;
        
        // initilize weights
        Conv2D(int w_h, int w_w, int u, int rand=3)
            : g(rand), w_height(w_h), w_width(w_w), units(u) {}
        
        Tensor forward_pass(const Tensor& px) override;
        Tensor forward_pass_multi(const Tensor& px);
        Tensor backward_pass(const Tensor& dy, const double lr) override;
        Tensor backward_pass_multi(const Tensor& dy, const double lr);
    };

class MaxPool2D : public Layer {
    
    public:
        
        int k_height, k_width;
        int height, width, ch;

        std::unique_ptr<size_t[]> argmax;
        
        Tensor X, out, dx;
        bool init = false;
        
        // initilize weights
        MaxPool2D(int k_h, int k_w)
            : k_height(k_h), k_width(k_w) { if (k_h < 1 || k_w < 1) throw std::invalid_argument("kernel must be greater than 0"); }
        
        Tensor forward_pass(const Tensor& px) override;
        Tensor backward_pass(const Tensor& dy, const double lr) override;
    };

class ReduceSum : public Layer {
    public:
        Tensor X, out_keepdims, out, dx;
        int ax, keepdims_rank;
        bool keepdims = false;
        bool init = false;

        std::unique_ptr<int[]> keepdims_shape;
        std::unique_ptr<int[]> reshape_shape;

        ReduceSum(int a, bool kd=false) : ax(a), keepdims(kd) { if (ax < 0) throw std::invalid_argument("axis connot be negative"); }

        Tensor forward_pass(const Tensor& px) override;
        Tensor backward_pass(const Tensor& dy, double)  override;
    };


class LayerNorm : public Layer {
    
    public:

        Tensor X;
        bool init = false;

        int axis;
        double ax_val;
        double eps = 0.01;
        Tensor beta, gamma, mu, x_mu, var, inv_std, x_i_hat, y_i, d_gamma, d_beta, dx;

        // initilize weights
        LayerNorm(const int ax, const double ep=0.01) : axis(ax), eps(ep) {}
        
        Tensor forward_pass(const Tensor& px) override;
        Tensor backward_pass(const Tensor& dy, const double lr) override;
};


class Flatten : public Layer {
    
public:
    Tensor X, out, dx;
    bool init = false;

    Tensor forward_pass(const Tensor& px) override 
    {
        X = Tensor(px);
        int flat = 1;
        for (int i = 1; i < px.rank; i++) flat *= px.shape[i];

        int out_shape[2] = {px.shape[0], flat};
        out = Tensor::create(out_shape, 2);
        double* out_ptr = out.tensor.get();
        double* px_ptr = px.tensor.get();
        for (size_t i = 0; i < out.tot_size; i++) out_ptr[i] = px_ptr[i];

        return out;
    }
    Tensor backward_pass(const Tensor& dy, double) override 
    {
        dx = Tensor::create(X.shape.get(), X.rank);
        double* dx_ptr = dx.tensor.get();
        double* dy_ptr = dy.tensor.get();
        for (size_t i = 0; i < dx.tot_size; i++) dx_ptr[i] = dy_ptr[i];

        return dx;
    }
};


#endif