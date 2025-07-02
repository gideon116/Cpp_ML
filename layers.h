#ifndef LAYERS_H
#define LAYERS_H

#include <iostream>
#include <random>
#include "matrix_operations.h"

class Layer {
    public:
        virtual Tensor forward_pass(const Tensor& px, matrixOperations& wf) = 0;
        virtual Tensor backward_pass(const Tensor& dy, const double lr, matrixOperations& wf) = 0;
        virtual ~Layer() = default;
};

class Linear : public Layer {
    

    public:
        int units;
        std::normal_distribution<double> dist;
        std::mt19937 g;
        Tensor W;
        Tensor X;
        bool init = false;
        
        // initilize weights
        Linear(int unit, int rand=3) : units(unit), dist(0.0, 1.0/std::sqrt(units)), g(rand) {}
        
        Tensor forward_pass(const Tensor& px, matrixOperations& wf) override;
        Tensor backward_pass(const Tensor& dy, const double lr, matrixOperations& wf) override;
};

class ReLU : public Layer {
    public:
        Tensor X;

        Tensor forward_pass(const Tensor& px, matrixOperations& wf) 
        override { 
            X = wf.relu(px);
            return X;
        }

        Tensor backward_pass(const Tensor& dy, double, matrixOperations& wf)
        override {
            return wf.d_relu(X) * dy;
        }
};

class sigmoid : public Layer {
    public:
        Tensor X;

        Tensor forward_pass(const Tensor& px, matrixOperations& wf) 
        override { 
            X = wf.sigmoid(px);
            return X;
        }

        Tensor backward_pass(const Tensor& dy, double, matrixOperations& wf) 
        override {
            return wf.d_sigmoid(X) * dy;
        }
};


class Conv2D : public Layer {
    
    public:
        
        std::mt19937 g;
        
        int w_height;
        int w_width;
        int units;
        
        int height;
        int width;
        int ch;

        std::normal_distribution<double> dist;
        
        Tensor W;
        Tensor X;
        Tensor out;
        bool init = false;
        
        // initilize weights
        Conv2D(int w_h, int w_w, int u, int rand=3)
            : g(rand), w_height(w_h), w_width(w_w), units(u) {}
        
        Tensor forward_pass(const Tensor& px, matrixOperations& wf) override;
        Tensor backward_pass(const Tensor& dy, const double lr, matrixOperations& wf) override;
    };

class MaxPool2D : public Layer {
    
    public:
        
        int k_height;
        int k_width;
        
        int height;
        int width;
        int ch;

        std::unique_ptr<size_t[][4]> argmax;
        
        Tensor X;
        Tensor out;
        bool init = false;
        
        // initilize weights
        MaxPool2D(int k_h, int k_w)
            : k_height(k_h), k_width(k_w) { if (k_h < 1 || k_w < 1) throw std::invalid_argument("kernel must be greater than 0"); }
        
        Tensor forward_pass(const Tensor& px, matrixOperations& wf) override;
        Tensor backward_pass(const Tensor& dy, const double lr, matrixOperations& wf) override;
    };

class ReduceSum : public Layer {
    public:
        Tensor X;
        int ax;
        bool keepdims = false;
        bool init = false;
        int keepdims_rank;
        std::unique_ptr<int[]> keepdims_shape;
        std::unique_ptr<int[]> reshape_shape;

        ReduceSum(int a, bool kd=false) : ax(a), keepdims(kd) { if (ax < 0) throw std::invalid_argument("axis connot be negative"); }

        Tensor forward_pass(const Tensor& px, matrixOperations& wf) override;
        Tensor backward_pass(const Tensor& dy, double, matrixOperations& wf)  override;
    };

#endif