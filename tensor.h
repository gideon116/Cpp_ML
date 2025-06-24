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
        
        Tensor forward_pass(const Tensor& px, matrixOperations& wf) 
        override {
            if (!init) 
            {   
                W = Tensor::create({px.col, units});
                double* pm = W.tensor.get();
                for (size_t i = 0; i < size_t(px.col) * units; i++) pm[i] = dist(g);
                init = true;
            }
            else
            {
                if (W.row != px.col) throw std::invalid_argument("cannot reuse layer");
            }
            X = Tensor(px);
            return wf.matmul(px, W);
        }

        Tensor backward_pass(const Tensor& dy, const double lr, matrixOperations& wf) 
        override {
            // gradient wrt the layer below
            Tensor dx = wf.matmul(dy, wf.transpose(W));
            
            // gradient wrt weights
            Tensor dw = wf.batchsum(wf.matmul(wf.transpose(X), dy));

            W = wf.subtract(W, wf.constMul(dw, lr));
            
            return dx;
        }
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

#endif
