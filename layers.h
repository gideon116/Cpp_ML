#include <iostream>
#include <random>
#include "matrix_operations.h"


using matrix2D = std::vector<std::vector<double>>;
using matrixType = std::vector<matrix2D>;

// random init for weights
std::mt19937 gen(4);

class Layer {
    public:
        virtual matrixType forward_pass(const matrixType& px, matrixOperations& wf) = 0;
        virtual matrixType backward_pass(const matrixType& dy, double lr, matrixOperations& wf) = 0;
        virtual ~Layer() = default; // destructor
};

class Linear : public Layer {

    public:
        matrix2D W;
        matrixType X;

        // initilize weights
        Linear(int rows, int cols, std::mt19937 &g=gen) : W(rows, std::vector<double>(cols)) {
            
            std::normal_distribution<double> dist(0.0, 1.0/std::sqrt(cols));
            for (std::vector<double>& r : W) {
                for (double& i : r) {
                    i = dist(g);
                }
            }
        }

        matrixType forward_pass(const matrixType& px, matrixOperations& wf) override { 
            X = px;
            return wf.matmul(px, W);
        }

        matrixType backward_pass(const matrixType& dy, double lr, matrixOperations& wf) override {
            
            // gradient wrt the layer below
            matrixType dx = wf.matmul(dy, wf.transpose(W));
            
            // gradient wrt weights
            matrix2D dw = wf.sumBatch(wf.matmul(wf.transpose(X), dy));

            W = wf.subtract(W, wf.constMul(lr, dw));
            
            
            return dx;
        }
};

class ReLU : public Layer {
    public:
        matrixType X;

        matrixType forward_pass(const matrixType& px, matrixOperations& wf) override { 
            X = wf.relu(px);
            return X;
        }

        matrixType backward_pass(const matrixType& dy, double, matrixOperations& wf) override {

            return wf.elemwise(wf.d_relu(X), dy);
        }
};
