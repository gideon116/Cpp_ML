#include <iostream>
#include <random>
#include "matrix_operations.h"


using matrix2D = std::vector<std::vector<double>>;
using matrixType = std::vector<matrix2D>;

int main() {

    matrixOperations o;

    matrixType input = { // batch, 2, 3
        {{3.5, 3.69, 3.44}, {4.34, 4.42, 2.37}},
        {{3.5, 3.69, 3.43}, {4.34, 4.42, 2.37}},
    };

    matrixType real = { // batch, 2, 1
        {{18}, {3}},
        {{1}, {17}},
    };

    int batch = o.shape(input)[0];
    int rows = o.shape(input)[1];
    int cols = o.shape(input)[2];

    matrix2D w1( // 3, 2
        cols, std::vector<double>(rows)
    );

    matrix2D w2( // 2, 1
        rows, std::vector<double>(1)
    );

    // random init for weights
    std::mt19937 gen(42);
    std::normal_distribution<double> dist(0.0, 1.0/std::sqrt(cols));
    for (std::vector<double>& d1 : w1) {
        for (double& d2 : d1) {
            d2 = dist(gen);
        }
    }
    for (std::vector<double>& d1 : w2) {
        for (double& d2 : d1) {
            d2 = dist(gen);
        }
    }


    matrix2D w2T;
    matrixType inputT;
    matrixType aT;

    matrixType pred;
    matrixType dpred;
    matrix2D dw1;
    matrixType dlayer1;
    matrix2D dw2;

    matrixType layer1;
    matrixType a;
    matrixType da;
    double loss;

    double lr = 0.01;

    for (int i = 0; i < 100; i++) {
        layer1 = o.matmul(input, w1);  // b, 2, 2
        a = o.relu(layer1);
        pred = o.matmul(a, w2);  // b, 2, 1
        
        // backpropagation
        dpred = o.subtract(pred, real);  // b, 2, 1
        dpred = o.constMul(2.0 / (batch * rows), dpred);  // b, 2, 1

        aT = o.transpose(a);  // b, 2, 2
        aT = o.matmul(aT, dpred);  // b, 2, 1
        dw2 = o.sumBatch(aT);  // 2, 1

        w2T = o.transpose(w2);  // 1, 2
        da = o.matmul(dpred, w2T);  // b, 2, 2
        dlayer1 = o.d_relu(a);
        dlayer1 = o.elemwise(dlayer1, da);

        inputT = o.transpose(input);
        inputT = o.matmul(inputT, dlayer1);
        dw1 = o.sumBatch(inputT);

        // update weights
        dw1 = o.constMul(lr, dw1);
        dw2 = o.constMul(lr, dw2);

        w1 = o.subtract(w1, dw1);
        w2 = o.subtract(w2, dw2);

        // loss calc
        loss = o.l2(pred, real);

        std::cout << "epoch " << i << std::endl;
        std::cout << loss << std::endl;
        

    }

    return 0;
}