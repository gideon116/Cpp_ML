#include <iostream>
#include <random>
#include "matrix_operations.h"
#include "layers.h"

using matrix2D = std::vector<std::vector<double>>;
using matrixType = std::vector<matrix2D>;

int main() {

    matrixOperations wf;

    matrixType input = { // batch, 2, 3
        {{3.5, 3.69, 3.44}, {4.34, 4.42, 2.37}},
        {{3.5, 3.69, 3.43}, {4.34, 4.42, 2.37}},
    };

    matrixType real = { // batch, 2, 2
        {{18, 3}, {3, 3}},
        {{1, 17}, {1, 3}},
    };

    int batch = wf.shape(input)[0];
    int rows = wf.shape(input)[1];
    int cols = wf.shape(input)[2];

    double loss;
    double lr = 0.01;

    Linear layer1a(cols, rows), layer2a(rows, 1), layer3a(1, rows);
    ReLU relu1, relu2;
    std::vector<Layer*> network = {&layer1a, &relu1, &layer2a, &relu2, &layer3a};

    for (int epoch = 0; epoch < 100; epoch++) {

        matrixType y = input;

        for (Layer* layer : network) {
            y = (*layer).forward_pass(y, wf);
        }

        matrixType dy = wf.constMul(
            2.0 / (batch * rows * cols),
            wf.subtract(y, real)
        );

        for (int i = static_cast<int>(network.size()) - 1; i >= 0; i--) {
            dy = (*network[i]).backward_pass(dy, lr, wf);
        }
        
        // loss calc
        loss = wf.l2(y, real);
        std::cout << "epoch " << epoch << std::endl;
        std::cout << loss << std::endl;
        
    }

    return 0;
}
