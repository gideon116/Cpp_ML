#include <iostream>
#include "layers.h"



using matrix2D = std::vector<std::vector<double>>;
using matrix3D = std::vector<matrix2D>;

int main() {

    matrixOperations wf;
    
    matrix3D m1 = { // batch, 2, 3
        {{3.5, 3.69, 3.44}, {4.34, 4.42, 2.37}},
        {{3.5, 3.69, 3.43}, {4.34, 4.42, 2.37}},
    };

    matrix3D m2 = { // batch, 2, 2
        {{18, 3}, {3, 3}},
        {{1, 17}, {1, 3}},
    };
    
    Tensor input(m1);
    Tensor real(m2);
    
    double loss;
    double lr = 0.01;

    int units1 = 10;
    int units2 = 10;

    Linear layer1a(input.col, units1), layer2a(units1, units2), layer3a(units2, real.col);
    ReLU relu1, relu2;
    std::vector<Layer*> network = {&layer1a, &relu1, &layer2a, &relu2, &layer3a};
    
    for (int epoch = 0; epoch < 10; epoch++) {
        
        // train
        Tensor y(input);
        for (Layer* layer : network) {
            y = (*layer).forward_pass(y, wf);
        }

        // loss calc
        loss = wf.l2(y, real);
        std::cout << "epoch: " << epoch << " loss = " << loss << std::endl;

        // backprop
        Tensor dy = (y - real) * 2.0 / (real.batch * real.row * real.col);

        for (int i = static_cast<int>(network.size()) - 1; i >= 0; i--) {
            dy = (*network[i]).backward_pass(dy, lr, wf);
        }
    }

    return 0;
}
