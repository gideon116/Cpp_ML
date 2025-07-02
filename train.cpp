#include <iostream>
#include "layers.h"
#include "tensor.h"

// TO DO: add a print shape method to tensor, its getting tedious

int main() {

    matrixOperations wf;
    
    Tensor input = {
        {
            {{1, 1, 1}, {1, 1, 9}, {1, 1, 9}, {1, 1, 9}, {1, 1, 9}},
            {{2, 2, 2}, {1, 5, 1}, {2, 2, 2}, {1, 5, 1}, {2, 2, 2}}
        },
        {
            {{1, 1, 1}, {1, 1, 9}, {1, 1, 9}, {1, 1, 9}, {1, 1, 9}},
            {{2, 2, 2}, {1, 5, 1}, {2, 2, 2}, {1, 5, 1}, {2, 2, 2}}
        },
    };

    Tensor real = {
        {
            1, 4
        },
        {
            1, 4
        }
    };

    input.printShape();
    real.printShape();
    std::cout << real.col << " " << real.row << " " << real.batch << " " << std::endl;

    double loss;
    double lr = 0.001;

    int units1 = 10;
    int units2 = 10;

    Linear layer1a(units1), layer2a(units2), layer3a(real.col);
    ReLU relu1, relu2;
    ReduceSum r1(2), r2(2);
    std::vector<Layer*> network = {&layer1a, &relu1, &layer2a, &relu2, &layer3a, &r1, &r2};
    
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

