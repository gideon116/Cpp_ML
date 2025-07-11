#ifndef MODEL_H
#define MODEL_H

#include <iostream>
#include "layers.h"
#include "tensor.h"
#include "matrix_operations.h"

class Model
{
    public:
        Model() {}
        Model(std::vector<Layer*> inputNetwork) : network(inputNetwork) {}
        void add(Layer* i) { network.push_back(i); }
        void fit(const Tensor& real, const Tensor& input, const int& epochs, const double& lr);
        Tensor predict(Tensor& input)
        { 
            input.printShape();
            for (Layer* layer : network) {input = (*layer).forward_pass(input, wf);}
            return input; 
        }

    private:
        std::vector<Layer*> network;
        matrixOperations wf;
};

void Model::fit(const Tensor& real, const Tensor& input, const int& epochs, const double& lr)
{
    double loss;
    for (int epoch = 0; epoch < epochs; epoch++) 
    {
        // train
        Tensor y(input);
        for (Layer* layer : network) y = (*layer).forward_pass(y, wf);

        // loss calc
        Tensor dy(y);
        loss = wf.categoricalcrossentropy(real, y, dy);
        std::cout << "epoch: " << epoch << " loss = " << loss << std::endl;

        // backprop
        for (int i = (int)network.size() - 1; i >= 0; i--) {
            dy = (*network[i]).backward_pass(dy, lr, wf);
        }
    }
}
#endif