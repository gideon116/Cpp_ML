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
        void summary();
        Tensor predict(Tensor& input)
        { 
            input.printShape();
            for (Layer* layer : network) {input = (*layer).forward_pass(input);}
            return input; 
        }

    private:
        std::vector<Layer*> network;
};

void Model::fit(const Tensor& real, const Tensor& input, const int& epochs, const double& lr)
{
    double loss;
    for (int epoch = 0; epoch < epochs; epoch++) 
    {
        // train
        Tensor y(input);
        for (Layer* layer : network) y = (*layer).forward_pass(y);

        // loss calc
        Tensor dy(y);
        loss = wef::categoricalcrossentropy(real, y, dy);
        std::cout << "epoch: " << epoch + 1 << " loss = " << loss << std::endl;

        // backprop
        for (int i = (int)network.size() - 1; i >= 0; i--) {
            dy = (*network[i]).backward_pass(dy, lr);
        }
    }
}

void Model::summary()
{
    size_t nP = 0;
    for (Layer* l : network)
    {
        nP += l->num_param;
    }

    std::cout << "Number of parameters: " << nP << std::endl;
}
#endif