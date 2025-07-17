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
        void fit(const Tensor& real, const Tensor& input, const int epochs=10, const double lr=0.01);
        void fit(const Tensor& real, const Tensor& input, 
            const Tensor& valid_real, const Tensor& valid_input, 
            const int epochs=10, const double lr=0.01);
        void summary();
        Tensor predict(const Tensor& input)
        { 
            Tensor output(input);
            for (Layer* layer : network) {output = (*layer).forward_pass(output, false);}
            return output; 
        }

    private:
        std::vector<Layer*> network;
};

void Model::fit(const Tensor& real, const Tensor& input, const int epochs, const double lr)
{
    double loss;
    Tensor y, dy;
    for (int epoch = 0; epoch < epochs; epoch++) 
    {
        // train
        y = input;
        for (Layer* layer : network) y = (*layer).forward_pass(y);

        // loss calc
        dy = y;
        loss = wef::categoricalcrossentropy(real, y, dy);
        std::cout << "epoch: " << epoch + 1 << " loss = " << loss << std::endl;

        // backprop
        for (int i = (int)network.size() - 1; i >= 0; i--) {
            dy = (*network[i]).backward_pass(dy, lr);
        }
    }
}

void Model::fit(const Tensor& real, const Tensor& input, 
            const Tensor& valid_real, const Tensor& valid_input, 
            const int epochs, const double lr)
{
    double loss, val_loss;
    Tensor y, dy, val_pred;
    for (int epoch = 0; epoch < epochs; epoch++) 
    {
        // train
        y = input;
        for (Layer* layer : network) y = (*layer).forward_pass(y);

        // loss calc
        dy = y;
        loss = wef::categoricalcrossentropy(real, y, dy);
        std::cout << "epoch: " << epoch + 1 << "\n\ttrain_loss = " << loss << "\n";

        // validation
        val_pred = valid_input;
        for (Layer* layer : network) val_pred = (*layer).forward_pass(val_pred, false);
        val_loss = wef::categoricalcrossentropy(valid_real, val_pred);
        std::cout << "\tvalid_loss = " << val_loss << "\n";

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