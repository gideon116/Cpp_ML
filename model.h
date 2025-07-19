#ifndef MODEL_H
#define MODEL_H

#include <iostream>
#include <chrono>
#include "layers.h"
#include "tensor.h"
#include "matrix_operations.h"

class Timer
{
    public:
        Timer() { m_start_point = std::chrono::high_resolution_clock::now(); }
        ~Timer()
        {
            m_end_point = std::chrono::high_resolution_clock::now();

            auto start = std::chrono::time_point_cast<std::chrono::milliseconds>(m_start_point);
            auto end = std::chrono::time_point_cast<std::chrono::milliseconds>(m_end_point);
            auto duration = end - start;
            double sec = duration.count() * 0.001;
            std::cout << sec << " seconds" << std::endl;
        }
        
    private:
        std::chrono::time_point<std::chrono::high_resolution_clock> m_start_point;
        std::chrono::time_point<std::chrono::high_resolution_clock> m_end_point;
};


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
    Timer timer;
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
    Timer timer;
    double loss, val_loss;
    Tensor y, dy, val_pred;
    for (int epoch = 0; epoch < epochs; epoch++) 
    {
        Timer timer;

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

        std::cout << "\ttime per epoch = ";
    }

    std::cout << "\n____________________________________________";
    std::cout << "\nTraining complete";
    std::cout << "\nTotal training time = ";
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