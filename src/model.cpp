#include "../include/model.h"

void Model::fit(const Tensor& real, const Tensor& input, const Tensor& valid_real, const Tensor& valid_input, const int epochs, const float lr)
{
    Timer timer;
    std::cout << "\n____________________________________________";
    std::cout << "\nBeginning training\n\n";

    float loss, val_loss;
    const Tensor* y_ptr = nullptr;
    Tensor* dy_ptr = nullptr;
    Tensor dy;
    const Tensor* val_pred_ptr = nullptr;
    for (int epoch = 0; epoch < epochs; epoch++) 
    {
        Timer timer;
    
        // train
        y_ptr = &input;
        for (Layer* layer : network)
        {
            y_ptr = (*layer).forward_pass(*y_ptr);
        }

        // loss calc
        if (!dy_ptr) dy = *y_ptr;

        loss = wef::categoricalcrossentropy(real, *y_ptr, dy);
        std::cout << "epoch: " << epoch + 1 << "\n\ttrain_loss = " << loss << "\n";
        
        // validation
        val_pred_ptr = &valid_input;
        for (Layer* layer : network) val_pred_ptr = (*layer).forward_pass(*val_pred_ptr, false);
        val_loss = wef::categoricalcrossentropy(valid_real, *val_pred_ptr);
        std::cout << "\tvalid_loss = " << val_loss << "\n";

        // backprop
        dy_ptr = &dy;
        for (int i = (int)network.size() - 1; i >= 0; i--) {
            dy_ptr = (*network[i]).backward_pass(*dy_ptr, lr);
        }

        std::cout << "\ttime per epoch = ";
    }

    std::cout << "\n____________________________________________";
    std::cout << "\nTraining complete";
    std::cout << "\nTotal training time = ";
}

void Model::fit(const Tensor& real, const Tensor& input, const int epochs, const float lr)
{
    Timer timer;
    std::cout << "\n____________________________________________";
    std::cout << "\nBeginning training\n\n";

    float loss;
    const Tensor* y_ptr = nullptr;
    Tensor* dy_ptr = nullptr;
    Tensor dy;

    for (int epoch = 0; epoch < epochs; epoch++) 
    {
        Timer timer;
        
        // train
        y_ptr = &input;
        for (Layer* layer : network) y_ptr = (*layer).forward_pass(*y_ptr);

        // loss calc
        if (!dy_ptr) dy = *y_ptr;

        loss = wef::categoricalcrossentropy(real, *y_ptr, dy);
        std::cout << "epoch: " << epoch + 1 << "\n\tloss = " << loss << "\n";

        // backprop
        dy_ptr = &dy;
        for (int i = (int)network.size() - 1; i >= 0; i--) {
            dy_ptr = (*network[i]).backward_pass(*dy_ptr, lr);
        }

        std::cout << "\ttime per epoch = ";
    }

    std::cout << "\n____________________________________________";
    std::cout << "\nTraining complete";
    std::cout << "\nTotal training time = ";
}

Tensor Model::predict(const Tensor& input)
{ 
    const Tensor* y_ptr = &input;
    for (Layer* layer : network) {y_ptr = (*layer).forward_pass(*y_ptr, false);}
    return *y_ptr;
}

void Model::summary()
{
    int nP = 0;
    for (Layer* l : network)
    {
        std::cout << "____________________________________________\n" << l->m_name << "\n\tparameters: " << l->m_num_param;
        nP += l->m_num_param;

        std::cout << "\n\toutput shape: [ ";
        for (int i = 0; i < l->m_out_rank; i++) std::cout << (l->m_out_shape)[i] << " ";
        std::cout << "]\n";
    }

    std::cout << "\nTotal number of parameters: " << nP << std::endl;
}