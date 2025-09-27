#include "model.h"
#include <thread>

void Model::fit(
            const Tensor& real, const Tensor& input,
            const Tensor& valid_real, const Tensor& valid_input,
            const int epochs, const float lr, size_t batch_size, std::vector<float>* logging, std::vector<float>* val_logging, std::mutex* m_)
{
    if (!batch_size)
        batch_size = input.m_shape[0];
    size_t num_batches = input.m_shape[0] / batch_size; // drop remainder

    Tensor dy;
    
    size_t* mini_batch_shape = new size_t[input.m_rank];
    memcpy(mini_batch_shape, input.m_shape, sizeof(size_t) * input.m_rank);
    mini_batch_shape[0] = batch_size;
    Tensor mini_input = Tensor::create(mini_batch_shape, input.m_rank);
    delete[] mini_batch_shape;

    size_t* r_mini_batch_shape = new size_t[real.m_rank];
    memcpy(r_mini_batch_shape, real.m_shape, sizeof(size_t) * real.m_rank);
    r_mini_batch_shape[0] = batch_size;
    Tensor mini_real = Tensor::create(r_mini_batch_shape, real.m_rank);
    delete[] r_mini_batch_shape;
    
    Timer timer;
    std::cout << "\n____________________________________________";
    std::cout << "\nBeginning training\n\n";
        
    for (int epoch = 0; epoch < epochs; epoch++) 
    {
        Timer timer;

        #if 1
        double loss = 0.0;
        for (size_t b = 0; b < num_batches; b++)
        {
            memcpy(mini_input.m_tensor, input.m_tensor + b * mini_input.m_size, sizeof(float) * mini_input.m_size);
            memcpy(mini_real.m_tensor, real.m_tensor + b * mini_real.m_size, sizeof(float) * mini_real.m_size);

            // train
            const Tensor* y_ptr = &mini_input;
            for (Layer* layer : m_network)
                y_ptr = (*layer).forward_pass(y_ptr, /*training=*/true, m_gpu);
            
            // loss calc
            if (epoch == 0 && b == 0) dy = *y_ptr;
        
            loss += wef::categoricalcrossentropy(mini_real, *y_ptr, dy);
            
            // backprop
            Tensor* dy_ptr = &dy;
            for (int i = (int)m_network.size() - 1; i >= 0; i--)
                dy_ptr = (*m_network[i]).backward_pass(dy_ptr, lr, m_gpu);

        }
        
        std::cout << "epoch: " << epoch + 1 << "\n\ttrain_loss = " << loss/num_batches << "\n";


        #else

        double loss = 0.0;

        // train
        const Tensor* y_ptr = &input;
        for (Layer* layer : m_network)
            y_ptr = (*layer).forward_pass(*y_ptr, /*training=*/true, m_gpu);
        
        // loss calc
        if (epoch == 0) dy = *y_ptr;

        loss = wef::categoricalcrossentropy(real, *y_ptr, dy);
        std::cout << "epoch: " << epoch + 1 << "\n\ttrain_loss = " << loss << "\n";

        // backprop
        Tensor* dy_ptr = &dy;
        for (int i = (int)m_network.m_size() - 1; i >= 0; i--) {
            dy_ptr = (*m_network[i]).backward_pass(*dy_ptr, lr, m_gpu);
        }

        #endif
        
        // validation
        const Tensor* val_pred_ptr = &valid_input;
        for (Layer* layer : m_network)
            val_pred_ptr = (*layer).forward_pass(val_pred_ptr, false, m_gpu);
        float val_loss = wef::categoricalcrossentropy(valid_real, *val_pred_ptr);
        std::cout << "\tvalid_loss = " << val_loss << "\n";

        if (m_)
        {
            m_->lock();
            logging->push_back(loss/num_batches);
            val_logging->push_back(val_loss);
            m_->unlock();
        }

        float acc = 0;
        for (int i = 0; i < valid_real.m_size; i++)
            acc += wef::argmax(wef::softmax(*val_pred_ptr)).m_tensor[i] == valid_real.m_tensor[i];
        acc /= valid_real.m_size;
        std::cout << "\tval_accuracy = " << acc << std::endl;
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
        for (Layer* layer : m_network) y_ptr = (*layer).forward_pass(y_ptr, true, m_gpu);

        // loss calc
        if (!dy_ptr) dy = *y_ptr;

        loss = wef::categoricalcrossentropy(real, *y_ptr, dy);
        std::cout << "epoch: " << epoch + 1 << "\n\tloss = " << loss << "\n";

        // backprop
        dy_ptr = &dy;
        for (int i = (int)m_network.size() - 1; i >= 0; i--) {
            dy_ptr = (*m_network[i]).backward_pass(dy_ptr, lr, m_gpu);
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
    for (Layer* layer : m_network) {y_ptr = (*layer).forward_pass(y_ptr, false, m_gpu);}
    return *y_ptr;
}

void Model::summary()
{
    int nP = 0;
    for (Layer* l : m_network)
    {
        std::cout << "____________________________________________\n" << l->m_name << "\n\tparameters: " << l->m_num_param;
        nP += l->m_num_param;

        std::cout << "\n\toutput shape: [ ";
        for (int i = 0; i < l->m_out_rank; i++) std::cout << (l->m_out_shape)[i] << " ";
        std::cout << "]\n";
    }

    std::cout << "\nTotal number of parameters: " << nP << std::endl;
}