#include "example_models.h"
#include "tokenizer.h"
#include "layers.h"
#include "tensor.h"
#include "model.h"
#include <iostream>

class functional
{
private:
    size_t m_d_model = 128;
    float m_lr = 0.0f;
    float m_loss = 0.0f;
    Tensor m_dy;

    Linear_GPU ffn1{m_d_model, true, 7}, ffn2{m_d_model, true, 8}, ffn3{m_d_model, true, 8};
    ReLU relu1, relu2, relu3;

    UseGPU gpu; // if using gpu otherwise use void* gpu
    
public:

    functional(size_t d_model) : m_d_model(m_d_model) {}

private:

    ValLayer call(const Tensor& input, const Tensor& target, const bool& training)
    {
        ValLayer x = {nullptr, &input};

        x = ffn1.call(x, training, &gpu);
        x = relu1.call(x, training, &gpu);

        ValLayer h = ffn2.call(x, training, &gpu);
        h = relu1.call(h, training, &gpu);

        Tensor temp = *x.val + *h.val;
        x.val = &temp;

        x = ffn3.call(x, training, &gpu);
        x = relu3.call(x, training, &gpu);

        return x;
    }
    
    void backward(const Tensor& target, const ValLayer& pred)
    {
        m_loss = wef::categoricalcrossentropy(target, *(pred.val), m_dy);
        ((Layer*)pred.layer)->rev(&m_dy, m_lr, &gpu);
    }

    Tensor create_minibatch(const Tensor& original, const size_t& mini_batch_size)
    {
        size_t* mini_batch_shape = new size_t[original.m_rank];
        memcpy(mini_batch_shape, original.m_shape, sizeof(size_t) * original.m_rank);
        mini_batch_shape[0] = mini_batch_size;
        Tensor mini_batch = Tensor::create(mini_batch_shape, original.m_rank);
        delete[] mini_batch_shape;
        return mini_batch;
    }

public:
    
    void train(const Tensor& input, const Tensor& target, const Tensor& val_input, const Tensor& val_target, int epochs, float lr, size_t batch_size=0)
    {
        if (!input.m_size) // use this to add input checks
            throw std::invalid_argument("[ERROR ] because of xyz");

        Timer timer;
        std::cout << "\n____________________________________________\nBeginning training\n\n";

        m_lr = lr;

        
        // batching stuff
        if (!batch_size)
            batch_size = input.m_shape[0];
        size_t num_batches = input.m_shape[0] / batch_size; // drop remainder
        Tensor min_input = create_minibatch(input, batch_size);
        Tensor min_target = create_minibatch(target, batch_size);

        for (int epoch = 0; epoch < epochs; epoch++)
        {
            Timer timer;

            for (size_t b = 0; b < num_batches; b++)
            {
                memcpy(min_input.m_tensor, input.m_tensor + b * min_input.m_size, sizeof(float) * min_input.m_size);
                memcpy(min_target.m_tensor, target.m_tensor + b * min_target.m_size, sizeof(float) * min_target.m_size);
                
                ValLayer pred = call(min_input, min_target, true);
                if (!epoch)
                    m_dy = *(pred.val); // only set m_dy during epoch 0
                backward(min_target, pred);
            }
            
            std::cout << "epoch: " << epoch + 1 << "\n\tloss = " << m_loss << "\n";

            // validation
            ValLayer val_pred_ptr = call(val_input, val_target, false);
            float val_loss = wef::categoricalcrossentropy(val_target, *(val_pred_ptr.val));
            std::cout << "\tvalid_loss = " << val_loss << "\n";

            
            std::cout << "\ttime per epoch = ";
        }

        std::cout << "\n____________________________________________";
        std::cout << "\nTraining complete";
        std::cout << "\nTotal training time = ";
    }

};