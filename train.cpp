#if 1
#include "include/layers.h"
#include "include/tensor.h"
#include "include/model.h"
#include "include/mnist.h"


Tensor create_padding_mask(const Tensor& seq)
{
    Tensor mask = seq;
    for (size_t i = 0; i < mask.m_size; i++)
        if (mask.m_tensor[i] == 0)
            mask.m_tensor[i] == 1;
        else
            mask.m_tensor[i] == 0;
    mask.reshape((size_t[4]){mask.m_shape[0], 1, 1, mask.m_shape[2]}, 4);
    return mask;
}



class TrainingTimer
{
public:
    Timer timer;
    TrainingTimer()
    {
        std::cout << "\n____________________________________________";
        std::cout << "\nBeginning training\n\n";
    }
    ~TrainingTimer()
    {
        std::cout << "\n____________________________________________";
        std::cout << "\nTraining complete";
        std::cout << "\nTotal training time = ";
    }
};

class functional_model
{
public:
    functional_model() {}

private:
    
    const size_t m_units1 = 16;
    const size_t m_units2 = 16;
    const size_t m_units5 = 10;

    Conv2D_Fast cov1{3, 3, m_units1, true, 4}, cov2{3, 3, m_units2, true, 1}, cov3{3, 3, m_units2, true, 2};
    Linear out{m_units5, true, 7}, ffn1{16, true, 8}, ffn2{16, true, 8};
    ReLU relu1, relu2, relu3;
    ReduceSum r1{1}, r2{1};
    Flatten flat;
    LayerNorm norm{1}, norm2{1}, norm3{1}, norm4{1}, norm5{1};
    MaxPool2D mp{2, 2}, mp2{2, 2};


    Embedding embedding{16, m_d_model}, embedding_out{16, m_d_model};

    MHA mha_input{m_d_model, /*self_attention*/true, /*num_heads*/1, /*use_bias*/true, /*use_mask*/false, /*use_gpu*/false};
    MHA mha_output{m_d_model, true, 1, true, true, false};
    MHA mha_cross{m_d_model, false, 1, true, false, false};

    UseGPU gpu;



private:

    ValLayer call(const Tensor& input, const bool& training)
    {
        // input is of shape [batch, max_len]
        size_t max_len = input.m_shape[1];

        ValLayer x = {nullptr, &input};
        x = embedding.call(x, training, &gpu);
        // according to tensorflow : "This factor sets the relative scale of the embedding and positonal_encoding."
        Tensor temp = *x.val;
        temp *= std::sqrt(16);
        temp += wef::positional_encoding(max_len, 16);
        x.val = &temp;

        // MHA
        ValLayer c = mha_input.call(x, x, x, training, &gpu);

        // Add and norm
        temp = *c.val + *x.val;
        x.val = &temp;
        x = norm.call(x, training, &gpu);

        // ffn then add and norm
        c = ffn1.call(x, training, &gpu);
        temp = *c.val + *x.val;
        x.val = &temp;
        x = norm2.call(x, training, &gpu);

    


        // output
        ValLayer x_out = {nullptr, &input}; // imagine input is output shifted one to the right
        x_out = embedding_out.call(x_out, training, &gpu);
        temp = *x_out.val;
        temp *= std::sqrt(16);
        temp += wef::positional_encoding(max_len, 16);
        x_out.val = &temp;

        // masked MHA
        ValLayer d = mha_output.call(x_out, x_out, x_out, training, &gpu, x_out); // last x_out should be mask
        
        // Add and norm
        temp = *d.val + *x_out.val;
        x_out.val = &temp;
        x_out = norm3.call(x_out, training, &gpu);

        // cross attn
        ValLayer e = mha_cross.call(x, x, x_out, training, &gpu, x_out); // use another mask instead of last x_out

        // Add and norm
        temp = *e.val + *x_out.val;
        x_out.val = &temp;
        x_out = norm4.call(x_out, training, &gpu);

        // ffn then add and norm
        e = ffn1.call(x_out, training, &gpu);
        temp = *e.val + *x_out.val;
        x_out.val = &temp;
        x_out = norm5.call(x_out, training, &gpu);

        e = out.call(x_out, training, &gpu);

        return e;
    }
    void backward(const Tensor& real, const ValLayer& pred)
    {
        m_loss = wef::categoricalcrossentropy(real, *(pred.val), m_dy);
        ((Layer*)pred.layer)->rev(&m_dy, m_lr, &gpu); // TODO : PLEASE CHECK
    }

    void valid(const Tensor& val_input, const Tensor& val_real)
    {
        // validation
        ValLayer val_pred_ptr = call(val_input, false);
        float val_loss = wef::categoricalcrossentropy(val_real, *(val_pred_ptr.val));
        std::cout << "\tvalid_loss = " << val_loss << "\n";

        float acc = 0;
        for (int i = 0; i < val_real.m_size; i++)
            acc += wef::argmax(wef::softmax(*(val_pred_ptr.val))).m_tensor[i] == val_real.m_tensor[i];
        acc /= val_real.m_size;
        std::cout << "\tval_accuracy = " << acc << std::endl;
        std::cout << "\ttime per epoch = ";
    }

public:
    
    void train(Tensor& input, const Tensor& real, Tensor& val_input, const Tensor& val_real, int epochs, float lr)
    {
        TrainingTimer a;
        m_lr = lr;

        input.reshape((size_t[2]){input.m_shape[0], input.m_shape[1] * input.m_shape[2]}, 2); // remove and add const please
        val_input.reshape((size_t[2]){val_input.m_shape[0], val_input.m_shape[1] * val_input.m_shape[2]}, 2);

        for (int epoch = 0; epoch < epochs; epoch++)
        {
            Timer timer;
            ValLayer pred = call(input, true);
            if (!epoch) m_dy = *(pred.val); // only set m_dy during epoch 0
            backward(real, pred);

            std::cout << "epoch: " << epoch + 1 << "\n\tloss = " << m_loss << "\n";
            valid(val_input, val_real);
        }

    }

private:
    size_t m_d_model = 16;
    float m_lr = 0.0f;
    float m_loss = 0.0f;
    Tensor m_dy;

};


int main()
{


    int n_test = 100;
    int n_train = 1000;

    Tensor train_im = load_mnist_images("mnist/train-images-idx3-ubyte", n_train);
    Tensor train_l = load_mnist_labels("mnist/train-labels-idx1-ubyte", n_train);

    Tensor test_im = load_mnist_images("mnist/t10k-images-idx3-ubyte", n_test);
    Tensor test_l = load_mnist_labels("mnist/t10k-labels-idx1-ubyte", n_test);

    std::cout << "train image shape is: "; train_im.print_shape();
    std::cout << "train label shape is: "; train_l.print_shape();
    
    
    functional_model model;
    model.train(train_im, train_l, test_im, test_l, 3, 0.01f);
  

    return 0;
}
#else
#include "include/layers.h"
#include "include/tensor.h"
#include "include/model.h"
#include "include/mnist.h"

// TODO: mutithread tensor ops and maxpool
// TODO: add gpu support for linux and windows
// TODO: tile dw and dx shaders for conv2d backward pass
// TODO: make tensor class tempelatable
// TODO: add shaders for reduce sum / layer normalization
// TODO: use function pointers in shaders
// TODO: GPU version of transpose (should be simple flat gx=256, gy=1, gz=1)

int main() {

    int n_test = 100;
    int n_train = 1000;

    Tensor train_im = load_mnist_images("mnist/train-images-idx3-ubyte", n_train);
    Tensor train_l = load_mnist_labels("mnist/train-labels-idx1-ubyte", n_train);

    Tensor test_im = load_mnist_images("mnist/t10k-images-idx3-ubyte", n_test);
    Tensor test_l = load_mnist_labels("mnist/t10k-labels-idx1-ubyte", n_test);

    std::cout << "train image shape is: "; train_im.print_shape();
    std::cout << "train label shape is: "; train_l.print_shape();

    float lr = 0.01f;

    int m_units1 = 16;
    int m_units2 = 16;
    int m_units5 = 10;
    
    Conv2D_GPU cov1(3, 3, m_units1, true, 3), cov2(3, 3, m_units2, true, 4), cov3(3, 3, m_units2, true, 5);
    Linear_GPU out(m_units5, true, 7), ffn1(16, true, 8), ffn2(512, true, 8), ffn3(512, true, 8);
    ReLU relu1, relu2, relu3;
    ReduceSum r1(1), r2(1);
    Flatten flat;
    LayerNorm norm(1);
    MaxPool2D_GPU mp(2, 2), mp2(2, 2);

    std::vector<Layer*> network = {&cov1, &relu1, &mp, &cov2, &relu3, &mp2, &cov3, &relu2, &flat, &ffn1, &out};
    Model model(network, true);
    
    /*
    //we can also build a model this way
    Model model;
    model.add(&cov1); model.add(&relu1); model.add(&cov2); model.add(&r1); model.add(&r2); model.add(&layer);
    */
    
    model.fit(train_l, train_im, test_l, test_im, 10, lr);

    Tensor pred = model.predict(test_im);

    std::cout << "\npred: { ";
    for (int i = 0; i < 10; i++) std::cout << wef::argmax(wef::softmax(pred)).m_tensor[i] << " ";
    std::cout << "}\nreal: { ";
    for (int i = 0; i < 10; i++) std::cout << test_l.m_tensor[i] << " ";
    std::cout << "} \n\n";

    model.summary();
    std::cout << "\n";

    return 0;
}

#endif