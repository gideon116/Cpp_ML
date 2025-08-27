#if 1
#include "include/layers.h"
#include "include/tensor.h"
#include "include/model.h"
#include "include/mnist.h"

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
    Linear out{m_units5, true, 7}, ffn1{16, true, 8};
    ReLU relu1, relu2, relu3;
    ReduceSum r1{1}, r2{1};
    Flatten flat;
    LayerNorm norm{1};
    MaxPool2D mp{2, 2}, mp2{2, 2};

    MHA mha{16, true, 1, true, false, false};

    UseGPU gpu;

    ValLayer m_xlr;

private:

    Tensor* call(Tensor& input, const bool& training)
    {
        Tensor* x = &input;
        x->reshape((size_t[3]){x->m_shape[0], x->m_shape[1], x->m_shape[2]}, 3);
        Tensor aa[4] = {*x, *x, *x,  Tensor()};
        m_xlr = {nullptr, aa};
        ValLayer* xl = &m_xlr;
    
        xl = mha.call(xl, training, &gpu);
        xl = flat.call(xl, training, &gpu);
        // xl = ffn1.call(xl, training, &gpu);
        xl = out.call(xl, training, &gpu);

        return xl->val;
    }
    void backward(const Tensor& real)
    {
        m_loss = wef::categoricalcrossentropy(real, *m_x, m_dy);
     
        m_y = &m_dy;
        ((Layer*)(m_xlr.layer))->rev(m_y, m_lr, &gpu);
        
    }

    void valid(Tensor& val_input, Tensor& val_real)
    {
        // validation
        Tensor* val_pred_ptr = call(val_input, false);
        float val_loss = wef::categoricalcrossentropy(val_real, *val_pred_ptr);
        std::cout << "\tvalid_loss = " << val_loss << "\n";

        float acc = 0;
        for (int i = 0; i < val_real.m_size; i++)
            acc += wef::argmax(wef::softmax(*val_pred_ptr)).m_tensor[i] == val_real.m_tensor[i];
        acc /= val_real.m_size;
        std::cout << "\tval_accuracy = " << acc << std::endl;
        std::cout << "\ttime per epoch = ";

    }

public:
    
    void train(Tensor& input, Tensor& real, Tensor& val_input, Tensor& val_real, int epochs, float lr)
    {
        TrainingTimer a;
        m_lr = lr;

        for (int epoch = 0; epoch < epochs; epoch++)
        {
            Timer timer;
            m_x = call(input, true);
            if (!epoch) m_dy = *m_x; // only set m_dy during epoch 0
            backward(real);

            std::cout << "epoch: " << epoch + 1 << "\n\tloss = " << m_loss << "\n";
            valid(val_input, val_real);
        }

    }

private:
    Tensor* m_x;
    float m_lr = 0.0f;
    float m_loss = 0.0f;
    Tensor m_dy;
    Tensor* m_y = nullptr;

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