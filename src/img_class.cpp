#include "../include/example_models.h"
#include "../include/layers.h"
#include "../include/tensor.h"
#include "../include/model.h"
#include "../include/mnist.h"

void img_class() {

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
}