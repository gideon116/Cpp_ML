
#if 1
#include "layers.h"
#include "tensor.h"
#include "model.h"
#include "mnist.h"

int main() {

    int n_test = 10;
    int n_train = 10;


    Tensor train_im = load_mnist_images("mnist/train-images-idx3-ubyte", n_train);
    Tensor train_l = load_mnist_labels("mnist/train-labels-idx1-ubyte", n_train);

    Tensor test_im = load_mnist_images("mnist/t10k-images-idx3-ubyte", n_test);
    Tensor test_l = load_mnist_labels("mnist/t10k-labels-idx1-ubyte", n_test);

    std::cout << "train image shape is: "; train_im.printShape();
    std::cout << "train label shape is: "; train_l.printShape();

    double lr = 0.01;

    int units1 = 32;
    int units2 = 16;
    int units5 = 10;
    
    Conv2D_Fast cov1(3, 3, units1, false, 3), cov2(3, 3, units2, false, 4), cov3(3, 3, units2, false, 5);
    Linear_Fast out(units5, false, 7), ffn1(16, false, 8), ffn2(128, false, 8), ffn3(64, false, 6), ffn4(32, false, 8);
    ReLU relu1, relu2, relu3;
    ReduceSum r1(1), r2(1);
    Flatten flat;
    LayerNorm norm(1);
    MaxPool2D mp(2, 2), mp2(2, 2);

    std::vector<Layer*> network = {&cov1, &relu1, &mp, &cov2, &relu2, &flat, &ffn1, &out};
    // std::vector<Layer*> network = {&flat, &ffn2, &relu1, &ffn3, &relu2, &ffn1, &out};
    
    Model model(network);
    // model.add(&cov1); model.add(&relu1); model.add(&cov2); model.add(&r1); model.add(&r2); model.add(&layer);
    model.fit(train_l, train_im, 10, lr);
    // model.fit(train_l, train_im, 10, lr/10);

    Tensor pred = model.predict(test_im);

    std::cout << "\npred: { ";
    for (int i = 0; i < n_test; i++) std::cout << wef::argmax(wef::softmax(pred)).tensor[i] << " ";
    std::cout << "}\nreal: { ";
    for (int i = 0; i < n_test; i++) std::cout << test_l.tensor[i] << " ";
    std::cout << "} \n\n";

    model.summary();
    std::cout << "\n";


    return 0;
}
#endif