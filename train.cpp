#include "layers.h"
#include "tensor.h"
#include "model.h"
#include "mnist.h"

int main() {

    int n_test = 10;
    int n_train = 1000;

    matrixOperations wf;

    Tensor train_im = load_mnist_images("mnist/train-images-idx3-ubyte", n_train);
    Tensor train_l = load_mnist_labels("mnist/train-labels-idx1-ubyte", n_train);

    Tensor test_im = load_mnist_images("mnist/t10k-images-idx3-ubyte", n_test);
    Tensor test_l = load_mnist_labels("mnist/t10k-labels-idx1-ubyte", n_test);

    std::cout << "train image shape is: "; train_im.printShape();
    std::cout << "train label shape is: "; train_l.printShape();

    double lr = 0.05;

    int units1 = 16;
    int units2 = 16;
    int units5 = 10;
    
    Conv2D cov1(3, 3, units1, 3), cov2(3, 3, units2, 4);
    Linear out(units5, 7), ffn1(32, 8), ffn2(128, 8), ffn3(128, 6), ffn4(32, 8);
    ReLU relu1, relu2;
    ReduceSum r1(1), r2(1);
    Flatten flat;
    LayerNorm norm(1);

    std::vector<Layer*> network = {&cov1, &relu1, &cov2, &relu2, &flat, &ffn1, &out};
    
    Model model(network);
    // model.add(&cov1); model.add(&relu1); model.add(&cov2); model.add(&r1); model.add(&r2); model.add(&layer);
    model.fit(train_l, train_im, 50, lr);
    model.fit(train_l, train_im, 10, lr/10);

    Tensor pred = model.predict(test_im);

    std::cout << "\npred: { ";
    for (int i = 0; i < n_test; i++) std::cout << wf.argmax(wf.softmax(pred)).tensor[i] << " ";
    std::cout << "}\nreal: { ";
    for (int i = 0; i < n_test; i++) std::cout << test_l.tensor[i] << " ";
    std::cout << "} \n\n";

    return 0;
}