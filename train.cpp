#include "layers.h"
#include "tensor.h"
#include "model.h"
#include "mnist.h"

int main() {

    int n_test = 10;
    int n_train = 100;

    matrixOperations wf;

    Tensor train_im = load_mnist_images("mnist/train-images-idx3-ubyte", n_train);
    Tensor train_l = load_mnist_labels("mnist/train-labels-idx1-ubyte", n_train);

    Tensor test_im = load_mnist_images("mnist/t10k-images-idx3-ubyte", n_test);
    Tensor test_l = load_mnist_labels("mnist/t10k-labels-idx1-ubyte", n_test);

    std::cout << "train image shape is: "; train_im.printShape();
    std::cout << "train label shape is: "; train_l.printShape();

    double lr = 0.001;

    int units1 = 8;
    int units2 = 8;
    int units3 = 100;
    int units4 = 20;
    int units5 = 10;
    
    Conv2D cov1(3, 3, units1, 3), cov2(3, 3, units2, 4);
    Linear ffn1(units3, 5), ffn2(units4, 6), out(units5, 7);
    ReLU relu1, relu2;
    ReduceSum r1(1), r2(1);

    std::vector<Layer*> network = {&cov1, &relu1, &ffn1, &cov2, &r1, &r2, &ffn2, &out};
    
    Model model(network);
    // model.add(&cov1); model.add(&relu1); model.add(&cov2); model.add(&r1); model.add(&r2); model.add(&layer);
    model.fit(train_l, train_im, 10, lr);
    
    Tensor pred = model.predict(test_im);
    std::cout << "pred: ";
    wf.print(wf.argmax(wf.softmax(pred)));
    std::cout << "\nreal: ";
    wf.print(test_l);

    return 0;
}
