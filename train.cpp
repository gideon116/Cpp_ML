#include <iostream>
#include "layers.h"
#include "tensor.h"
#include "mnist.h"



int main(int argc, char* argv[]) {

    int n_test = 10;
    int n_train = 10;


    matrixOperations wf;

    Tensor train_im = load_mnist_images("mnist/train-images-idx3-ubyte", 10);
    Tensor train_l = load_mnist_labels("mnist/train-labels-idx1-ubyte", 10);

    Tensor test_im = load_mnist_images("mnist/t10k-images-idx3-ubyte", 10);
    Tensor test_l = load_mnist_labels("mnist/t10k-labels-idx1-ubyte", 10);

    train_im.printShape();
    train_l.printShape();

    double loss;
    double lr = 0.001;

    int units1 = 8;
    int units2 = 8;
    int units3 = 10;
    
    Conv2D cov1(3, 3, units1, 3), cov2(3, 3, units2, 4);
    Linear layer(units3, 5);
    ReLU relu1, relu2;
    ReduceSum r1(1), r2(1);
    std::vector<Layer*> network = {&cov1, &relu1, &cov2, &r1, &r2, &layer};

    for (int epoch = 0; epoch < 10; epoch++) 
    {
        // train
        Tensor y(train_im);
        for (Layer* layer : network) y = (*layer).forward_pass(y, wf);

        // loss calc
        Tensor dy(y);
        loss = wf.categoricalcrossentropy(train_l, y, dy);
        std::cout << "epoch: " << epoch << " loss = " << loss << std::endl;

        // backprop

        for (int i = (int)network.size() - 1; i >= 0; i--) {
            dy = (*network[i]).backward_pass(dy, lr, wf);
        }
    }




    return 0;
}
