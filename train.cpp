
#if 0

#include <iostream>
#include "layers.h"
#include "tensor.h"
#include "model.h"
#include "mnist.h"

int main()
{

    matrixOperations wef;

    Tensor a = Tensor::create({2, 3, 2, 3});
    for (int i = 0; i < a.tot_size; i++) a.tensor.get()[i] = i+1;
    

    // follwoing Ba et al. 2016
    int ax = 1;
    Tensor mu = wef::reducesum(a, /*axis=*/ax) / a.shape[ax];
    Tensor x_mu = a - mu;
    Tensor var = wef::reducesum(x_mu * x_mu, /*axis=*/ax) / a.shape[ax];
    double eps = 0.01;
    std::unique_ptr<int[]> beta_shape = std::make_unique<int[]>(a.rank);
    std::unique_ptr<int[]> gamma_shape = std::make_unique<int[]>(a.rank);
    for (int i = 0; i < a.rank; i++) { beta_shape[i] = 1; gamma_shape[i] = 1; }
    beta_shape[ax] = a.shape[ax]; gamma_shape[ax] = a.shape[ax];
    Tensor beta = Tensor::create(beta_shape.get(), a.rank);
    Tensor gamma = Tensor::create(gamma_shape.get(), a.rank);

    Tensor inv_std = wef::constPower(var + eps, -0.5);
    Tensor x_i_hat = x_mu * inv_std;
    Tensor y_i = x_i_hat * gamma + beta;

    // y_i.printShape();
    // wef::print(y_i);

    Tensor X = Tensor::create({2, 3, 2, 3});
    for (int i = 0; i < X.tot_size; i++) X.tensor.get()[i] = i;
    Tensor dy = Tensor::create({2, 3, 2, 3});
    for (int i = 0; i < dy.tot_size; i++) dy.tensor.get()[i] = i;

        x_mu = X - mu;
    Tensor d_gamma = dy * x_mu * inv_std;
    
    for (int i = 0; i < 4; i++) 
    {
        if (i != 1)
        {
            d_gamma = wef::reducesum(d_gamma, i);
        }
    }

    // d_gamma = wef::batchsum(d_gamma);
    Tensor d_beta = wef::batchsum(dy);

    Tensor d_x_hat = gamma * dy;
    Tensor sum_dx  = wef::reducesum(d_x_hat, ax);
    Tensor sum_dxh = wef::reducesum(d_x_hat * x_mu, ax);

    Tensor term1 = d_x_hat * a.shape[ax] - sum_dx;
    Tensor term2 = x_mu * inv_std * sum_dxh;
    Tensor dx = inv_std * (1.0 / a.shape[ax]) * (term1 - term2);
    d_gamma.printShape();
    gamma.printShape();
    gamma = gamma - d_gamma * 0.001 / dy.shape[0];
    beta  = beta  - d_beta * 0.001 / dy.shape[0];

    return 0;
}

#endif



#if 1
#include "layers.h"
#include "tensor.h"
#include "model.h"
#include "mnist.h"

int main() {

    int n_test = 10;
    int n_train = 1000;


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
    for (int i = 0; i < n_test; i++) std::cout << wef::argmax(wef::softmax(pred)).tensor[i] << " ";
    std::cout << "}\nreal: { ";
    for (int i = 0; i < n_test; i++) std::cout << test_l.tensor[i] << " ";
    std::cout << "} \n\n";

    return 0;
}
#endif