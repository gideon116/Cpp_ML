#ifndef MATRIX_OPERATIONS_H
#define MATRIX_OPERATIONS_H

#include <iostream>
#include <cmath>
#include <initializer_list>
#include <stdexcept>
#include <thread>

#include "tensor.h"

namespace wef {

        // base ops
        Tensor cops(const Tensor& m1, const double con, double (*f)(double, double));
        Tensor matmul(const Tensor& m1, const Tensor& m2);
        Tensor matmul(const Tensor& m1, const Tensor& m2, bool, int n_threads=0);
        Tensor transpose(const Tensor& m1);
        Tensor argmax(const Tensor& m1);
        Tensor activation(const Tensor& m1, const char ops);
        Tensor softmax(const Tensor& m1);
        Tensor reducesum(const Tensor& m1, const int ax);
        double l2(const Tensor& m1, const Tensor& m2);
        double binarycrossentropy(const Tensor& m1, const Tensor& m2);
        double categoricalcrossentropy(const Tensor& m1, const Tensor& m2, Tensor& m);
        double categoricalcrossentropy(const Tensor& m1, const Tensor& m2);
        void print(const Tensor& m1, size_t arr[]=nullptr, size_t num=0);
        
        // wrappers
        Tensor pow(const Tensor& m1, const double con);
        Tensor relu(const Tensor& m1);
        Tensor d_relu(const Tensor& m1);
        Tensor sigmoid(const Tensor& m1);
        Tensor d_sigmoid(const Tensor& m1);

};
#endif
