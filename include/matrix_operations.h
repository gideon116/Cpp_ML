#ifndef MATRIX_OPERATIONS_H
#define MATRIX_OPERATIONS_H

#include <iostream>
#include <cmath>
#include <initializer_list>
#include <stdexcept>
#include <thread>

#include "tensor.h"
#include "useGPU.h"

namespace wef {

        // base ops
        Tensor cops(const Tensor& m1, const float con, float (*f)(float, float));
        Tensor matmul(const Tensor& m1, const Tensor& m2);
        Tensor matmul(const Tensor& m1, const Tensor& m2, bool, size_t n_threads=0);
        Tensor transpose(const Tensor& m1);
        Tensor argmax(const Tensor& m1);
        Tensor activation(const Tensor& m1, const char ops);
        Tensor softmax(const Tensor& m1);
        Tensor reducesum(const Tensor& m1, const int ax);
        float l2(const Tensor& m1, const Tensor& m2);
        float binarycrossentropy(const Tensor& m1, const Tensor& m2);
        float categoricalcrossentropy(const Tensor& m1, const Tensor& m2, Tensor& m);
        float categoricalcrossentropy(const Tensor& m1, const Tensor& m2);
        void print(const Tensor& m1, size_t* arr=nullptr, size_t num=0, bool allc=false);
        
        // wrappers
        Tensor pow(const Tensor& m1, const float con);
        Tensor relu(const Tensor& m1);
        Tensor d_relu(const Tensor& m1);
        Tensor sigmoid(const Tensor& m1);
        Tensor d_sigmoid(const Tensor& m1);

        // GPU
        Tensor matmul_GPU(const void* gpu, const Tensor& m1, const Tensor& m2);
        Tensor elemwise_GPU(const void* gpu, const Tensor& m1, const Tensor& m2, const int operation/* 0 add, 1 sub, 2 mul*/);

};
#endif
