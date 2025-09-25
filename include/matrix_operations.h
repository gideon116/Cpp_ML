#pragma once

#include <iostream>
#include <cmath>
#include <initializer_list>
#include <stdexcept>
#include <thread>

#include "tensor.h"
#include "use_GPU.h"

namespace wef {

        // base ops
        Tensor cops(const Tensor& m1, const float con, float (*f)(float, float));
        Tensor matmul(const Tensor& m1, const Tensor& m2);
        Tensor matmul(const Tensor& m1, const Tensor& m2, bool, size_t n_threads=0);
        Tensor transpose(const Tensor& m1);
        Tensor transpose(const Tensor& m1, const size_t* perm);
        Tensor argmax(const Tensor& m1);
        Tensor activation(const Tensor& m1, const char ops);
        Tensor softmax(const Tensor& m1);
        Tensor reducesum(const Tensor& m1, const int ax, const bool keepdims=true);
        float l2(const Tensor& m1, const Tensor& m2);
        float binarycrossentropy(const Tensor& m1, const Tensor& m2);
        float categoricalcrossentropy(const Tensor& m1, const Tensor& m2, Tensor& m, Tensor* mask=nullptr);
        float categoricalcrossentropy(const Tensor& m1, const Tensor& m2, Tensor* mask=nullptr);
        Tensor positional_encoding(const size_t& length, const size_t& depth);
        void print(const Tensor& m1, size_t* arr=nullptr, size_t num=0, bool allc=false);

        // wrappers
        Tensor pow(const Tensor& m1, const float con);
        Tensor relu(const Tensor& m1);
        Tensor d_relu(const Tensor& m1);
        Tensor sigmoid(const Tensor& m1);
        Tensor d_sigmoid(const Tensor& m1);

        // GPU
        Tensor matmul_GPU(const void* gpu, const Tensor& m1, const Tensor& m2);
        Tensor elemwise_GPU(const void* gpu, const Tensor& m1, const Tensor& m2, const int operation/* 0 add, 1 sub, 2 mul, 3 div*/);
        Tensor c_elemwise_GPU(const void* gpu, const Tensor& m1, const float& constant, const int operation/* 0 add, 1 sub, 2 mul, 3 div, 4 pow*/);

};

