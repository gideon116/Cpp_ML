#ifndef MATRIX_OPERATIONS_H
#define MATRIX_OPERATIONS_H

#include <iostream>
#include <cmath>
#include "tensor.h"

class matrixOperations {
    public:
    
        // base ops
        Tensor mops(const Tensor& m1, const Tensor& m2, double (*f)(double, double)); // third param is a fn pointer
        Tensor cops(const Tensor& m1, const double con, double (*f)(double, double));
        Tensor matmul(const Tensor& m1, const Tensor& m2);
        Tensor transpose(const Tensor& m1);
        Tensor activation(const Tensor& m1, const char ops);
        Tensor batchsum(const Tensor& m1);
        double l2(const Tensor& m1, const Tensor& m2);
        double binarycrossentropy(const Tensor& m1, const Tensor& m2);
        double categoricalcrossentropy(const Tensor& m1, const Tensor& m2, Tensor& m);
        void print(const Tensor& m1, std::vector<size_t> v={});
        
        // wrappers
        Tensor subtract(const Tensor& m1, const Tensor& m2) { return mops(m1, m2, [](double a, double b) { return a - b; }); }
        Tensor add(const Tensor& m1, const Tensor& m2) { return mops(m1, m2, [](double a, double b) { return a + b; }); }
        Tensor absdiff(const Tensor& m1, const Tensor& m2) { return mops(m1, m2, [](double a, double b) { return std::abs(a - b); }); }
        Tensor elemwise(const Tensor& m1, const Tensor& m2) { return mops(m1, m2, [](double a, double b) { return a * b; }); }

        Tensor constAdd(const Tensor& m1, const double con) { return cops(m1, con, [](double a, double b){ return a + b; }); }
        Tensor constSub(const Tensor& m1, const double con) { return cops(m1, con, [](double a, double b){ return a - b; }); }
        Tensor constMul(const Tensor& m1, const double con) { return cops(m1, con, [](double a, double b){ return a * b; }); }
        Tensor constDiv(const Tensor& m1, const double con) { return cops(m1, con, [](double a, double b){ return a / b; }); }
        Tensor constPower(const Tensor& m1, const double con) { return cops(m1, con, [](double a, double b){ return std::pow(a, b); }); }

        Tensor relu(const Tensor& m1) { return activation(m1, 'a'); }
        Tensor d_relu(const Tensor& m1) { return activation(m1, 'b'); }
        Tensor sigmoid(const Tensor& m1) { return activation(m1, 'c'); }
        Tensor d_sigmoid(const Tensor& m1) { return activation(m1, 'd'); }

        Tensor tensor_from_shape(std::initializer_list<int> shape) { return Tensor::create(shape); }
        
        Tensor make_tensor(const std::initializer_list<Tensor>& vs) { return Tensor(vs); }

};
#endif
