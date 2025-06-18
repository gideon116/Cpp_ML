#ifndef MATRIX_OPERATIONS_H
#define MATRIX_OPERATIONS_H

#include <iostream>
#include "tensor.h"

using matrix2D = std::vector<std::vector<double>>;
using matrixType = std::vector<matrix2D>;

class matrixOperations {
    public:

        // base ops
        Tensor mops(const Tensor& m1, const Tensor& m2, const char ops);
        Tensor cops(const Tensor& m1, const double con, const char ops);
        Tensor matmul(const Tensor& m1, const Tensor& m2);
        Tensor transpose(const Tensor& m1);
        Tensor activation(const Tensor& m1, const char ops);
        Tensor batchsum(const Tensor& m1);
        double l2(const Tensor& m1, const Tensor& m2);
        void display(const Tensor& m1);
        
        // wrappers
        Tensor subtract(const Tensor& m1, const Tensor& m2) { return mops(m1, m2, 's'); }
        Tensor add(const Tensor& m1, const Tensor& m2) { return mops(m1, m2, 'a'); }
        Tensor absdiff(const Tensor& m1, const Tensor& m2) { return mops(m1, m2, 'z'); }
        Tensor elemwise(const Tensor& m1, const Tensor& m2) { return mops(m1, m2, 'm'); }

        Tensor constAdd(const Tensor& m1, const double con) { return cops(m1, con, 'a'); }
        Tensor constSub(const Tensor& m1, const double con) { return cops(m1, con, 's'); }
        Tensor constMul(const Tensor& m1, const double con) { return cops(m1, con, 'm'); }
        Tensor constDiv(const Tensor& m1, const double con) { return cops(m1, con, 'd'); }
        Tensor constPower(const Tensor& m1, const double con) { return cops(m1, con, 'p'); }

        Tensor relu(const Tensor& m1) { return activation(m1, 'r'); }
        Tensor d_relu(const Tensor& m1) { return activation(m1, 'd'); }

};
#endif