#ifndef MATRIX_OPERATIONS_H
#define MATRIX_OPERATIONS_H

#include <iostream>

using matrix2D = std::vector<std::vector<double>>;
using matrixType = std::vector<matrix2D>;

class matrixOperations {
    public:
        matrixType operation(const matrixType& m1, const matrixType& m2, std::string op);
        matrixType operation(const matrixType& m1, const matrix2D& m2, std::string op);
        matrix2D operation(const matrix2D& m1, const matrix2D& m2, std::string op);

        matrixType subtract(const matrixType& m1, const matrixType& m2) { return operation(m1, m2, "subtract"); }
        matrixType subtract(const matrixType& m1, const matrix2D& m2) { return operation(m1, m2, "subtract"); }
        matrix2D subtract(const matrix2D& m1, const matrix2D& m2) { return operation(m1, m2, "subtract"); }

        matrixType add(const matrixType& m1, const matrixType& m2) { return operation(m1, m2, "add"); }
        matrixType add(const matrixType& m1, const matrix2D& m2) { return operation(m1, m2, "add"); }
        matrix2D add(const matrix2D& m1, const matrix2D& m2) { return operation(m1, m2, "add"); }

        matrixType diff(const matrixType& m1, const matrixType& m2) { return operation(m1, m2, "diff"); }
        matrixType diff(const matrixType& m1, const matrix2D& m2) { return operation(m1, m2, "diff"); }
        matrix2D diff(const matrix2D& m1, const matrix2D& m2) { return operation(m1, m2, "diff"); }

        matrixType elemwise(const matrixType& m1, const matrixType& m2) { return operation(m1, m2, "multiply"); }
        matrixType elemwise(const matrixType& m1, const matrix2D& m2) { return operation(m1, m2, "multiply"); }
        matrix2D elemwise(const matrix2D& m1, const matrix2D& m2) { return operation(m1, m2, "multiply"); }

        matrixType matmul(const matrixType& m1, const matrixType& m2);
        matrixType matmul(const matrixType& m1, const matrix2D& m2);
        
        matrixType constOperation(const double con, const matrixType& m, const std::string operation);
        matrix2D constOperation(const double con, const matrix2D& m, const std::string operation);
        matrixType constOperation(const std::vector<double>& con, const matrixType& m, const std::string operation);

        matrixType constAdd(const double con, const matrixType& m) { return constOperation(con, m, "add"); }
        matrix2D constAdd(const double con, const matrix2D& m) { return constOperation(con, m, "add"); }
        matrixType constAdd(const std::vector<double>& con, const matrixType& m) { return constOperation(con, m, "add"); }

        matrixType constSub(const double con, const matrixType& m) { return constOperation(con, m, "subtract"); }
        matrix2D constSub(const double con, const matrix2D& m) { return constOperation(con, m, "subtract"); }
        matrixType constSub(const std::vector<double>& con, const matrixType& m) { return constOperation(con, m, "subtract"); }

        matrixType constMul(const double con, const matrixType& m) { return constOperation(con, m, "multiply"); }
        matrix2D constMul(const double con, const matrix2D& m) { return constOperation(con, m, "multiply"); }
        matrixType constMul(const std::vector<double>& con, const matrixType& m) { return constOperation(con, m, "multiply"); }

        matrixType constDiv(const double con, const matrixType& m) { return constOperation(con, m, "divide"); }
        matrix2D constDiv(const double con,const matrix2D& m) { return constOperation(con, m, "divide"); }
        matrixType constDiv(const std::vector<double>& con, const matrixType& m) { return constOperation(con, m, "divide"); }

        matrixType constPower(const double con, const matrixType& m) { return constOperation(con, m, "power"); }
        matrix2D constPower(const double con, const matrix2D& m) { return constOperation(con, m, "power"); }

        matrixType transpose(const matrixType& m);
        matrix2D transpose(const matrix2D& m);

        matrixType activations(const matrixType& m, std::string op);
        matrixType relu(const matrixType& m) { return activations(m, "relu"); }
        matrixType d_relu(const matrixType& m) { return activations(m, "d_relu"); }
        
        void display(const matrixType& m1);
        double l2(const matrixType& m1, const matrixType& m2);
        std::vector<double> sum(const matrixType& m);
        matrix2D sumBatch(const matrixType& m);
        std::array<int,3> shape(const matrixType& m);
        
};
#endif