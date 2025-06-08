#include <iostream>

using matrix2D = std::vector<std::vector<double>>;
using matrixType = std::vector<matrix2D>;

class matrixOperations {
    public:
        matrixType operation(matrixType& m1, matrixType& m2, std::string op);
        matrixType operation(matrixType& m1, matrix2D& m2, std::string op);
        matrix2D operation(matrix2D& m1, matrix2D& m2, std::string op);

        matrixType subtract(matrixType& m1, matrixType& m2) { return operation(m1, m2, "subtract"); }
        matrixType subtract(matrixType& m1, matrix2D& m2) { return operation(m1, m2, "subtract"); }
        matrix2D subtract(matrix2D& m1, matrix2D& m2) { return operation(m1, m2, "subtract"); }

        matrixType add(matrixType& m1, matrixType& m2) { return operation(m1, m2, "add"); }
        matrixType add(matrixType& m1, matrix2D& m2) { return operation(m1, m2, "add"); }
        matrix2D add(matrix2D& m1, matrix2D& m2) { return operation(m1, m2, "add"); }

        matrixType diff(matrixType& m1, matrixType& m2) { return operation(m1, m2, "diff"); }
        matrixType diff(matrixType& m1, matrix2D& m2) { return operation(m1, m2, "diff"); }
        matrix2D diff(matrix2D& m1, matrix2D& m2) { return operation(m1, m2, "diff"); }

        matrixType elemwise(matrixType& m1, matrixType& m2) { return operation(m1, m2, "multiply"); }
        matrixType elemwise(matrixType& m1, matrix2D& m2) { return operation(m1, m2, "multiply"); }
        matrix2D elemwise(matrix2D& m1, matrix2D& m2) { return operation(m1, m2, "multiply"); }

        matrixType matmul(matrixType& m1, matrixType& m2);
        matrixType matmul(matrixType& m1, matrix2D& m2);
        
        matrixType constOperation(double con, matrixType m, std::string operation);
        matrix2D constOperation(double con, matrix2D m, std::string operation);
        matrixType constOperation(std::vector<double>& con, matrixType m, std::string operation);

        matrixType constAdd(double con, matrixType& m) { return constOperation(con, m, "add"); }
        matrix2D constAdd(double con, matrix2D& m) { return constOperation(con, m, "add"); }
        matrixType constAdd(std::vector<double>& con, matrixType& m) { return constOperation(con, m, "add"); }

        matrixType constSub(double con, matrixType& m) { return constOperation(con, m, "subtract"); }
        matrix2D constSub(double con, matrix2D& m) { return constOperation(con, m, "subtract"); }
        matrixType constSub(std::vector<double>& con, matrixType& m) { return constOperation(con, m, "subtract"); }

        matrixType constMul(double con, matrixType& m) { return constOperation(con, m, "multiply"); }
        matrix2D constMul(double con, matrix2D& m) { return constOperation(con, m, "multiply"); }
        matrixType constMul(std::vector<double>& con, matrixType& m) { return constOperation(con, m, "multiply"); }

        matrixType constDiv(double con, matrixType& m) { return constOperation(con, m, "divide"); }
        matrix2D constDiv(double con, matrix2D& m) { return constOperation(con, m, "divide"); }
        matrixType constDiv(std::vector<double>& con, matrixType& m) { return constOperation(con, m, "divide"); }

        matrixType constPower(double con, matrixType& m) { return constOperation(con, m, "power"); }
        matrix2D constPower(double con, matrix2D& m) { return constOperation(con, m, "power"); }

        matrixType transpose(matrixType& m);
        matrix2D transpose(matrix2D& m);

        matrixType activations(matrixType& m, std::string op);
        matrixType relu(matrixType& m) { return activations(m, "relu"); }
        matrixType d_relu(matrixType& m) { return activations(m, "drelu"); }
        
        void display(matrixType& m1);
        double l2(matrixType& m1, matrixType& m2);
        std::vector<double> sum(matrixType& m);
        matrix2D sumBatch(matrixType& m);
        std::array<int,3> shape(matrixType& m);
        
};
