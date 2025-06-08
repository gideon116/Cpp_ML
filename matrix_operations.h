#include <iostream>

// 2d matrix with batch dimenstion
using matrixType = std::vector<std::vector<std::vector<double>>>;

class matrixOperations {
    public:
        matrixType subtract(matrixType& m1, matrixType& m2);
        matrixType add(matrixType& m1, matrixType& m2);
        matrixType diff(matrixType& m1, matrixType& m2);
        matrixType elemwise(matrixType& m1, matrixType& m2);
        matrixType matmul(matrixType& m1, matrixType& m2);
        std::vector<double> sum(matrixType& m);
        
        matrixType constOperation(double con, matrixType m, std::string operation);
        matrixType constOperation(std::vector<double>& con, matrixType m, std::string operation);

        matrixType constAdd(double con, matrixType& m) { return constOperation(con, m, "add"); }
        matrixType constAdd(std::vector<double>& con, matrixType& m) { return constOperation(con, m, "add"); }

        matrixType constSub(double con, matrixType& m) { return constOperation(con, m, "subtract"); }
        matrixType constSub(std::vector<double>& con, matrixType& m) { return constOperation(con, m, "subtract"); }

        matrixType constMul(double con, matrixType& m) { return constOperation(con, m, "multiply"); }
        matrixType constMul(std::vector<double>& con, matrixType& m) { return constOperation(con, m, "multiply"); }

        matrixType constDiv(double con, matrixType& m) { return constOperation(con, m, "divide"); }
        matrixType constDiv(std::vector<double>& con, matrixType& m) { return constOperation(con, m, "divide"); }

        matrixType constPower(double con, matrixType& m) { return constOperation(con, m, "power"); }

        matrixType transpose(matrixType& m);
        
        void display(matrixType& m1);
        std::array<int,3> shape(matrixType& m);
};
