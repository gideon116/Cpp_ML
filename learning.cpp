#include <iostream>
#include "matrix_operations.h"

// 2d matrix with batch dimenstion
using matrixType = std::vector<std::vector<std::vector<double>>>;

int main() {

    matrixOperations o;

    matrixType input = { // batch, 2, 3
        {{3.5, 3.69, 3.44}, {4.34, 4.42, 2.37}},
        {{3.5, 3.69, 3.43}, {4.34, 4.42, 2.37}},
    };

    matrixType real = { // batch, 2, 1
        {{18}, {3}},
        {{1}, {17}},
    };

    int batch = o.shape(input)[0];
    int rows = o.shape(input)[1];
    int cols = o.shape(input)[2];

    matrixType weight1( // batch, 3, 2
        batch, std::vector<std::vector<double>>(cols, std::vector<double>(rows))
    );

    matrixType weight2( // batch, 2, 1
        batch, std::vector<std::vector<double>>(rows, std::vector<double>(1))
    );

    matrixType pred;
    matrixType dL_dpred;
    matrixType dL_dw1;
    matrixType dlayer1;
    matrixType dL_dw2;
    matrixType layer1;

    std::vector<double> loss(batch, 0);

    for (int i = 0; i < 1; i++) {
        layer1 = o.matmul(input, weight1); // batch, 2, 2
        pred = o.matmul(layer1, weight2); // batch, 2, 1
        
        dL_dpred = o.subtract(pred, real);
        dL_dpred = o.constDiv(batch, dL_dpred);

        dL_dw2 = o.transpose(layer1);
        dL_dw2 = o.matmul(dL_dw2, dL_dpred);

        dlayer1 = o.transpose(weight2); // b, 1, 2
        dlayer1 = o.matmul(dL_dpred, dlayer1); // b, 2, 2

        dL_dw1 = o.transpose(input); // b, 3, 2
        o.matmul(dL_dw1, dlayer1);

    }


    return 0;
}