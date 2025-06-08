#include <iostream>
#include "matrix_operations.h"

using matrix2D = std::vector<std::vector<double>>;
using matrixType = std::vector<matrix2D>;

std::array<int,3> matrixOperations::shape(matrixType& m) {
    int num_batch = static_cast<int>((m).size());
    int num_rows = static_cast<int>((m)[0].size());
    int num_cols = static_cast<int>((m)[0][0].size());

    return {num_batch, num_rows, num_cols};
}

matrixType matrixOperations::operation(matrixType& m1, matrixType& m2, std::string op) {
    
    std::array<int,3> m1shape = shape(m1);
    std::array<int,3> m2shape = shape(m2);

    if (m1shape[1] != m2shape[1]) {throw std::invalid_argument("matrix size mismatch");}
    if (m1shape[2] != m2shape[2]) {throw std::invalid_argument("matrix size mismatch");}

    matrixType m = m1;

    for (int b = 0; b < m1shape[0]; b++) {
        for (int r = 0; r < m1shape[1]; r++) {
            for (int c = 0; c <  m1shape[2]; c++) {
                if (op == "subtract") {
                    m[b][r][c] = m1[b][r][c] - (m2shape[0] == 1 ? m2[0][r][c] : m2[b][r][c]);
                }
                else if (op == "add") {
                    m[b][r][c] = m1[b][r][c] + (m2shape[0] == 1 ? m2[0][r][c] : m2[b][r][c]);
                }
                else if (op == "multiply") {
                    m[b][r][c] = m1[b][r][c] * (m2shape[0] == 1 ? m2[0][r][c] : m2[b][r][c]);
                }
                else if (op == "diff") {
                    m[b][r][c] = std::abs(m1[b][r][c] -  (m2shape[0] == 1 ? m2[0][r][c] : m2[b][r][c]));
                }
                
            }
        }
    }
    return m;
}

matrixType matrixOperations::operation(matrixType& m1, matrix2D& m2, std::string op) {
    
    std::array<int,3> m1shape = shape(m1);
    std::array<int,2> m2shape = {static_cast<int>((m2).size()), static_cast<int>((m2)[0].size())};

    if (m1shape[1] != m2shape[0]) {throw std::invalid_argument("matrix size mismatch");}
    if (m1shape[2] != m2shape[1]) {throw std::invalid_argument("matrix size mismatch");}

    matrixType m = m1;

    for (int b = 0; b < m1shape[0]; b++) {
        for (int r = 0; r < m1shape[1]; r++) {
            for (int c = 0; c <  m1shape[2]; c++) {
                if (op == "subtract") {
                    m[b][r][c] = m1[b][r][c] - m2[r][c];
                }
                else if (op == "add") {
                    m[b][r][c] = m1[b][r][c] + m2[r][c];
                }
                else if (op == "multiply") {
                    m[b][r][c] = m1[b][r][c] * m2[r][c];
                }
                else if (op == "diff") {
                    m[b][r][c] = std::abs(m1[b][r][c] -  m2[r][c]);
                }
                
            }
        }
    }
    return m;
}

matrix2D matrixOperations::operation(matrix2D& m1, matrix2D& m2, std::string op) {
    
    std::array<int,3> m1shape = {static_cast<int>((m1).size()), static_cast<int>((m1)[0].size())};
    std::array<int,2> m2shape = {static_cast<int>((m2).size()), static_cast<int>((m2)[0].size())};

    if (m1shape[0] != m2shape[0]) {throw std::invalid_argument("matrix size mismatch");}
    if (m1shape[1] != m2shape[1]) {throw std::invalid_argument("matrix size mismatch");}

    matrix2D m = m1;

    for (int r = 0; r < m1shape[0]; r++) {
        for (int c = 0; c <  m1shape[1]; c++) {
            if (op == "subtract") {
                m[r][c] = m1[r][c] - m2[r][c];
            }
            else if (op == "add") {
                m[r][c] = m1[r][c] + m2[r][c];
            }
            else if (op == "multiply") {
                m[r][c] = m1[r][c] * m2[r][c];
            }
            else if (op == "diff") {
                m[r][c] = std::abs(m1[r][c] -  m2[r][c]);
            }
            
        }
    }
    return m;
}

double matrixOperations::l2(matrixType& m1, matrixType& m2) {

    std::array<int,3> m1shape = shape(m1);
    std::array<int,3> m2shape = shape(m2);


    if (m1shape[1] != m2shape[1]) {throw std::invalid_argument("matrix size mismatch");}
    if (m1shape[2] != m2shape[2]) {throw std::invalid_argument("matrix size mismatch");}

    double mo = 0.0;
    double diff = 0.0;

    for (int b = 0; b < m1shape[0]; b++) {
        for (int r = 0; r < m1shape[1]; r++) {
            for (int c = 0; c <  m1shape[2]; c++) {
                diff = m1[b][r][c] - (m2shape[0] == 1 ? m2[0][r][c] : m2[b][r][c]);
                mo += diff * diff;
            }
        }
    }
    return mo / (m1shape[0] * m1shape[1] * m1shape[2]);
}

matrixType matrixOperations::matmul(matrixType& m1, matrixType& m2){

    std::array<int,3> m1shape = shape(m1);
    std::array<int,3> m2shape = shape(m2);

    int batch = m1shape[0];
    int n = m1shape[2];
    int n1 = m1shape[1];
    int n2 = m2shape[2];

    matrixType m(batch, matrix2D(n1, std::vector<double>(n2, 0)));

    // ensure m1 cols == m2 rows
    if (n != m2shape[1]) {throw std::invalid_argument("matrix size mismatch");}

    for (int b = 0; b < batch; b++){
        for (int i = 0; i < n1; i++) {
            for (int j = 0; j < n; j++) {
                for (int k = 0; k < n2; k++) {
                    m[b][i][k] += (m1[b][i][j] * (m2shape[0] == 1 ? m2[0][j][k] : m2[b][j][k]));
                }
            }
        }
    }
    return m;
}

matrixType matrixOperations::matmul(matrixType& m1, matrix2D& m2){

    std::array<int,3> m1shape = shape(m1);
    std::array<int,2> m2shape = {static_cast<int>((m2).size()), static_cast<int>((m2)[0].size())};

    int batch = m1shape[0];
    int n = m1shape[2];
    int n1 = m1shape[1];
    int n2 = m2shape[1];

    matrixType m(batch, matrix2D(n1, std::vector<double>(n2, 0)));

    // ensure m1 cols == m2 rows
    if (n != m2shape[0]) {throw std::invalid_argument("matrix size mismatch");}

    for (int b = 0; b < batch; b++){
        for (int i = 0; i < n1; i++) {
            for (int j = 0; j < n; j++) {
                for (int k = 0; k < n2; k++) {
                    m[b][i][k] += (m1[b][i][j] * m2[j][k]);
                }
            }
        }
    }
    return m;
}

matrixType matrixOperations::constOperation(double con, matrixType m, std::string operation){
    std::array<int,3> mshape = shape(m);
    for (int b = 0; b < mshape[0]; b++) {
        for (int r = 0; r < mshape[1]; r++) {
            for (int c = 0; c <  mshape[2]; c++) {
                if (operation == "add") { m[b][r][c] += con; }
                else if (operation == "subtract") { m[b][r][c] -= con; }
                else if (operation == "multiply") { m[b][r][c] *= con; }
                else if (operation == "divide") { m[b][r][c] /= con; }
                else if (operation == "power") { m[b][r][c] = std::pow(m[b][r][c], con); }
            }
        }
    }
    return m;
}

matrix2D matrixOperations::constOperation(double con, matrix2D m, std::string operation){
    std::array<int,2> mshape = {static_cast<int>((m).size()), static_cast<int>((m)[0].size())};

    for (int r = 0; r < mshape[0]; r++) {
        for (int c = 0; c <  mshape[1]; c++) {
            if (operation == "add") { m[r][c] += con; }
            else if (operation == "subtract") { m[r][c] -= con; }
            else if (operation == "multiply") { m[r][c] *= con; }
            else if (operation == "divide") { m[r][c] /= con; }
            else if (operation == "power") { m[r][c] = std::pow(m[r][c], con); }
        }
    }
    return m;
}

matrixType matrixOperations::constOperation(std::vector<double>& con, matrixType m, std::string operation){
    std::array<int,3> mshape = shape(m);
    for (int b = 0; b < mshape[0]; b++) {
        for (int r = 0; r < mshape[1]; r++) {
            for (int c = 0; c <  mshape[2]; c++) {
                if (operation == "add") { m[b][r][c] += con[b]; }
                else if (operation == "subtract") { m[b][r][c] -= con[b]; }
                else if (operation == "multiply") { m[b][r][c] *= con[b]; }
                else if (operation == "divide") { m[b][r][c] /= con[b]; }
            }
        }
    }
    return m;
}

std::vector<double> matrixOperations::sum(matrixType& m){

    std::array<int,3> mshape = shape(m);
    std::vector<double> ms(mshape[0], 0);

    for (int b = 0; b < mshape[0]; b++) {
        for (int r = 0; r < mshape[1]; r++) {
            for (int c = 0; c <  mshape[2]; c++) {
                ms[b] += m[b][r][c];
            }
        }
    }
    return ms;
}

matrix2D matrixOperations::sumBatch(matrixType& m){

    std::array<int,3> mshape = shape(m);
    matrix2D ms(mshape[1], std::vector<double>(mshape[2], 0));

    for (int b = 0; b < mshape[0]; b++) {
        for (int r = 0; r < mshape[1]; r++) {
            for (int c = 0; c <  mshape[2]; c++) {
                ms[r][c] += m[b][r][c];
            }
        }
    }
    return ms;
}

matrixType matrixOperations::transpose(matrixType& m){

    std::array<int,3> mshape = shape(m);
    matrixType ms(mshape[0], matrix2D(mshape[2], std::vector<double>(mshape[1])));

    for (int b = 0; b < mshape[0]; b++) {
        for (int r = 0; r < mshape[1]; r++) {
            for (int c = 0; c <  mshape[2]; c++) {
                ms[b][c][r] = m[b][r][c];
            }
        }
    }
    return ms;
}

matrix2D matrixOperations::transpose(matrix2D& m){

    std::array<int,2> mshape = {static_cast<int>((m).size()), static_cast<int>((m)[0].size())};
    
    matrix2D ms(mshape[1], std::vector<double>(mshape[0]));

    for (int r = 0; r < mshape[0]; r++) {
        for (int c = 0; c <  mshape[1]; c++) {
            ms[c][r] = m[r][c];
        }
    }
    
    return ms;
}

matrixType matrixOperations::activations(matrixType& m, std::string op){

    std::array<int,3> mshape = shape(m);
    
    matrixType ms(mshape[0], matrix2D(mshape[1], std::vector<double>(mshape[2])));

    for (int b = 0; b < mshape[0]; b++) {
        for (int r = 0; r < mshape[1]; r++) {
            for (int c = 0; c <  mshape[2]; c++) {

                if (op == "relu") {
                    ms[b][r][c] = m[b][r][c] > 0 ? m[b][r][c] : 0.0;
                }
                if (op == "d_relu") {
                    ms[b][r][c] = m[b][r][c] > 0 ? 1.0 : 0.0;
                }
                
            }
        }
    }
    return ms;
}

void matrixOperations::display(matrixType& m){
    for (matrix2D i : m){
        std::cout << "[ ";
        for (std::vector<double> j : i){
            std::cout << " [ ";
            for (double k : j){
                std::cout << k << " ";
            }
            std::cout << "] ";
        }
        std::cout << "] \n";
        
    }
}