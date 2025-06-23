#include <iostream>
#include "matrix_operations.h"

// for simple operations (i.e., no matmul) on 2D or 3D matrices
Tensor matrixOperations::mops(const Tensor& m1, const Tensor& m2, const char ops) 
{
    if (m1.row != m2.row || m1.col != m2.col) {throw std::invalid_argument("matrix size mismatch");}

    // either its 2d or batchs match
    if (!(m2.batch == 1 || m2.batch == m1.batch)) {throw std::invalid_argument("matrix size mismatch");}

    bool bcast = (m2.batch == 1);
    Tensor m(std::vector<int>({m1.batch, m1.row, m1.col}));
    m.rank = m1.rank;

    const double* pm1 = m1.tensor.get(); // grab raw pointers for speeeed
    const double* pm2 = m2.tensor.get();
    double* pm = m.tensor.get();

    if (!bcast) 
    {   
        #pragma omp parallel for
        for (size_t i = 0; i < m1.batch * m1.row * m1.col; i++) 
        {
            switch (ops) {
                case 's':
                    pm[i] = pm1[i] - pm2[i];
                    break;
                case 'a':
                    pm[i] = pm1[i] + pm2[i];
                    break;
                case 'm':
                    pm[i] = pm1[i] * pm2[i];
                    break;
                case 'z':
                    pm[i] = std::abs(pm1[i] - pm2[i]);
                    break;
                default:
                    std::cout << "ERROR, SOMETHING WENT WRONG; THATS ALL I KNOW" << std::endl;
                    break;
                }
            }
        } else if (bcast) 
        {   
            #pragma omp parallel for
            for (size_t i = 0; i < m1.batch * m1.row * m1.col; i++) {
                switch (ops) {
                    case 's':
                        pm[i] = pm1[i] - pm2[i % (m1.row * m1.col)];
                        break;
                    case 'a':
                        pm[i] = pm1[i] + pm2[i % (m1.row * m1.col)];
                        break;
                    case 'm':
                        pm[i] = pm1[i] * pm2[i % (m1.row * m1.col)];
                        break;
                    case 'z':
                        pm[i] = std::abs(pm1[i] - pm2[i % (m1.row * m1.col)]);
                        break;
                    default:
                        std::cout << "ERROR, SOMETHING WENT WRONG; THATS ALL I KNOW" << std::endl;
                        break;
                    }
                }
            }
    return m;
}

Tensor matrixOperations::matmul(const Tensor& m1, const Tensor& m2)
{
    if (m1.col != m2.row) {throw std::invalid_argument("matrix size mismatch");}
    const bool bcast = (m2.batch == 1);
    if (!bcast && (m1.batch != m2.batch)) {throw std::invalid_argument("matrix size mismatch");}

    Tensor m(std::vector<int>({m1.batch, m1.row, m2.col}));
    m.rank = m1.rank;

    const double* pm1 = m1.tensor.get(); // grab raw pointers for speeeed
    const double* pm2 = m2.tensor.get();
    double* pm = m.tensor.get();

    const size_t m1size = m1.row * m1.col;
    const size_t m2size = m2.row * m2.col;
    const size_t msize = m1.row * m2.col;
    
    #pragma omp parallel for
    for (size_t i = 0; i < m1.batch * msize; i++) {pm[i]=0.0;}
    
    #pragma omp parallel for collapse(3) schedule(static)
    for (int b = 0; b < m1.batch; b++){
        
        const double* pm1temp = pm1 + b * m1size; // shift pm1 by one batch worth
        const double* pm2temp = !bcast ? pm2 + b * m2size : pm2; // only shift if m2 is 3D
        double* pmtemp = pm + b * msize;

        for (size_t i = 0; i < m1.row; i++) {
            for (size_t k = 0; k < m2.col; k++) {

                double sum = 0;
                for (size_t j = 0; j < m1.col; j++) {
                    sum += pm1temp[i * m1.col + j] * pm2temp[j * m2.col + k];
                }
                pmtemp[i * m2.col + k] = sum;
            }
        }
    }
    return m;
}

Tensor matrixOperations::cops(const Tensor& m1, const double con, const char ops) 
{
    Tensor m(std::vector<int>({m1.batch, m1.row, m1.col}));
    m.rank = m1.rank;

    const double* pm1 = m1.tensor.get();
    double* pm = m.tensor.get();

    #pragma omp parallel for
    for (size_t i = 0; i < m1.batch * m1.row * m1.col; i++) 
    {
        switch (ops) {
            case 's':
                pm[i] = pm1[i] - con;
                break;
            case 'a':
                pm[i] = pm1[i] + con;
                break;
            case 'm':
                pm[i] = pm1[i] * con;
                break;
            case 'd':
                pm[i] = pm1[i] / con;
                break;
            case 'p':
                pm[i] = std::pow(pm1[i], con);
                break;
            default:
                std::cout << "ERROR, SOMETHING WENT WRONG; THATS ALL I KNOW" << std::endl;
                break;
            }
        }
    return m;

}

Tensor matrixOperations::transpose(const Tensor& m1)
{
    Tensor m(std::vector<int>({m1.batch, m1.col, m1.row}));
    m.rank = m1.rank;

    const double* pm1 = m1.tensor.get();
    double* pm = m.tensor.get();

    const size_t msize = m1.row * m1.col;

    #pragma omp parallel for collapse(3) schedule(static)
    for (int b = 0; b < m1.batch; b++){

        const double* pm1temp = pm1 + b * msize;
        double* pmtemp = pm + b * msize;

        for (size_t i = 0; i < m1.row; i++) {
            for (size_t j = 0; j < m1.col; j++) {
                pmtemp[j * m1.row + i] = pm1temp[i * m1.col + j];
            }
        }
    }
    return m;
}

Tensor matrixOperations::activation(const Tensor& m1, const char ops)
{
    Tensor m(std::vector<int>({m1.batch, m1.row, m1.col}));
    m.rank = m1.rank;

    const double* pm1 = m1.tensor.get();
    double* pm = m.tensor.get();

    #pragma omp parallel for
    for (size_t i = 0; i < m1.batch * m1.row * m1.col; i++) 
    {
        switch (ops) {
            case 'r':
                pm[i] = pm1[i] > 0 ?  pm1[i] : 0.0;
                break;
            case 'd':
                pm[i] = pm1[i] > 0 ? 1.0 : 0.0;
                break;
            default:
                std::cout << "ERROR, SOMETHING WENT WRONG; THATS ALL I KNOW" << std::endl;
                break;
            }
        }
    return m;

}

Tensor matrixOperations::batchsum(const Tensor& m1)
{   

    Tensor m(std::vector<int>({m1.row, m1.col})); 
    const double* pm1 = m1.tensor.get();
    double* pm = m.tensor.get();

    const size_t m1size = m1.row * m1.col;
    for (size_t i = 0; i < m1size; i++) {pm[i]=0.0;}

    #pragma omp parallel for collapse(3) schedule(static)
    for (int b = 0; b < m1.batch; b++){
        
        const double* pm1temp = pm1 + b * m1size;
        
        for (size_t i = 0; i < m1.row; i++) {
            for (size_t j = 0; j < m1.col; j++) {
                #pragma omp atomic
                pm[i * m1.col + j] += pm1temp[i * m1.col + j];
            }
        }
    }
    return m;
}

double matrixOperations::l2(const Tensor& m1, const Tensor& m2)
{
    if (m1.row != m2.row || m1.col != m2.col) {throw std::invalid_argument("matrix size mismatch");}

    // either its 2d or batchs match
    if (!(m2.batch == 1 || m2.batch == m1.batch)) {throw std::invalid_argument("matrix size mismatch");}

    bool bcast = (m2.batch == 1);
    const size_t m1size = m1.row * m1.col;

    const double* pm1 = m1.tensor.get(); // grab raw pointers for speeeed
    const double* pm2 = m2.tensor.get();
    double loss = 0.0;
    

    if (!bcast) 
    {
        #pragma omp parallel for reduction(+:loss)
        for (size_t i = 0; i < m1.batch * m1size; i++) 
        {
            
            double diff = pm1[i] - pm2[i];
            loss += diff * diff;
        }
    } else
    {
        #pragma omp parallel for reduction(+:loss)
        for (size_t i = 0; i < m1.batch * m1size; i++) 
        {
            
            double diff = pm1[i] - pm2[i % m1size];
            loss += diff * diff;
        }
    }
    return loss / (m1.batch * m1size);
}

void matrixOperations::display(const Tensor& m1){


    //[ [ [1], [2], [3], [4] ], [ [5], [6], [7], [8] ] ] shape = 2, 4, 1
    // [1], 2, 3, 4, 5, 6, 7, 
    // [[1] [2] [3] [4]] [5] [6] [7] [8] skip 12 (4 * 1 + 2) c, skip 1 c
    //  skip 28 (2 * 12 + 2) c, skip 1 c
    // (2 * (4 * 1 + 2) + 2)

    /*
    std::string s = "";
    for (size_t i = 0; i < m1.batch * m1.row * m1.col; i++) s += m1.tensor[i];

    for (int i = m1.rank - 1; i > -1; i--)
    {
        for (int j = 0; j < m1.shape[i]; j ++)
        {
            m1.tensor[1];
        }
    }
    */



    for (size_t i = 0; i < m1.batch; i++) {
        std::cout << "[ ";
        for (size_t j = 0; j < m1.row; j++) {
            std::cout << " [ ";
            for (size_t k = 0; k < m1.col; k++) {
                std::cout << m1.index({i, j, k}) << " ";
            }
            std::cout << "] ";
        }
        std::cout << "] \n";
        
    }
}
