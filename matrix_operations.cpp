#include <iostream>
#include "matrix_operations.h"

// for simple operations (i.e., no matmul) on 2D or 3D matrices
Tensor matrixOperations::mops(const Tensor& m1, const Tensor& m2, double (*f)(double, double)) 
{
    if (m1.row != m2.row || m1.col != m2.col) {throw std::invalid_argument("matrix size mismatch");}

    // either its 2d or batchs match
    if (!(m2.batch == 1 || m2.batch == m1.batch)) {throw std::invalid_argument("matrix size mismatch");}

    bool bcast = (m2.batch == 1);
    std::vector<int> temp(m1.rank);
    for (int i = 0; i < m1.rank; i ++) temp[i] = m1.shape[i];
    Tensor m = Tensor::create(temp);

    const double* pm1 = m1.tensor.get(); // grab raw pointers for speeeed
    const double* pm2 = m2.tensor.get();
    double* pm = m.tensor.get();

    if (!bcast) 
    {   
        #pragma omp parallel for
        for (size_t i = 0; i < m1.batch * m1.row * m1.col; i++)  pm[i] = f(pm1[i], pm2[i]);

    } else if (bcast) 
        {
            #pragma omp parallel for
            for (size_t i = 0; i < m1.batch * m1.row * m1.col; i++)  pm[i] = f(pm1[i], pm2[i % (m1.row * m1.col)]);
        }
    return m;
}

Tensor matrixOperations::matmul(const Tensor& m1, const Tensor& m2)
{
    if (m1.col != m2.row) {throw std::invalid_argument("matrix size mismatch");}
    const bool bcast = (m2.batch == 1);
    if (!bcast && (m1.batch != m2.batch)) {throw std::invalid_argument("matrix size mismatch");}
    
    std::vector<int> temp(m1.rank);

    // TO DO: CATCH < 1 RANK
    for (int i = 0; i < m1.rank - 1; i ++) temp[i] = m1.shape[i];
    temp[m1.rank - 1] = m2.col;

    Tensor m = Tensor::create(temp);


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

Tensor matrixOperations::cops(const Tensor& m1, const double con, double (*f)(double, double)) 
{
    std::vector<int> temp(m1.rank);
    for (int i = 0; i < m1.rank; i ++) temp[i] = m1.shape[i];
    Tensor m = Tensor::create(temp);

    const double* pm1 = m1.tensor.get();
    double* pm = m.tensor.get();

    #pragma omp parallel for
    for (size_t i = 0; i < m1.batch * m1.row * m1.col; i++) pm[i] = f(pm1[i], con);

    return m;

}

Tensor matrixOperations::transpose(const Tensor& m1)
{
    std::vector<int> temp(m1.rank);
    // TO DO: CATCH < 2 RANK
    for (int i = 0; i < m1.rank - 2; i ++) temp[i] = m1.shape[i];
    temp[m1.rank - 1] = m1.row;
    temp[m1.rank - 2] = m1.col;
    Tensor m = Tensor::create(temp);


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
    std::vector<int> temp(m1.rank);
    for (int i = 0; i < m1.rank; i ++) temp[i] = m1.shape[i];
    Tensor m = Tensor::create(temp);

    const double* pm1 = m1.tensor.get();
    double* pm = m.tensor.get();

    // softmax helper
    double sum = 1e-19;
    for (int ind = 0; ind < m1.tot_size; ind++) sum += std::exp(pm1[ind]);

    #pragma omp parallel for
    for (size_t i = 0; i < m1.batch * m1.row * m1.col; i++) 
    {
        switch (ops) {
            // relu
            case 'a':
                pm[i] = pm1[i] > 0 ?  pm1[i] : 0.0;
                break;
            // derivative relu
            case 'b':
                pm[i] = pm1[i] > 0 ? 1.0 : 0.0;
                break;
            // sigmoid
            case 'c':
                pm[i] = 1 / (1 + std::exp(-1 * pm1[i]));
                break;
            // derivative sigmoid
            case 'd':
                pm[i] = pm1[i] * (1 - pm1[i]);
                break;
            // softmax
            case 'e':
                pm[i] = std::exp(pm1[i]) / sum;
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
    // TO DO: CONFRIM THIS IS ON (just using row and col for batch sum. Usually for weights)
    Tensor m = Tensor::create({m1.row, m1.col}); 
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
        for (size_t i = 0; i < m1.tot_size; i++) 
        {
            
            double diff = pm1[i] - pm2[i];
            loss += diff * diff;
        }
    } else
    {
        #pragma omp parallel for reduction(+:loss)
        for (size_t i = 0; i < m1.tot_size; i++) 
        {
            // if second one need be repeated then just mod
            double diff = pm1[i] - pm2[i % m1size];
            loss += diff * diff;
        }
    }
    return loss / (m1.tot_size);
}

double matrixOperations::binarycrossentropy(const Tensor& m1, const Tensor& m2) // m1 is real and m2 pred !!
{
    // TO DO: catch mismatch tensor

    const double* pm1 = m1.tensor.get(); // grab raw pointers for speeeed
    const double* pm2 = m2.tensor.get();
    double loss = 0.0;
    const double eps = 1e-19;
    

    #pragma omp parallel for reduction(+:loss)
    for (size_t i = 0; i < m1.tot_size; i++)
    {   
        int temp_real = pm1[i] > 0.5;
        
        loss += -(temp_real * std::log(pm2[i] + eps) + (1 - temp_real) * std::log(1 - pm2[i] + eps));
    }

    return loss / m1.batch;
}

double matrixOperations::categoricalcrossentropy(const Tensor& m1, const Tensor& m2, Tensor& m /*m is same as pred*/) // m1 is real and m2 pred !!
{
    // TO DO: catch mismatch tensor

    const double* pm1 = m1.tensor.get(); // grab raw pointers for speeeed
    const double* pm2 = m2.tensor.get();
    double* pm = m.tensor.get();
    double loss = 0.0;
    const double eps = 1e-19;
    


    #pragma omp parallel for reduction(+:loss)
    for (size_t i = 0; i < m1.tot_size; i++) 
    {   
        double sum = 1e-19;
       
        for (size_t j = 0; j < m2.shape[m2.rank - 1]; j++) sum += std::exp(pm2[i * m2.shape[m2.rank - 1] + j]);
        
        for (size_t j = 0; j < m2.shape[m2.rank - 1]; j++)
        {
            double p = std::exp(pm2[i * m2.shape[m2.rank - 1] + j]) / sum;
            if (j == pm1[i])
            {
                loss -= std::log(p + eps);
                pm[i * m.shape[m.rank - 1] + j] = p - 1; // gradient
            }
            else pm[i * m.shape[m.rank - 1] + j] = p;
        }
        
    }
    
    return loss / m1.batch;
}

void matrixOperations::print(const Tensor& m1, std::vector<size_t> v)
{   
    int n = static_cast<int>(v.size());
    
    if (n < m1.rank - 1)
    {   
        
        std::cout << "{ ";
        
        for (int i = 0; i < m1.shape[n]; i++)
        {
            v.push_back(i);
            n == 0 ? std::cout << "\n" : std::cout << "";
            print(m1, v);
            v.pop_back();
            
        }
        n == 0 ? std::cout << "\n" : std::cout << "";
        std::cout << "} ";
        
    }
    else
    {
        std::cout << "{ ";
        for (int i = 0; i < m1.shape[n]; i++) 
        {
            v.push_back(i);
            std::cout << m1.index(v) << " ";
            v.pop_back();
        }
        std::cout << "} ";
    }

}

