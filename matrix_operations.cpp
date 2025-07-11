#include <iostream>
#include "matrix_operations.h"

// for simple operations (i.e., no matmul) on 2D or 3D matrices
Tensor matrixOperations::mops(const Tensor& m1, const Tensor& m2, double (*f)(double, double)) 
{
    if (m1.row != m2.row || m1.col != m2.col) {throw std::invalid_argument("matrix size mismatch");}

    // either its 2d or batchs match
    if (!(m2.batch == 1 || m2.batch == m1.batch)) {throw std::invalid_argument("matrix size mismatch");}

    bool bcast = (m2.batch == 1);
    Tensor m = Tensor::create(m1.shape.get(), m1.rank);

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

Tensor matrixOperations::matmul(const Tensor& m1, const Tensor& m2, bool, int n_threads)
{
    if (m1.col != m2.row) {throw std::invalid_argument("matrix size mismatch");}
    const bool bcast = (m2.batch == 1);
    if (!bcast && (m1.batch != m2.batch)) {throw std::invalid_argument("matrix size mismatch");}
    
    std::vector<int> temp(m1.rank);

    // TO DO: CATCH < 1 RANK
    for (int i = 0; i < m1.rank - 1; i ++) temp[i] = m1.shape[i];
    temp[m1.rank - 1] = m2.col;

    Tensor m = Tensor::create(temp);

    const double* pm1 = m1.tensor.get();
    const double* pm2 = m2.tensor.get();
    double* pm = m.tensor.get();

    const size_t m1size = m1.row * m1.col;
    const size_t m2size = m2.row * m2.col;
    const size_t msize = m1.row * m2.col;

    // multi thread additions

    if (n_threads == 0)
    {
        int avaliable_threads = std::thread::hardware_concurrency(); // may be 0
        n_threads = std::min<int>( m1.row,  avaliable_threads > 0 ? avaliable_threads : 1 );
    }

    const int stride = m1.row / n_threads;
    const int rem = m1.row % n_threads;

    // spin up
    std::thread* threads = new std::thread[n_threads];

    for (int b = 0; b < m1.batch; b++){
        
        const double* pm1temp = pm1 + b * m1size; // shift pm1 by one batch worth
        const double* pm2temp = !bcast ? pm2 + b * m2size : pm2; // only shift if m2 is 3D
        double* pmtemp = pm + b * msize;

        for (int th = 0; th < n_threads; th++)
        {
            int temp = (th < n_threads - 1) ? stride : stride + rem;
            threads[th] = std::thread(

                // we dont want to capture everything in scope !
                [th, stride, temp, pm1temp, pm2temp, pmtemp](size_t m1col, size_t m2col)
                {
                    for (size_t i = th * stride; i < (th * stride) + temp; i++) {
                        for (size_t k = 0; k < m2col; k++) {

                            double sum = 0;
                            for (size_t j = 0; j < m1col; j++) {
                                sum += pm1temp[i * m1col + j] * pm2temp[j * m2col + k];
                            }
                            pmtemp[i * m2col + k] = sum;
                        }
                    }
                },
                
                // pass these are parameters cause we dont want to copy the entire tensor
                m1.col, m2.col);
        }

        // free
        for (int i = 0; i < n_threads; i++) threads[i].join();
    }

    // clean up
    delete[] threads;
    return m;
}

Tensor matrixOperations::cops(const Tensor& m1, const double con, double (*f)(double, double)) 
{
    Tensor m = Tensor::create(m1.shape.get(), m1.rank);

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

Tensor matrixOperations::argmax(const Tensor& m1)
{
    std::vector<int> temp(m1.rank - 1);
    for (int i = 0; i < m1.rank - 1; i ++) temp[i] = m1.shape[i];
    Tensor m = Tensor::create(temp);

    const double* pm1 = m1.tensor.get();
    double* pm = m.tensor.get();

    #pragma omp parallel for
    for (size_t i = 0; i < m1.batch * m1.row; i++)
    {
        double temp_val = -1e19;
        for (size_t j = 0; j < m1.col; j++) 
        {
            if (pm1[i * m1.col + j] > temp_val) {pm[i] = j; temp_val = pm1[i * m1.col + j];}
        }
    }
    return m;
}

Tensor matrixOperations::softmax(const Tensor& m1)
{
    Tensor m = Tensor::create(m1.shape.get(), m1.rank);

    const double* pm1 = m1.tensor.get();
    double* pm = m.tensor.get();

    #pragma omp parallel for
    for (size_t i = 0; i < m1.batch * m1.row; i++)
    {
        double sum = 1e-19;
        for (size_t j = 0; j < m1.col; j++) sum += std::exp(pm1[i * m1.col + j]);
        for (size_t j = 0; j < m1.col; j++) pm[i * m1.col + j] = std::exp(pm1[i * m1.col + j]) / sum;
    }
    return m;
}

Tensor matrixOperations::activation(const Tensor& m1, const char ops)
{
    Tensor m = Tensor::create(m1.shape.get(), m1.rank);

    const double* pm1 = m1.tensor.get();
    double* pm = m.tensor.get();

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

    return loss / m1.tot_size;
}

double matrixOperations::categoricalcrossentropy(const Tensor& m1, const Tensor& m2, Tensor& m /*m is same as pred*/) // m1 is real and m2 pred !!
{
    // TO DO: catch mismatch tensor

    const double* pm1 = m1.tensor.get(); // grab raw pointers for speeeed
    const double* pm2 = m2.tensor.get();
    double* pm = m.tensor.get();
    double loss = 0.0;
    const double eps = 1e-19;
    
    const size_t num_classes = m2.shape[m2.rank - 1];

    #pragma omp parallel for reduction(+:loss)
    for (size_t i = 0; i < m1.tot_size; i++) 
    {   
        size_t tempid = i * num_classes;

        // find max per class and subtract to make stable
        double cur_max = pm2[tempid];
        for (size_t j = 0; j < num_classes; j++) { if (pm2[tempid + j] > cur_max) cur_max = pm2[tempid + j]; }
    
        double sum = 1e-19;
        for (size_t j = 0; j < num_classes; j++) sum += std::exp(pm2[tempid + j] - cur_max);
        
        for (size_t j = 0; j < num_classes; j++)
        {
            double p = std::exp(pm2[tempid + j] - cur_max) / sum;
            if (j == (size_t)pm1[i])
            {
                loss -= std::log(p + eps);
                pm[i * m.shape[m.rank - 1] + j] = p - 1; // gradient
            }
            else pm[i * m.shape[m.rank - 1] + j] = p;
        }
        
    }
    
    return loss / m1.tot_size;
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

