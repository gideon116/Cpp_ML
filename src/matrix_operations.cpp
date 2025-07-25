#include <iostream>
#include "../include/matrix_operations.h"

Tensor wef::matmul(const Tensor& m1, const Tensor& m2)
{
    if (m1.col != m2.row) throw std::invalid_argument("matrix size mismatch [3]");
    const bool not_bcast = (m2.batch != 1);
    if (not_bcast && (m1.batch != m2.batch)) throw std::invalid_argument("matrix size mismatch [4]");
    
    std::unique_ptr<size_t[]> temp_shape = std::make_unique<size_t[]>(m1.rank);

    // TO DO: CATCH < 1 RANK
    for (size_t i = 0; i < m1.rank - 1; i ++) temp_shape[i] = m1.shape[i];
    temp_shape[m1.rank - 1] = m2.col;

    Tensor m = Tensor::create(temp_shape.get(), m1.rank);

    const float* pm1 = m1.tensor.get(); // grab raw pointers for speeeed
    const float* pm2 = m2.tensor.get();
    float* pm = m.tensor.get();

    const size_t m1size = m1.row * m1.col; // to shift pm1 by one batch worth
    const size_t m2size = m2.row * m2.col * not_bcast; // only shift if m2 is 3D

    /*
    // second option for matmul, maybe slower but its a flat loop
    #pragma omp parallel for schedule(static)
    for (size_t elem = 0; elem < m.tot_size; elem++)
    {
        size_t b = elem / (m.row * m.col);
        size_t c = b * m2size + elem % m.col;

        size_t r = b * m1size + ((elem / m.col) % m.row) * m1.col;

        float sum = 0;
        for (size_t i = 0; i < m1.col; i++)
                sum += pm1[i + r] * pm2[i * m2.col + c];
                
        pm[elem] = sum;
    }
    */

    const size_t msize = m1.row * m2.col;
    const float* pm1temp;
    const float* pm2temp;
    float* pmtemp;
    for (size_t b = 0; b < m1.batch; b++){
        
        pm1temp = pm1 + b * m1size; // shift pm1 by one batch worth
        pm2temp = pm2 + b * m2size; // only shift if m2 is 3D
        pmtemp = pm + b * msize;

        #pragma omp parallel for collapse(2) schedule(static)
        for (size_t i = 0; i < m1.row; i++) {
            for (size_t k = 0; k < m2.col; k++) {

                float sum = 0.0f;
                for (size_t j = 0; j < m1.col; j++) {
                    sum += pm1temp[i * m1.col + j] * pm2temp[j * m2.col + k];
                }
                pmtemp[i * m2.col + k] = sum;
            }
        }
    }
    return m;
}

Tensor wef::matmul(const Tensor& m1, const Tensor& m2, bool, size_t n_threads)
{
    if (m1.col != m2.row) throw std::invalid_argument("matrix size mismatch [3]");
    const bool not_bcast = (m2.batch != 1);
    if (not_bcast && (m1.batch != m2.batch)) throw std::invalid_argument("matrix size mismatch [4]");
    
    std::unique_ptr<size_t[]> temp_shape = std::make_unique<size_t[]>(m1.rank);

    // TO DO: CATCH < 1 RANK
    for (size_t i = 0; i < m1.rank - 1; i ++) temp_shape[i] = m1.shape[i];
    temp_shape[m1.rank - 1] = m2.col;

    Tensor m = Tensor::create(temp_shape.get(), m1.rank);

    const float* pm1 = m1.tensor.get();
    const float* pm2 = m2.tensor.get();
    float* pm = m.tensor.get();

    const size_t m1size = m1.row * m1.col; // to shift pm1 by one batch worth
    const size_t m2size = m2.row * m2.col * not_bcast; // only shift if m2 is 3D
    const size_t msize = m1.row * m2.col;
    const float* pm1temp;
    const float* pm2temp;
    float* pmtemp;

    // multi thread additions
    if (n_threads == 0)
    {
        size_t avaliable_threads = std::thread::hardware_concurrency(); // may be 0
        n_threads = std::min<size_t>( m1.row,  avaliable_threads > 0 ? avaliable_threads : 1 );
    }
    const size_t stride = m1.row / n_threads;
    const size_t rem = m1.row % n_threads;

    // spin up
    std::thread* threads = new std::thread[n_threads];

    for (size_t b = 0; b < m1.batch; b++){
        
        pm1temp = pm1 + b * m1size; // shift pm1 by one batch worth
        pm2temp = pm2 + b * m2size; // only shift if m2 is 3D
        pmtemp = pm + b * msize;

        for (size_t th = 0; th < n_threads; th++)
        {
            size_t temp = (th < n_threads - 1) ? stride : stride + rem;
            threads[th] = std::thread(

                // we dont want to capture everything in scope !
                [th, stride, temp, pm1temp, pm2temp, pmtemp](size_t m1col, size_t m2col)
                {
                    // #pragma omp parallel for collapse(2) schedule(static) <--------- remove?
                    for (size_t i = th * stride; i < (th * stride) + temp; i++) {
                        for (size_t k = 0; k < m2col; k++) {

                            float sum = 0;
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
        for (size_t i = 0; i < n_threads; i++) threads[i].join();
    }

    // clean up
    delete[] threads;
    return m;
}

Tensor wef::cops(const Tensor& m1, const float con, float (*f)(float, float)) 
{
    Tensor m = Tensor::create(m1.shape.get(), m1.rank);

    const float* pm1 = m1.tensor.get();
    float* pm = m.tensor.get();

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < m1.tot_size; i++) pm[i] = f(pm1[i], con);

    return m;

}

Tensor wef::transpose(const Tensor& m1)
{
    std::unique_ptr<size_t[]> temp_shape = std::make_unique<size_t[]>(m1.rank);
    // TO DO: CATCH < 2 RANK
    for (size_t i = 0; i < m1.rank - 2; i ++) temp_shape[i] = m1.shape[i];
    temp_shape[m1.rank - 1] = m1.row;
    temp_shape[m1.rank - 2] = m1.col;
    Tensor m = Tensor::create(temp_shape.get(), m1.rank);

    const float* pm1 = m1.tensor.get();
    float* pm = m.tensor.get();

    const size_t msize = m1.row * m1.col;
    const float* pm1temp;
    float* pmtemp;

    for (size_t b = 0; b < m1.batch; b++){

        pm1temp = pm1 + b * msize;
        pmtemp = pm + b * msize;

        #pragma omp parallel for collapse(2) schedule(static)
        for (size_t i = 0; i < m1.row; i++) {
            for (size_t j = 0; j < m1.col; j++) {
                pmtemp[j * m1.row + i] = pm1temp[i * m1.col + j];
            }
        }
    }
    return m;
}

Tensor wef::argmax(const Tensor& m1)
{
    // ENSURE RANK > 0 and make it work with other axis not just -1
    std::unique_ptr<size_t[]> temp_shape = std::make_unique<size_t[]>(m1.rank - 1);

    for (size_t i = 0; i < m1.rank - 1; i ++) temp_shape[i] = m1.shape[i];
    Tensor m = Tensor::create(temp_shape.get(), m1.rank - 1);

    const float* pm1 = m1.tensor.get();
    float* pm = m.tensor.get();

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < m1.tot_size / m1.col; i++)
    {
        float temp_val = -1e19f;
        for (size_t j = 0; j < m1.col; j++) 
        {
            if (pm1[i * m1.col + j] > temp_val) {pm[i] = j; temp_val = pm1[i * m1.col + j];}
        }
    }
    return m;
}

Tensor wef::softmax(const Tensor& m1)
{
    Tensor m = Tensor::create(m1.shape.get(), m1.rank);

    const float* pm1 = m1.tensor.get();
    float* pm = m.tensor.get();

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < m1.tot_size / m1.col; i++)
    {
        float sum = 1e-19f;
        for (size_t j = 0; j < m1.col; j++) sum += std::exp(pm1[i * m1.col + j]);
        for (size_t j = 0; j < m1.col; j++) pm[i * m1.col + j] = std::exp(pm1[i * m1.col + j]) / sum;
    }
    return m;
}

Tensor wef::activation(const Tensor& m1, const char ops)
{
    Tensor m = Tensor::create(m1.shape.get(), m1.rank);

    const float* pm1 = m1.tensor.get();
    float* pm = m.tensor.get();

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < m1.tot_size; i++) 
    {
        switch (ops) {
            // relu
            case 'a':
                pm[i] = pm1[i] > 0 ?  pm1[i] : 0.0f;
                break;
            // derivative relu
            case 'b':
                pm[i] = pm1[i] > 0 ? 1.0f : 0.0f;
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

Tensor wef::reducesum(const Tensor& m1, const int ax)
{   
    if (ax >= m1.rank) throw std::invalid_argument("axis outside shape");
    if (ax < 0) throw std::invalid_argument("axis connot be negative");

    std::unique_ptr<size_t[]> out_shape = std::make_unique<size_t[]>(m1.rank); // [b, 1, w, c]
    
    for (size_t i = 0; i < m1.rank; i++)
    {
        if (i != ax) out_shape[i] = m1.shape[i];
        else out_shape[i] = 1;
    }

    Tensor m = Tensor::create(out_shape.get(), m1.rank); 
    const float* pm1 = m1.tensor.get();
    float* pm = m.tensor.get();
    
    std::memset(pm, 0, (m.tot_size) * sizeof(float));

    size_t eaa = 1; // everything after axis i.e. b, h w, axis, x1, x2 -> eaa = x1 * x2
    for (size_t i = ax + 1; i < m1.rank; i++) eaa *= m1.shape[i];
    size_t ax_help = m1.shape[ax]*eaa;

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < m1.tot_size; i++) pm[ (i % eaa) + eaa * (i / ax_help) ] += pm1[i];
    
    return m;
}

float wef::l2(const Tensor& m1, const Tensor& m2)
{
    if (m1.rank != m2.rank) throw std::invalid_argument("matrix rank mismatch [6]");
    for (size_t i = 0; i < m1.rank; i++) if (m1.shape[i] != m2.shape[i]) throw std::invalid_argument("matrix size mismatch [7]");

    const float* pm1 = m1.tensor.get(); // grab raw pointers for speeeed
    const float* pm2 = m2.tensor.get();
    float loss = 0.0f;

    #pragma omp parallel for reduction(+:loss) schedule(static) 
    for (size_t i = 0; i < m1.tot_size; i++) 
    {
        loss += std::pow(pm1[i] - pm2[i], 2);
    }

    return loss / (m1.tot_size);
}

float wef::binarycrossentropy(const Tensor& m1, const Tensor& m2) // m1 is real and m2 pred !!
{
    // TO DO: catch mismatch tensor

    const float* pm1 = m1.tensor.get(); // grab raw pointers for speeeed
    const float* pm2 = m2.tensor.get();
    float loss = 0.0f;
    const float eps = 1e-19f;
    
    #pragma omp parallel for reduction(+:loss) schedule(static) 
    for (size_t i = 0; i < m1.tot_size; i++)
    {   
        size_t temp_real = pm1[i] > 0.5;
        
        loss += -(temp_real * std::log(pm2[i] + eps) + (1 - temp_real) * std::log(1 - pm2[i] + eps));
    }

    return loss / m1.tot_size;
}

float wef::categoricalcrossentropy(const Tensor& m1, const Tensor& m2, Tensor& m /*m is same as pred*/) // m1 is real and m2 pred !!
{
    // Note m1 is actual labels and m2 is probabilities 
    // eg: m1 = {{1}, {2}}, m2 = {{0, 1, 0}, {0, 0, 1}}

    // TO DO: catch mismatch tensor

    const float* pm1 = m1.tensor.get(); // grab raw pointers for speeeed
    const float* pm2 = m2.tensor.get();
    float* pm = m.tensor.get();
    float loss = 0.0f;
    const float eps = 1e-19f;
    
    const size_t num_classes = m2.shape[m2.rank - 1];

    #pragma omp parallel for reduction(+:loss) schedule(static)
    for (size_t i = 0; i < m1.tot_size; i++) 
    {   
        size_t tempid = i * num_classes;
        size_t base = i * m.shape[m.rank - 1];

        // find max per class and subtract to make stable
        float cur_max = pm2[tempid];
        for (size_t j = 0; j < num_classes; j++) { if (pm2[tempid + j] > cur_max) cur_max = pm2[tempid + j]; }
    
        float sum = 1e-19f;
        for (size_t j = 0; j < num_classes; j++) sum += std::exp(pm2[tempid + j] - cur_max);
        
        for (size_t j = 0; j < num_classes; j++)
        {
            float p = std::exp(pm2[tempid + j] - cur_max) / sum;
            if (j == (size_t)pm1[i])
            {
                loss -= std::log(p + eps);
                pm[base + j] = p - 1; // gradient
            }
            else pm[base + j] = p;
        }
        
    }
    
    return loss / m1.tot_size;
}

float wef::categoricalcrossentropy(const Tensor& m1, const Tensor& m2) // m1 is real and m2 pred !!
{
    // TO DO: catch mismatch tensor

    // Note m1 is actual labels and m2 is probabilities 
    // eg: m1 = {{1}, {2}}, m2 = {{0, 1, 0}, {0, 0, 1}}

    const float* pm1 = m1.tensor.get(); // grab raw pointers for speeeed
    const float* pm2 = m2.tensor.get();
    float loss = 0.0f;
    const float eps = 1e-19f;
    
    const size_t num_classes = m2.shape[m2.rank - 1];

    #pragma omp parallel for reduction(+:loss) schedule(static) 
    for (size_t i = 0; i < m1.tot_size; i++) 
    {   
        size_t tempid = i * num_classes;

        // find max per class and subtract to make stable
        float cur_max = pm2[tempid];
        for (size_t j = 0; j < num_classes; j++) { if (pm2[tempid + j] > cur_max) cur_max = pm2[tempid + j]; }
    
        float sum = 1e-19f;
        for (size_t j = 0; j < num_classes; j++) sum += std::exp(pm2[tempid + j] - cur_max);
        
        for (size_t j = 0; j < num_classes; j++)
        {
            float p = std::exp(pm2[tempid + j] - cur_max) / sum;
            if (j == (size_t)pm1[i]) loss -= std::log(p + eps);
        }
    }
    
    return loss / m1.tot_size;
}

void wef::print(const Tensor& m1, size_t* arr, size_t num, bool allc)
{   
    if (!allc) arr = new size_t[m1.rank];
    
    if (num < m1.rank - 1)
    {   
        std::cout << "{ ";

        for (size_t i = 0; i < m1.shape[num]; i++)
        {
            arr[num] = i;
            
            num == 0 ? std::cout << "\n" : std::cout << "";

            print(m1, arr, num + 1, true);
            
        }
        num == 0 ? std::cout << "\n" : std::cout << "";
        std::cout << "} ";
    }
    else
    {
        std::cout << "{ ";
        for (size_t i = 0; i < m1.shape[num]; i++) 
        {
            arr[num] = i;
            std::cout << m1.index(arr) << " ";
            
        }
        std::cout << "} ";
    }

    if (!allc) delete[] arr;

}

Tensor wef::pow(const Tensor& m1, const float con) { return cops(m1, con, [](float a, float b){ return std::pow(a, b); }); }
Tensor wef::relu(const Tensor& m1) { return activation(m1, 'a'); }
Tensor wef::d_relu(const Tensor& m1) { return activation(m1, 'b'); }
Tensor wef::sigmoid(const Tensor& m1) { return activation(m1, 'c'); }
Tensor wef::d_sigmoid(const Tensor& m1) { return activation(m1, 'd'); }