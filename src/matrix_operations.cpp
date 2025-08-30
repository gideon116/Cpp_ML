#include <iostream>
#include "../include/matrix_operations.h"

Tensor wef::matmul(const Tensor& m1, const Tensor& m2)
{
    if (m1.m_rank < 2 || m2.m_rank < 2)
        throw std::invalid_argument("tensor 1 and tensor 2 rank must be > 1");

    size_t M = m1.m_shape[m1.m_rank - 2];
    size_t N = m1.m_shape[m1.m_rank - 1];
    size_t K = m2.m_shape[m2.m_rank - 1];
    if (m2.m_shape[m2.m_rank - 2] != N)
        throw std::invalid_argument("matrix size mismatch [3]");

    bool bcast = false;
    if (m2.m_rank == 2) // check broadcast
        bcast = true;
    else // if m2 is not a simple matrix like a 2d weight then compare the whole tensors up to the last 2 elements
        if (memcmp(m1.m_shape, m2.m_shape, sizeof(size_t) * m1.m_rank - 2 * sizeof(size_t))) // compare everything but the last 2
            throw std::invalid_argument("matrix size mismatch [4]");

    std::unique_ptr<size_t[]> temp_shape = std::make_unique<size_t[]>(m1.m_rank);
    memcpy(temp_shape.get(), m1.m_shape, sizeof(size_t) * m1.m_rank);
    temp_shape[m1.m_rank - 1] = K;

    Tensor m = Tensor::create(temp_shape.get(), m1.m_rank);

    const float* pm1 = m1.m_tensor; // grab raw pointers for speeeed
    const float* pm2 = m2.m_tensor;
    float* pm = m.m_tensor;

    size_t batch = m1.m_size/(M*N);

    const size_t m1size = M * N; // to shift pm1 by one batch worth
    const size_t m2size = N * K * !bcast; // only shift if m2 is not broadcast

    /*
    // second option for matmul, maybe slower but its a flat loop
    #pragma omp parallel for schedule(static)
    for (size_t elem = 0; elem < m.m_size; elem++)
    {
        size_t b = elem / (M * K);
        size_t c = b * m2size + elem % K;

        size_t r = b * m1size + ((elem / K) % M) * N;

        float sum = 0;
        for (size_t i = 0; i < N; i++)
                sum += pm1[i + r] * pm2[i * K + c];
        pm[elem] = sum;
    }
    */

    const size_t msize = M * K;
    const float* pm1temp;
    const float* pm2temp;
    float* pmtemp;
    for (size_t b = 0; b < batch; b++)
    {
        pm1temp = pm1 + b * m1size; // shift pm1 by one batch worth
        pm2temp = pm2 + b * m2size; // only shift if m2 is 3D
        pmtemp = pm + b * msize;

        #pragma omp parallel for collapse(2) schedule(static)
        for (size_t i = 0; i < M; i++)
            for (size_t q = 0; q < K; q++)
            {
                float sum = 0.0f;
                for (size_t j = 0; j < N; j++)
                    sum += pm1temp[i * N + j] * pm2temp[j * K + q];
                pmtemp[i * K + q] = sum;
            }
    }

    return m;
}

Tensor wef::matmul(const Tensor& m1, const Tensor& m2, bool, size_t n_threads)
{
    if (m1.m_rank < 2 || m2.m_rank < 2)
        throw std::invalid_argument("tensor 1 and tensor 2 rank must be > 1");

    size_t M = m1.m_shape[m1.m_rank - 2];
    size_t N = m1.m_shape[m1.m_rank - 1];
    size_t K = m2.m_shape[m2.m_rank - 1];
    if (m2.m_shape[m2.m_rank - 2] != N)
        throw std::invalid_argument("matrix size mismatch [3]");

    bool bcast = false;
    if (m2.m_rank == 2) // check broadcast
        bcast = true;
    else // if m2 is not a simple matrix like a 2d weight then compare the whole tensors up to the last 2 elements
        if (memcmp(m1.m_shape, m2.m_shape, sizeof(size_t) * m1.m_rank - 2 * sizeof(size_t))) // compare everything but the last 2
            throw std::invalid_argument("matrix size mismatch [4]");

    std::unique_ptr<size_t[]> temp_shape = std::make_unique<size_t[]>(m1.m_rank);
    memcpy(temp_shape.get(), m1.m_shape, sizeof(size_t) * m1.m_rank);
    temp_shape[m1.m_rank - 1] = K;

    Tensor m = Tensor::create(temp_shape.get(), m1.m_rank);

    const float* pm1 = m1.m_tensor; // grab raw pointers for speeeed
    const float* pm2 = m2.m_tensor;
    float* pm = m.m_tensor;

    size_t batch = m1.m_size/(M*N);

    const size_t m1size = M * N; // to shift pm1 by one batch worth
    const size_t m2size = N * K * !bcast; // only shift if m2 is not broadcast

    const size_t msize = M * K;
    const float* pm1temp;
    const float* pm2temp;
    float* pmtemp;

    // multi thread additions
    if (n_threads == 0)
    {
        size_t avaliable_threads = std::thread::hardware_concurrency(); // may be 0
        n_threads = std::min<uint>(M, avaliable_threads > 0 ? avaliable_threads : 1);
    }
    const size_t stride = M / n_threads;
    const size_t rem = M % n_threads;

    // spin up
    std::thread* threads = new std::thread[n_threads];

    for (size_t b = 0; b < batch; b++){
        
        pm1temp = pm1 + b * m1size; // shift pm1 by one batch worth
        pm2temp = pm2 + b * m2size; // only shift if m2 is 3D
        pmtemp = pm + b * msize;

        for (size_t th = 0; th < n_threads; th++)
        {
            size_t temp = (th < n_threads - 1) ? stride : stride + rem;
            threads[th] = std::thread(
                // we dont want to capture everything in scope !
                [th, stride, temp, pm1temp, pm2temp, pmtemp, N, K]()
                {
                    // #pragma omp parallel for collapse(2) schedule(static) <--------- remove?
                    for (size_t i = th * stride; i < (th * stride) + temp; i++)
                        for (size_t q = 0; q < K; q++)
                        {
                            float sum = 0;
                            for (size_t j = 0; j < N; j++)
                                sum += pm1temp[i * N + j] * pm2temp[j * K + q];
                            pmtemp[i * K + q] = sum;
                        }
                }
            );
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
    Tensor m = Tensor::create(m1.m_shape, m1.m_rank);

    const float* pm1 = m1.m_tensor;
    float* pm = m.m_tensor;

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < m1.m_size; i++) pm[i] = f(pm1[i], con);

    return m;

}

Tensor wef::transpose(const Tensor& m1)
{
    // TODO: make this a wrapper for the full perm below
    if (m1.m_rank < 2) throw std::invalid_argument("tensor rank must be > 1");

    size_t* perm = new size_t[m1.m_rank];
    for (size_t i = 0; i < m1.m_rank - 2; i++)
        perm[i] = i;
    perm[m1.m_rank - 1] = m1.m_rank - 2;
    perm[m1.m_rank - 2] = m1.m_rank - 1;

    return transpose(m1, perm);
    
    /*
    size_t M = m1.m_shape[m1.m_rank - 2];
    size_t N = m1.m_shape[m1.m_rank - 1];

    Tensor m = m1; // creating new tensor and value will be overwritten
    m.m_shape[m1.m_rank - 1] = M;
    m.m_shape[m1.m_rank - 2] = N;

    const float* pm1 = m1.m_tensor;
    float* pm = m.m_tensor;

    const size_t msize = M * N;
    size_t batch = m1.m_size/(M*N);

    const float* pm1temp;
    float* pmtemp;

    for (size_t b = 0; b < batch; b++)
    {
        pm1temp = pm1 + b * msize;
        pmtemp = pm + b * msize;

        #pragma omp parallel for collapse(2) schedule(static)
        for (size_t i = 0; i < M; i++)
            for (size_t j = 0; j < N; j++)
                pmtemp[j * M + i] = pm1temp[i * N + j];
    }
    return m;
    */
}

void transpose_helper(const size_t* old_shape, size_t* new_shape, const size_t rank, const size_t* perm,
                        size_t* old_stride, size_t*  new_stride, size_t* p_inv)
{
    old_stride[rank - 1] = 1;
    new_stride[rank - 1] = 1;

    for (size_t i = 0; i < rank; i++)
        new_shape[i] = old_shape[perm[i]];

    for (int i = (int)rank - 2; i >= 0; i--)
    {
        old_stride[i] = old_stride[i + 1] * old_shape[i + 1];
        new_stride[i] = new_stride[i + 1] * new_shape[i + 1]; 
    }

    for (size_t i = 0; i < rank; i++) 
        p_inv[perm[i]] = i; 
}

size_t transpose_index_mapper(const size_t index, const size_t rank, const size_t* new_shape, 
                                size_t* multi, const size_t* new_stride, const size_t* old_stride, const size_t* p_inv)
{
    // dest linear to dest multi
    for (size_t i = 0; i < rank; i++) 
        multi[i] = (index / new_stride[i]) % new_shape[i]; 

    // dest multi to source linear
    size_t prem_index = 0;
    for (size_t i = 0; i < rank; i++)
        prem_index += multi[p_inv[i]] * old_stride[i];

    return prem_index;
}

Tensor wef::transpose(const Tensor& m1, const size_t* perm)
{
    // TODO : add prem check

    if (m1.m_rank < 2)
        throw std::invalid_argument("tensor rank must be > 1");
    
    std::unique_ptr<size_t[]> old_stride, new_stride, multi, p_inv;

    old_stride  = std::make_unique<size_t[]>(m1.m_rank);
    new_stride  = std::make_unique<size_t[]>(m1.m_rank);
    multi       = std::make_unique<size_t[]>(m1.m_rank);
    p_inv       = std::make_unique<size_t[]>(m1.m_rank);

    Tensor m = m1; // creating new tensor and value will be overwritten
    transpose_helper(m1.m_shape, m.m_shape, m1.m_rank, perm, old_stride.get(), new_stride.get(), p_inv.get());

    const float* pm1 = m1.m_tensor;
    float* pm = m.m_tensor;

    for (size_t i = 0; i < m.m_size; i++)
    {   
        size_t index = transpose_index_mapper(i, m.m_rank, m.m_shape, multi.get(), new_stride.get(), old_stride.get(), p_inv.get());
        // if (index >= m1.m_size)
        //     throw std::invalid_argument("[ERROR ] SOMETHING WENT WRONG IN TRANSPOSE");
        pm[i] = pm1[index];
    }

    return m;
}

Tensor wef::argmax(const Tensor& m1)
{
    if (m1.m_rank < 1)
        throw std::invalid_argument("tensor 1 rank must be > 0");

    // TODO : make it work with other axis not just -1
    std::unique_ptr<size_t[]> temp_shape = std::make_unique<size_t[]>(m1.m_rank - 1);
    for (size_t i = 0; i < m1.m_rank - 1; i ++) temp_shape[i] = m1.m_shape[i];

    Tensor m = Tensor::create(temp_shape.get(), m1.m_rank - 1);

    const float* pm1 = m1.m_tensor;
    float* pm = m.m_tensor;

    size_t last_axis = m1.m_shape[m1.m_rank-1];

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < m1.m_size / last_axis; i++)
    {
        float temp_val = -1e19f;
        for (size_t j = 0; j < last_axis; j++) 
            if (pm1[i * last_axis + j] > temp_val)
            {
                pm[i] = j;
                temp_val = pm1[i * last_axis + j];
            }
    }
    return m;
}

Tensor wef::softmax(const Tensor& m1)
{
    Tensor m = Tensor::create(m1.m_shape, m1.m_rank);

    const float* pm1 = m1.m_tensor;
    float* pm = m.m_tensor;

    size_t last_axis = m1.m_shape[m1.m_rank-1];

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < m1.m_size / last_axis; i++)
    {
        float sum = 1e-19f;
        for (size_t j = 0; j < last_axis; j++)
            sum += std::exp(pm1[i * last_axis + j]);
        for (size_t j = 0; j < last_axis; j++)
            pm[i * last_axis + j] = std::exp(pm1[i * last_axis + j]) / sum;
    }
    return m;
}

Tensor wef::activation(const Tensor& m1, const char ops)
{
    Tensor m = Tensor::create(m1.m_shape, m1.m_rank);

    const float* pm1 = m1.m_tensor;
    float* pm = m.m_tensor;

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < m1.m_size; i++) 
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

Tensor wef::reducesum(const Tensor& m1, const int ax, const bool keepdims)
{   
    if (ax >= m1.m_rank)
        throw std::invalid_argument("axis outside shape");
    if (ax < 0)
        throw std::invalid_argument("axis connot be negative");

    size_t out_rank = keepdims ? m1.m_rank : m1.m_rank - 1;

    std::unique_ptr<size_t[]> out_shape = std::make_unique<size_t[]>(out_rank); // [b, 1, w, c]
    
    size_t j = 0;
    for (size_t i = 0; i < m1.m_rank; i++)
    {
        if (i != ax)
            out_shape[j++] = m1.m_shape[i];
        else
            if (keepdims)
                out_shape[j++] = 1;
    }

    Tensor m = Tensor::create(out_shape.get(), out_rank); 
    const float* pm1 = m1.m_tensor;
    float* pm = m.m_tensor;
    
    std::memset(pm, 0, (m.m_size) * sizeof(float));

    size_t eaa = 1; // everything after axis i.e. b, h w, axis, x1, x2 -> eaa = x1 * x2
    for (size_t i = ax + 1; i < m1.m_rank; i++)
        eaa *= m1.m_shape[i];
    size_t ax_help = m1.m_shape[ax]*eaa;

    #pragma omp parallel for schedule(static) // TODO : data races
    for (size_t i = 0; i < m1.m_size; i++)
        pm[ (i % eaa) + eaa * (i / ax_help) ] += pm1[i];
    
    return m;
}

float wef::l2(const Tensor& m1, const Tensor& m2)
{
    if (m1.m_rank != m2.m_rank)
        throw std::invalid_argument("matrix rank mismatch [6]");
    for (size_t i = 0; i < m1.m_rank; i++) if (m1.m_shape[i] != m2.m_shape[i])
        throw std::invalid_argument("matrix size mismatch [7]");

    const float* pm1 = m1.m_tensor; // grab raw pointers for speeeed
    const float* pm2 = m2.m_tensor;
    float loss = 0.0f;

    #pragma omp parallel for reduction(+:loss) schedule(static) 
    for (size_t i = 0; i < m1.m_size; i++) 
    {
        loss += std::pow(pm1[i] - pm2[i], 2);
    }

    return loss / (m1.m_size);
}

float wef::binarycrossentropy(const Tensor& m1, const Tensor& m2) // m1 is real and m2 pred !!
{
    // TODO: catch mismatch tensor

    const float* pm1 = m1.m_tensor; // grab raw pointers for speeeed
    const float* pm2 = m2.m_tensor;
    float loss = 0.0f;
    const float eps = 1e-19f;
    
    #pragma omp parallel for reduction(+:loss) schedule(static) 
    for (size_t i = 0; i < m1.m_size; i++)
    {   
        size_t temp_real = pm1[i] > 0.5;
        
        loss += -(temp_real * std::log(pm2[i] + eps) + (1 - temp_real) * std::log(1 - pm2[i] + eps));
    }

    return loss / m1.m_size;
}

float wef::categoricalcrossentropy(const Tensor& m1, const Tensor& m2, Tensor& m /*m is same as pred*/) // m1 is real and m2 pred !!
{
    // Note m1 is actual labels and m2 is probabilities 
    // eg: m1 = {{1}, {2}}, m2 = {{0, 1, 0}, {0, 0, 1}}

    // TODO: catch mismatch tensor

    const float* pm1 = m1.m_tensor; // grab raw pointers for speeeed
    const float* pm2 = m2.m_tensor;
    float* pm = m.m_tensor;
    float loss = 0.0f;
    const float eps = 1e-19f;
    
    const size_t num_classes = m2.m_shape[m2.m_rank - 1];

    #pragma omp parallel for reduction(+:loss) schedule(static)
    for (size_t i = 0; i < m1.m_size; i++) 
    {   
        size_t tempid = i * num_classes;
        size_t base = i * m.m_shape[m.m_rank - 1];

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
    
    return loss / m1.m_size;
}

float wef::categoricalcrossentropy(const Tensor& m1, const Tensor& m2) // m1 is real and m2 pred !!
{
    // TODO: catch mismatch tensor

    // Note m1 is actual labels and m2 is probabilities 
    // eg: m1 = {{1}, {2}}, m2 = {{0, 1, 0}, {0, 0, 1}}

    const float* pm1 = m1.m_tensor; // grab raw pointers for speeeed
    const float* pm2 = m2.m_tensor;
    float loss = 0.0f;
    const float eps = 1e-19f;
    
    const size_t num_classes = m2.m_shape[m2.m_rank - 1];

    #pragma omp parallel for reduction(+:loss) schedule(static) 
    for (size_t i = 0; i < m1.m_size; i++) 
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
    
    return loss / m1.m_size;
}

void wef::print(const Tensor& m1, size_t* arr, size_t num, bool allc)
{   
    if (!allc) arr = new size_t[m1.m_rank];
    
    if (num < m1.m_rank - 1)
    {   
        std::cout << "{ ";

        for (size_t i = 0; i < m1.m_shape[num]; i++)
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
        for (size_t i = 0; i < m1.m_shape[num]; i++) 
        {
            arr[num] = i;
            std::cout << m1[arr] << " ";
            
        }
        std::cout << "} ";
    }

    if (!allc) delete[] arr;

}

Tensor wef::elemwise_GPU(const void* gpu, const Tensor& m1, const Tensor& m2, const int operation/* 0 add, 1 sub, 2 mul*/)
{
    // TODO : add broadcast ability
    if (m1.m_rank != m2.m_rank)
        throw std::invalid_argument("tensor 1 and tensor 2 must have the same shape");
    if (memcmp(m1.m_shape, m2.m_shape, sizeof(size_t) * m1.m_rank)) // compare shapes
        throw std::invalid_argument("matrix size mismatch [4]");

    struct PC
    {
        uint32_t operation; // 0 add, 1 sub, 2 mul
        uint32_t size;
    } push_constant;

    const char* spv_path =  "shaders/binaries/elemwise.spv";
    VkDeviceSize bytes = sizeof(float) * m1.m_size;

    Tensor m = m1;

    push_constant.operation = operation;
    push_constant.size = m1.m_size;

    const uint32_t WG = 256;
    uint32_t gx = UseGPU::ceilDiv(m1.m_size, WG);
    uint32_t gy = 1;
    uint32_t gz = 1;

    ((UseGPU*)gpu)->program({bytes, bytes}, {bytes}, {m1.m_tensor, m2.m_tensor}, {m.m_tensor}, spv_path, (void*)&push_constant, sizeof(push_constant), gx, gy, gz);
    
    return m;
}

Tensor wef::matmul_GPU(const void* gpu, const Tensor& m1, const Tensor& m2)
{
    if (m1.m_rank < 2 || m2.m_rank < 2)
        throw std::invalid_argument("tensor 1 and tensor 2 rank must be > 1");

    size_t M = m1.m_shape[m1.m_rank - 2];
    size_t N = m1.m_shape[m1.m_rank - 1];
    size_t K = m2.m_shape[m2.m_rank - 1];
    if (m2.m_shape[m2.m_rank - 2] != N)
        throw std::invalid_argument("matrix size mismatch [3]");

    bool bcast = false;
    if (m2.m_rank == 2) // check broadcast
        bcast = true;
    else // if m2 is not a simple matrix like a 2d weight then compare the whole tensors up to the last 2 elements
        if (memcmp(m1.m_shape, m2.m_shape, sizeof(size_t) * m1.m_rank - 2 * sizeof(size_t))) // compare everything but the last 2
            throw std::invalid_argument("matrix size mismatch [4]");

    std::unique_ptr<size_t[]> temp_shape = std::make_unique<size_t[]>(m1.m_rank);
    memcpy(temp_shape.get(), m1.m_shape, sizeof(size_t) * m1.m_rank);
    temp_shape[m1.m_rank - 1] = K;

    Tensor m = Tensor::create(temp_shape.get(), m1.m_rank);

    const char* spv_path =  "shaders/binaries/matmul.spv";

    VkDeviceSize sizeA = sizeof(float) * m1.m_size;
    VkDeviceSize sizeB = sizeof(float) * m2.m_size;
    VkDeviceSize sizeC = sizeof(float) * m.m_size;

    struct PC
    {
        uint32_t    m1_r, m1_c, m2_c,
                    batch, 
                    m1_stride, m2_stride, m_stride;
    } push_constant;

    push_constant.m1_r = M;
    push_constant.m1_c = N;
    push_constant.m2_c = K;
    push_constant.batch = m1.m_size/(M*N); // assume no m1 bcast
    push_constant.m1_stride = M * N;
    push_constant.m2_stride = N * K * !bcast; // set to 0 to broadcast B across batches
    push_constant.m_stride = M * K;

    const uint32_t WGX = 16;
    const uint32_t WGY = 16;
    uint32_t gx = UseGPU::ceilDiv(K, WGX);
    uint32_t gy = UseGPU::ceilDiv(M, WGY);
    uint32_t gz = m1.m_size/(M*N);

    ((UseGPU*)gpu)->program({sizeA, sizeB}, {sizeC}, {m1.m_tensor, m2.m_tensor}, {m.m_tensor}, spv_path, (void*)&push_constant, sizeof(push_constant), gx, gy, gz);
    return m;
}

Tensor wef::positional_encoding(const size_t& length, const size_t& depth)
{
    Tensor pos_encoding = Tensor::create((size_t[3]){1, length, depth}, 3);

    for (size_t pos = 0; pos < length; pos++)
    {
        for (size_t dep = 0; dep < depth; dep+=2)
        {
            float angle_rates = 1 / std::pow(10000.0f, (float)dep / (float)depth);
            float angle_rads = (float)pos * angle_rates;

            pos_encoding.m_tensor[pos * depth + dep] = std::sin(angle_rads);

            if (dep + 1 < depth) // odd
                pos_encoding.m_tensor[pos * depth + dep + 1] = std::cos(angle_rads);
        }
    }
    return pos_encoding;
}

Tensor wef::pow(const Tensor& m1, const float con) { return cops(m1, con, [](float a, float b){ return std::pow(a, b); }); }
Tensor wef::relu(const Tensor& m1) { return activation(m1, 'a'); }
Tensor wef::d_relu(const Tensor& m1) { return activation(m1, 'b'); }
Tensor wef::sigmoid(const Tensor& m1) { return activation(m1, 'c'); }
Tensor wef::d_sigmoid(const Tensor& m1) { return activation(m1, 'd'); }