#include "layers.h"

static uint32_t s_ceil_div(uint32_t a, uint32_t b) { return (a + b - 1) / b; }

struct PC_C2DF
{
    uint32_t inH, inW, inC;
    uint32_t kH, kW, outC;
    uint32_t outH, outW;
    uint32_t batch;
    uint32_t biasOffset;
};

__global__ void k_conv2D_f(float* X, float* W, float* O, PC_C2DF pc)
{
    uint wioc   = blockIdx.x * blockDim.x + threadIdx.x;
    uint hi     = blockIdx.y * blockDim.y + threadIdx.y;
    uint bi     = blockIdx.z * blockDim.z + threadIdx.z;

    uint wi = wioc / pc.outC;
    uint oc = wioc % pc.outC;

    auto index_A = [&](uint b, uint hi, uint wi, uint ci)
    {
        return ((b * pc.inH + hi) * pc.inW + wi) * pc.inC + ci;
    };

    auto index_B = [&](uint hk, uint wk, uint ci, uint co)
    {
        return ((hk * pc.kW + wk) * pc.inC + ci) * pc.outC + co;
    };

    auto index_C = [&](uint b, uint ho, uint wo, uint co)
    {
        return ((b * pc.outH + ho) * pc.outW + wo) * pc.outC + co;
    };

    
    if (wi < pc.outW && hi < pc.outH && bi < pc.batch && oc < pc.outC)
    {
        float temp = 0.0f;
        for (uint h2 = 0; h2 < pc.kH; h2++)
            for (uint w2 = 0; w2 < pc.kW; w2++)
                for (uint c2 = 0; c2 < pc.inC; c2++)
                    temp += X[index_A(bi, h2 + hi, w2 + wi, c2)] * W[index_B(h2, w2, c2, oc)];

        if (pc.biasOffset != 0) 
            temp += W[pc.biasOffset + oc];
        O[index_C(bi, hi, wi, oc)] = temp;
    }
}

struct PC_C2DB
{
    uint32_t inH, inW, inC;
    uint32_t kH, kW, outC;
    uint32_t outH, outW;
    uint32_t batch;
};

__global__ void k_conv2D_b_dw(float* dy, float* X, float* dw, PC_C2DB pc)
{
    auto index_A = [&](uint b, uint hi, uint wi, uint ci)
    { return ((b * pc.inH + hi) * pc.inW + wi) * pc.inC + ci; };

    auto index_B = [&](uint hk, uint wk, uint ci, uint co)
    { return ((hk * pc.kW + wk) * pc.inC + ci) * pc.outC + co; };

    auto index_C = [&](uint b, uint ho, uint wo, uint co)
    { return ((b * pc.outH + ho) * pc.outW + wo) * pc.outC + co; };

    uint co = blockIdx.x * blockDim.x + threadIdx.x;
    uint cin = blockIdx.y * blockDim.y + threadIdx.y;
    uint hk_wk = blockIdx.z * blockDim.z + threadIdx.z;

    uint hk = hk_wk / pc.kW;
    uint wk = hk_wk % pc.kW;

    if (wk >= pc.kW || hk >= pc.kH || cin >= pc.inC || co >= pc.outC) {}

    else
    {
        float temp = 0.0f;
        for (uint bin = 0; bin < pc.batch; bin++)
        {
            for (uint ho = 0; ho < pc.outH; ho++)
            {
                uint hin = ho + hk;
                for (uint wo = 0; wo < pc.outW; wo++)
                {
                    uint win = wo + wk;
                    temp += dy[index_C(bin, ho, wo, co)] * X[index_A(bin, hin, win, cin)];
                }       
            }
        }

        dw[index_B(hk, wk, cin, co)] = temp;

    }
}

__global__ void k_conv2D_b_dx(float* W, float* dy, float* dx, PC_C2DB pc)
{

    auto index_A = [&](uint b, uint hi, uint wi, uint ci)
    { return ((b * pc.inH + hi) * pc.inW + wi) * pc.inC + ci; };

    auto index_B = [&](uint hk, uint wk, uint ci, uint co)
    { return ((hk * pc.kW + wk) * pc.inC + ci) * pc.outC + co; };

    auto index_C = [&](uint b, uint ho, uint wo, uint co)
    { return ((b * pc.outH + ho) * pc.outW + wo) * pc.outC + co; };

    uint bin = blockIdx.x * blockDim.x + threadIdx.x;
    uint win_cin = blockIdx.y * blockDim.y + threadIdx.y;
    uint hin = blockIdx.z * blockDim.z + threadIdx.z;

    uint win = win_cin / pc.inC;
    uint cin = win_cin % pc.inC;

    if (win >= pc.inW || hin >= pc.inH || bin >= pc.batch || cin >= pc.inC) {}
    else
    {
        float temp = 0.0f;
        for (uint hk = 0; hk < pc.kH; hk++)
        {
            if (hin < hk)
                continue;

            uint ho = hin - hk;
            for (uint wk = 0; wk < pc.kW; wk++)
            {
                if (win < wk)
                    continue;

                uint wo = win - wk;
                for (uint co = 0; co < pc.outC; co++)
                {
                    if (ho >= pc.outH || wo >= pc.outW)
                        continue;

                    temp += dy[index_C(bin, ho, wo, co)] * W[index_B(hk, wk, cin, co)];
                }
            }
        }

        dx[index_A(bin, hin, win, cin)] = temp;

    }
}

struct PC_MP2DF
{
    uint32_t inH, inW, inC;
    uint32_t kH, kW, outC;
    uint32_t outH, outW;
    uint32_t batch;
    bool training;
};

__global__ void k_mp2D_f(float* px, uint* argmax, float* O, PC_MP2DF pc)
{
    auto index_A = [&](uint b, uint hi, uint wi, uint ci)
    { return ((b * pc.inH + hi) * pc.inW + wi) * pc.inC + ci; };

    auto index_C = [&](uint b, uint ho, uint wo, uint co)
    { return ((b * pc.outH + ho) * pc.outW + wo) * pc.outC + co; };

    uint wioc = blockIdx.x * blockDim.x + threadIdx.x;
    uint hi = blockIdx.y * blockDim.y + threadIdx.y;
    uint bi = blockIdx.z * blockDim.z + threadIdx.z;

    uint wi = wioc / pc.outC;
    uint oc = wioc % pc.outC;

    if (wi < pc.outW && hi < pc.outH && bi < pc.batch && oc < pc.outC)
    {
        float temp_val = -1e19f;
        uint temp_ind[4];
        for (uint h2 = hi * pc.kH; h2 < hi * pc.kH + pc.kH; h2++)
        {
            if (h2 < pc.inH)
            {
                for (uint w2 = wi * pc.kW; w2 < wi * pc.kW + pc.kW; w2++)
                {
                    if (w2 < pc.inW)
                    {
                        float val = px[index_A(bi, h2, w2, oc)];
                        if (val > temp_val)
                        {
                            temp_val = val;
                            temp_ind[0] = bi;
                            temp_ind[1] = h2;
                            temp_ind[2] = w2;
                            temp_ind[3] = oc;
                        }
                    }
                }
            }
        }
        O[index_C(bi, hi, wi, oc)] = temp_val;

        if (pc.training)
            for (uint ii = 0; ii < 4; ii++)
                argmax[index_C(bi, hi, wi, oc) * 4 + ii] = temp_ind[ii];

    }
}

struct PC_MP2DB
{
    uint32_t inH, inW, inC;
    uint32_t outC;
    uint32_t outH, outW;
    uint32_t batch;
};

__global__ void k_mp2D_b(uint* argmax, float* dy, float* dx, PC_MP2DB pc)
{
    auto index_A = [&](uint b, uint hi, uint wi, uint ci)
    { return ((b * pc.inH + hi) * pc.inW + wi) * pc.inC + ci; };

    auto index_C = [&](uint b, uint ho, uint wo, uint co)
    { return ((b * pc.outH + ho) * pc.outW + wo) * pc.outC + co; };

    uint wioc   = blockIdx.x * blockDim.x + threadIdx.x;
    uint hi     = blockIdx.y * blockDim.y + threadIdx.y;
    uint bi     = blockIdx.z * blockDim.z + threadIdx.z;

    uint wi = wioc / pc.outC;
    uint oc = wioc % pc.outC;

    if (wi < pc.outW && hi < pc.outH && bi < pc.batch && oc < pc.outC)
    {
        uint i1[4];
        i1[0] = argmax[index_C(bi, hi, wi, oc) * 4 + 0];
        i1[1] = argmax[index_C(bi, hi, wi, oc) * 4 + 1];
        i1[2] = argmax[index_C(bi, hi, wi, oc) * 4 + 2];
        i1[3] = argmax[index_C(bi, hi, wi, oc) * 4 + 3];

        dx[index_A(i1[0], i1[1], i1[2], i1[3])] = dy[index_C(bi, hi, wi, oc)];
    }
}

void check_mem(size_t size_temp, size_t& m_size, float*& m_gpu, const char* name="", const char* location="")
{
    if (size_temp > m_size)
    {
        std::cout << "[WARNING ] GPU memory reallocation triggred for " << name << " in " << location << ".\n";
        if (m_gpu)
            cudaFree(m_gpu);
        m_gpu = nullptr;
        m_size = size_temp;
        cudaMalloc(&m_gpu, m_size);
    }
    cudaMemset(m_gpu, 0, m_size);
}

void check_mem(size_t size_temp, size_t& m_size, uint32_t*& m_gpu, const char* name="", const char* location="")
{
    if (size_temp > m_size)
    {
        std::cout << "[WARNING ] GPU memory reallocation triggred for " << name << " in " << location << ".\n";
        if (m_gpu)
            cudaFree(m_gpu);
        m_gpu = nullptr;
        m_size = size_temp;
        cudaMalloc(&m_gpu, m_size);
    }
    cudaMemset(m_gpu, 0, m_size);
}

Tensor* Linear_GPU::forward_pass(const Tensor* px, const bool training, void* gpu) 
{
    if (!m_init) 
    {   
        // initially initilize the shape of m_X later just copy the tensors
        m_X = Tensor(*px);
        m_dist = std::normal_distribution<float>(0.0f, std::sqrt( 2.0f / (px->m_shape[px->m_rank-1])));

        size_t w_shape[2] = {px->m_shape[px->m_rank-1], m_units};
        size_t b_shape[2] = {1, m_units};
        m_W = Tensor::create(w_shape, 2);
        m_B = Tensor::create(b_shape, 2);

        std::fill_n(m_B.m_tensor, m_B.m_size, 0.0f); // zero fill

        float* pm = m_W.m_tensor;
        for (size_t i = 0; i < size_t(px->m_shape[px->m_rank-1]) * m_units; i++)
            pm[i] = m_dist(m_g);

        m_num_param = m_W.m_size + (m_use_bias ? m_B.m_size : 0);

        m_out_rank = px->m_rank;
        m_out_shape = std::make_unique<size_t[]>(m_out_rank);
        std::memcpy(m_out_shape.get(), px->m_shape, m_out_rank * sizeof(size_t));
        
        // TODO: CATCH < 1 RANK
        m_out_shape[m_out_rank - 1] = m_units;

        m_size_a = sizeof(float) * px->m_size;
        m_size_b = sizeof(float) * m_W.m_size;
        m_size_c = sizeof(float);
        
        for (size_t i = 0; i < m_out_rank; i++)
            m_size_c *= m_out_shape[i];

        cudaMalloc(&m_a_gpu, m_size_a);
        cudaMalloc(&m_b_gpu, m_size_b);
        cudaMalloc(&m_c_gpu, m_size_c);

        m_init = true;
    }
    else
    {
        // if trying to use (reuse) the layer on a different tensor
        if (m_W.m_shape[m_W.m_rank-2] != px->m_shape[px->m_rank-1])
            throw std::invalid_argument("cannot reuse layer");
    }
    
    // copy px into m_X
    if (training)
        std::memcpy(m_X.m_tensor, px->m_tensor, m_X.m_size * sizeof(float));

    // recalculate size a b c
    size_t size_a_temp = sizeof(float) * px->m_size;
    size_t size_b_temp = sizeof(float) * m_W.m_size;
    size_t size_c_temp = sizeof(float) * m_units;
    for (size_t i = 0; i < px->m_rank - 1; i++)
        size_c_temp *= px->m_shape[i];

    check_mem(size_a_temp, m_size_a, m_a_gpu, "m_a_gpu", "Linear GPU CUDA");
    check_mem(size_b_temp, m_size_b, m_b_gpu, "m_b_gpu", "Linear GPU CUDA");
    check_mem(size_c_temp, m_size_c, m_c_gpu, "m_c_gpu", "Linear GPU CUDA");
    
    if (m_use_bias)
    {    
        out = wef::matmul_GPU(gpu, *px, m_W, m_a_gpu, m_b_gpu, m_c_gpu);
        out = out + m_B;
        // TODO: (MAJOR) we cant apply the sequence below cause broadcast is not enabled

        // size_a_temp = sizeof(float) * out.m_size;
        // size_b_temp = sizeof(float) * m_B.m_size;
        // size_c_temp = sizeof(float) * out.m_size;

        // check_mem(size_a_temp, m_size_a, m_a_gpu, "m_a_gpu", "Linear GPU CUDA");
        // check_mem(size_b_temp, m_size_b, m_b_gpu, "m_b_gpu", "Linear GPU CUDA");
        // check_mem(size_c_temp, m_size_c, m_c_gpu, "m_c_gpu", "Linear GPU CUDA");
        
        // out = wef::elemwise_GPU(gpu, out, m_B, /*operation=add=*/0, m_a_gpu, m_b_gpu, m_c_gpu);
    }
    else
        out = wef::matmul_GPU(gpu, *px, m_W, m_a_gpu, m_b_gpu, m_c_gpu);

    return &out;

}

Tensor* Linear_GPU::backward_pass(const Tensor* dy, const float lr, void* gpu) 
{

    // recalculate size a b c
    size_t size_a_temp = sizeof(float) * dy->m_size;
    size_t size_b_temp = sizeof(float) * m_W.m_size; // will be transposed later but that doesnt change the size
    size_t size_c_temp = sizeof(float) * m_W.m_shape[0]; // cause its going to be transposed
    for (size_t i = 0; i < dy->m_rank - 1; i++)
        size_c_temp *= dy->m_shape[i];

    check_mem(size_a_temp, m_size_a, m_a_gpu, "m_a_gpu", "Linear GPU CUDA");
    check_mem(size_b_temp, m_size_b, m_b_gpu, "m_b_gpu", "Linear GPU CUDA");
    check_mem(size_c_temp, m_size_c, m_c_gpu, "m_c_gpu", "Linear GPU CUDA");

    // gradient wrt the layer below
    m_dx = wef::matmul_GPU(gpu, *dy, wef::transpose(m_W), m_a_gpu, m_b_gpu, m_c_gpu);

    // recalculate size a b c
    Tensor X_T = wef::transpose(m_X);
    size_a_temp = sizeof(float) * X_T.m_size;
    size_b_temp = sizeof(float) * dy->m_size; 
    size_c_temp = sizeof(float) * dy->m_shape[dy->m_rank - 1];
    for (size_t i = 0; i < X_T.m_rank - 1; i++)
        size_c_temp *= X_T.m_shape[i];

    check_mem(size_a_temp, m_size_a, m_a_gpu, "m_a_gpu", "Linear GPU CUDA");
    check_mem(size_b_temp, m_size_b, m_b_gpu, "m_b_gpu", "Linear GPU CUDA");
    check_mem(size_c_temp, m_size_c, m_c_gpu, "m_c_gpu", "Linear GPU CUDA");

    // gradient wrt weights sum everything aside from the last two axes. 
    m_dw = wef::matmul_GPU(gpu, X_T, *dy, m_a_gpu, m_b_gpu, m_c_gpu);

    while (m_dw.m_rank > m_W.m_rank)
        m_dw = wef::reducesum(m_dw, /*axis*/0, /*keepkims*/false);

    // recalculate size a b c
    size_a_temp = sizeof(float) * m_W.m_size;
    size_b_temp = sizeof(float) * m_dw.m_size; 
    size_c_temp = sizeof(float) * m_W.m_size;

    check_mem(size_a_temp, m_size_a, m_a_gpu, "m_a_gpu", "Linear GPU CUDA");
    check_mem(size_b_temp, m_size_b, m_b_gpu, "m_b_gpu", "Linear GPU CUDA");
    check_mem(size_c_temp, m_size_c, m_c_gpu, "m_c_gpu", "Linear GPU CUDA");

    m_W = wef::elemwise_GPU(gpu, m_W, m_dw * lr / dy->m_shape[0], /*operation=subtract=*/1, m_a_gpu, m_b_gpu, m_c_gpu);
    // m_W -= m_dw * lr / dy->m_shape[0];

    if (m_use_bias)
    {
        // gradient wrt bias sum everything aside from the last axis
        m_db = *dy;
        while (m_db.m_rank > m_B.m_rank)
            m_db = wef::reducesum(m_db, /*axis*/0, /*keepkims*/false);
        m_db = wef::reducesum(m_db, 0, /*keepkims*/true); // because bias is shape = [1, bias]

        // recalculate size a b c
        size_a_temp = sizeof(float) * m_B.m_size;
        size_b_temp = sizeof(float) * m_db.m_size; 
        size_c_temp = sizeof(float) * m_B.m_size;

        check_mem(size_a_temp, m_size_a, m_a_gpu, "m_a_gpu", "Linear GPU CUDA");
        check_mem(size_b_temp, m_size_b, m_b_gpu, "m_b_gpu", "Linear GPU CUDA");
        check_mem(size_c_temp, m_size_c, m_c_gpu, "m_c_gpu", "Linear GPU CUDA");

        m_B = wef::elemwise_GPU(gpu, m_B, m_db * lr / dy->m_shape[0], 1, m_a_gpu, m_b_gpu, m_c_gpu);
        // m_B -= m_db * lr / dy->m_shape[0];
    }

    return &m_dx;
}

Tensor* Conv2D_GPU::forward_pass(const Tensor* px, const bool training, void*) 
{
    if (!m_init) 
    {   
        // initially initilize the shape of m_X later just copy the tensors
        m_X = Tensor(*px);

        // h, w, c, m_units
        m_height = px->m_shape[1]; m_width = px->m_shape[2]; m_ch = px->m_shape[3];
        m_dist = std::normal_distribution<float>(0.0f, std::sqrt( 2.0f / (m_k_height * m_k_width * m_ch))); 
        // for above now we have (2/fan_in = hwc)^0.5 good for relu we can use fan_out for tanh... which is hwu

        size_t w_shape[4] = {m_k_height, m_k_width, m_ch, m_units};
        m_W = Tensor::create(w_shape, 4);

        size_t m_B_shape[4] = {1, 1, 1, m_units};
        m_B = Tensor::create(m_B_shape, 4);
        std::fill_n(m_B.m_tensor, m_B.m_size, 0.0f);

        float* pm = m_W.m_tensor;
        for (size_t i = 0; i < m_W.m_size; i++) pm[i] = m_dist(m_g);

        m_out_rank = px->m_rank; // this is 4, its always 4
        m_out_shape = std::make_unique<size_t[]>(m_out_rank); // heap allocation is not the best but we only do this once pre layer so its whatever
        m_out_shape[1] = m_height - m_k_height + 1;
        m_out_shape[2] = m_width - m_k_width + 1;
        m_out_shape[3] = m_units;

        // gradients wrt weights and biases
        m_dx = Tensor(*px);

        // gradient wrt weights
        m_dw = Tensor(m_W);
        m_db = Tensor(m_B);

        // weight + bias buffer
        m_WB_size = m_W.m_size + (m_use_bias ? m_units : 0);
        m_WB = std::make_unique<float[]>(m_WB_size);

        m_num_param = m_W.m_size + (m_use_bias ? m_B.m_size : 0);

        m_size_a = sizeof(float) * px->m_size;
        m_size_b = sizeof(float) * m_WB_size;
        m_size_c = sizeof(float) * px->m_shape[0];
        
        for (size_t i = 1; i < px->m_rank; i++)
            m_size_c *= m_out_shape[i];

        cudaMalloc(&m_a_gpu, m_size_a);
        cudaMalloc(&m_b_gpu, m_size_b);
        cudaMalloc(&m_c_gpu, m_size_c);
        m_init = true;
    }

    else
    {
        // if trying to use (reuse) the layer on a different tensor
        if (px->m_shape[1] != m_height || 
            px->m_shape[2] != m_width ||
            px->m_shape[3] != m_ch)
                throw std::invalid_argument("cannot reuse layer");
    }
    
    // copy px into m_X
    if (training)
        std::memcpy(m_X.m_tensor, px->m_tensor, m_X.m_size * sizeof(float));

    m_out_shape[0] = px->m_shape[0]; // flexable batch
    m_out = Tensor::create(m_out_shape.get(), 4);
    std::memset(m_out.m_tensor, 0, (m_out.m_size) * sizeof(float));

    std::memcpy(m_WB.get(), m_W.m_tensor, m_W.m_size * sizeof(float));
    uint32_t biasOffset = 0;
    if (m_use_bias)
    {
        biasOffset = m_W.m_size;
        std::memcpy(m_WB.get() + m_W.m_size, m_B.m_tensor, m_units * sizeof(float));
    }

    PC_C2DF push_constant;
    push_constant.inH = m_height;
    push_constant.inW = m_width;
    push_constant.inC = m_ch;
    push_constant.kH = m_k_height;
    push_constant.kW = m_k_width;
    push_constant.outC = m_out_shape[3];
    push_constant.outH = m_out_shape[1];
    push_constant.outW = m_out_shape[2];
    push_constant.batch = m_out_shape[0];
    push_constant.biasOffset = biasOffset;

    const uint32_t WGX = 16;
    const uint32_t WGY = 16;
    const uint32_t WGZ = 1;

    uint32_t gx = s_ceil_div(push_constant.outW * push_constant.outC, WGX);
    uint32_t gy = s_ceil_div(push_constant.outH, WGY);
    uint32_t gz = s_ceil_div(push_constant.batch, WGZ);

    dim3 dimBlock(WGX, WGY, WGZ);
    dim3 dimGrid(gx, gy, gz);
    
    // recalculate size a b c
    size_t size_a_temp = sizeof(float) * px->m_size;
    size_t size_b_temp = sizeof(float) * m_WB_size;
    size_t size_c_temp = sizeof(float) * m_out.m_size;

    check_mem(size_a_temp, m_size_a, m_a_gpu, "m_a_gpu", "Conv2D GPU CUDA");
    check_mem(size_b_temp, m_size_b, m_b_gpu, "m_b_gpu", "Conv2D GPU CUDA");
    check_mem(size_c_temp, m_size_c, m_c_gpu, "m_c_gpu", "Conv2D GPU CUDA");

    m_a = px->m_tensor;
    m_b = m_WB.get();
    m_c = m_out.m_tensor;

    // copy inputs to gpu
    cudaMemcpy(m_a_gpu, m_a, size_a_temp, cudaMemcpyHostToDevice);
    cudaMemcpy(m_b_gpu, m_b, size_b_temp, cudaMemcpyHostToDevice);

    // operation on gpu
    k_conv2D_f<<<dimGrid, dimBlock>>>(m_a_gpu, m_b_gpu, m_c_gpu, push_constant);

    // copy res to cpu
    cudaMemcpy(m_c, m_c_gpu, size_c_temp, cudaMemcpyDeviceToHost);

    return &m_out;
}

Conv2D_GPU::~Conv2D_GPU()
{
    if (m_a_gpu)
        cudaFree(m_a_gpu);
    m_a_gpu = nullptr;

    if (m_b_gpu)
        cudaFree(m_b_gpu);
    m_b_gpu = nullptr;

    if (m_c_gpu)
        cudaFree(m_c_gpu);
    m_c_gpu = nullptr;
}

Tensor* Conv2D_GPU::backward_pass(const Tensor* dy, const float lr, void* gpu) 
{   
    float* m_dx_ptr = m_dx.m_tensor;
    float* m_dw_ptr = m_dw.m_tensor;

    std::memset(m_dx_ptr, 0, (m_dx.m_size) * sizeof(float)); // zero fill
    std::memset(m_dw_ptr, 0, (m_dw.m_size) * sizeof(float)); // zero fill

    // gpu computation
    PC_C2DB push_constant;

    push_constant.inH = m_height;
    push_constant.inW = m_width;
    push_constant.inC = m_ch;
    push_constant.kH = m_k_height;
    push_constant.kW = m_k_width;
    push_constant.outC = m_units;
    push_constant.outH = dy->m_shape[1];
    push_constant.outW = dy->m_shape[2];
    push_constant.batch = dy->m_shape[0];

    const uint32_t WGX = 16;
    const uint32_t WGY = 16;
    const uint32_t WGZ = 1;

    {
        float* b = m_W.m_tensor;
        float* c = dy->m_tensor;
        float* a = m_dx.m_tensor;

        uint32_t gx = s_ceil_div(m_dx.m_shape[0], WGX);
        uint32_t gy = s_ceil_div(m_dx.m_shape[2] * m_dx.m_shape[3], WGY);
        uint32_t gz = s_ceil_div(m_dx.m_shape[1], WGZ);

        dim3 dimBlock(WGX, WGY, WGZ);
        dim3 dimGrid(gx, gy, gz);

        // recalculate size a b c
        size_t size_b_temp = sizeof(float) * m_W.m_size;
        size_t size_c_temp = sizeof(float) * dy->m_size;
        size_t size_a_temp = sizeof(float) * m_dx.m_size;

        check_mem(size_a_temp, m_size_a, m_a_gpu, "m_a_gpu", "Conv2D Backward GPU CUDA");
        check_mem(size_b_temp, m_size_b, m_b_gpu, "m_b_gpu", "Conv2D Backward GPU CUDA");
        check_mem(size_c_temp, m_size_c, m_c_gpu, "m_c_gpu", "Conv2D Backward GPU CUDA");

        cudaMemcpy(m_b_gpu, b, size_b_temp, cudaMemcpyHostToDevice);
        cudaMemcpy(m_c_gpu, c, size_c_temp, cudaMemcpyHostToDevice);
        
        k_conv2D_b_dx<<<dimGrid, dimBlock>>>(m_b_gpu, m_c_gpu, m_a_gpu, push_constant);
        
        // cudaDeviceSynchronize();
        cudaMemcpy(a, m_a_gpu, size_a_temp, cudaMemcpyDeviceToHost);

    }

    {
        float* c = dy->m_tensor;
        float* a = m_X.m_tensor;
        float* b = m_dw.m_tensor;

        uint32_t gx = s_ceil_div(m_dw.m_shape[3], WGX);
        uint32_t gy = s_ceil_div(m_dw.m_shape[2], WGY);
        uint32_t gz = s_ceil_div(m_dw.m_shape[0] * m_dw.m_shape[1], WGZ);

        dim3 dimBlock(WGX, WGY, WGZ);
        dim3 dimGrid(gx, gy, gz);

        // recalculate size a b c
        size_t size_c_temp = sizeof(float) * dy->m_size;
        size_t size_a_temp = sizeof(float) * m_X.m_size;
        size_t size_b_temp = sizeof(float) * m_dw.m_size;

        check_mem(size_a_temp, m_size_a, m_a_gpu, "m_a_gpu", "Conv2D Backward GPU CUDA");
        check_mem(size_b_temp, m_size_b, m_b_gpu, "m_b_gpu", "Conv2D Backward GPU CUDA");
        check_mem(size_c_temp, m_size_c, m_c_gpu, "m_c_gpu", "Conv2D Backward GPU CUDA");
        
        cudaMemcpy(m_c_gpu, c, size_c_temp, cudaMemcpyHostToDevice);
        cudaMemcpy(m_a_gpu, a, size_a_temp, cudaMemcpyHostToDevice);
        
        k_conv2D_b_dw<<<dimGrid, dimBlock>>>(m_c_gpu, m_a_gpu, m_b_gpu, push_constant);
        
        // cudaDeviceSynchronize();
        cudaMemcpy(b, m_b_gpu, size_b_temp, cudaMemcpyDeviceToHost);
    }

    // divide lr by batch size
    m_W = wef::elemwise_GPU(gpu, m_W, m_dw * lr / dy->m_shape[0], 1); // or
    // m_W -= m_dw * lr / dy->m_shape[0];

    if (m_use_bias)
    {
        m_db = *dy;
        for (size_t i = 0; i < m_db.m_rank - 1; i++)
            m_db = wef::reducesum(m_db, i, /*keepkims*/true);
        m_B = wef::elemwise_GPU(gpu, m_B, m_db * lr / dy->m_shape[0], 1);
        // m_B -= m_db * lr / dy->m_shape[0];
    }

    return &m_dx;
}

Tensor* MaxPool2D_GPU::forward_pass(const Tensor* px, const bool training, void* gpu) 
{
    if (!m_init)
    {   
        // initially initilize the shape of m_X later just copy the tensors
        m_X = Tensor(*px);
        // h, w, c, m_units
        m_height = px->m_shape[1]; m_width = px->m_shape[2]; m_ch = px->m_shape[3];

        size_t ax1 = (m_height + (m_height%k_height)) / k_height;
        size_t ax2 = (m_width + (m_width%k_width)) / k_width;

        m_argmax_len = 4 * px->m_shape[0] * ax1 * ax2 * m_ch;
        
        // this get the argmax in a nested for loop (2D) I made it flat for speed
        argmax = std::make_unique<uint32_t[]>(m_argmax_len);

        m_out_rank = px->m_rank; // this is 4, its always 4
        m_out_shape = std::make_unique<size_t[]>(m_out_rank); // heap allocation is not the best but we only do this once pre layer so its whatever
        m_out_shape[1] = ax1;
        m_out_shape[2] = ax2;
        m_out_shape[3] = m_ch;
        
        // m_dx is gradient wrt the layer below
        m_dx = Tensor(*px);

        m_size_a = sizeof(float) * px->m_size;
        m_size_b = m_argmax_len * sizeof(uint32_t);
        m_size_c = sizeof(float) * px->m_shape[0];
        
        for (size_t i = 1; i < px->m_rank; i++)
            m_size_c *= m_out_shape[i];

        cudaMalloc(&m_a_gpu, m_size_a);
        cudaMalloc(&m_b_gpu, m_size_b);
        cudaMalloc(&m_c_gpu, m_size_c);

        m_init = true;
    }
    else
    {
        // if trying to use (reuse) the layer on a different tensor
        if (px->m_shape[1] != m_height || 
            px->m_shape[2] != m_width ||
            px->m_shape[3] != m_ch)
                throw std::invalid_argument("cannot reuse layer");
    }

    // copy px into m_X
    if (training) std::memcpy(m_X.m_tensor, px->m_tensor, m_X.m_size * sizeof(float)); // TODO : is m_X even used in back prop?

    // batch is flexable
    m_out_shape[0] = px->m_shape[0];
    m_out = Tensor::create(m_out_shape.get(), 4);

    PC_MP2DF push_constant;
    push_constant.inH = m_height;
    push_constant.inW = m_width;
    push_constant.inC = m_ch;
    push_constant.kH = k_height;
    push_constant.kW = k_width;
    push_constant.outC = m_out_shape[3];
    push_constant.outH = m_out_shape[1];
    push_constant.outW = m_out_shape[2];
    push_constant.batch = m_out_shape[0];
    push_constant.training = training;
    
    const uint32_t WGX = 16;
    const uint32_t WGY = 16;
    const uint32_t WGZ = 1;

    float* a = px->m_tensor;
    uint32_t* b = argmax.get();
    float* c = m_out.m_tensor;

    uint32_t gx = s_ceil_div(push_constant.outW * push_constant.outC, WGX);
    uint32_t gy = s_ceil_div(push_constant.outH, WGY);
    uint32_t gz = s_ceil_div(push_constant.batch, WGZ);

    dim3 dimBlock(WGX, WGY, WGZ);
    dim3 dimGrid(gx, gy, gz);

    // recalculate size a b c
    size_t size_a_temp = sizeof(float) * px->m_size;
    size_t size_b_temp = m_argmax_len * sizeof(uint32_t);
    size_t size_c_temp = sizeof(float) * m_out.m_size;

    check_mem(size_a_temp, m_size_a, m_a_gpu, "m_a_gpu", "MaxPool2D GPU CUDA");
    check_mem(size_b_temp, m_size_b, m_b_gpu, "m_b_gpu", "MaxPool2D GPU CUDA");
    check_mem(size_c_temp, m_size_c, m_c_gpu, "m_c_gpu", "MaxPool2D GPU CUDA");

    cudaMemcpy(m_a_gpu, a, size_a_temp, cudaMemcpyHostToDevice);
    
    k_mp2D_f<<<dimGrid, dimBlock>>>(m_a_gpu, m_b_gpu, m_c_gpu, push_constant);
    
    // cudaDeviceSynchronize();
    cudaMemcpy(b, m_b_gpu, size_b_temp, cudaMemcpyDeviceToHost);
    cudaMemcpy(c, m_c_gpu, size_c_temp, cudaMemcpyDeviceToHost);
        

    return &m_out;
}

Tensor* MaxPool2D_GPU::backward_pass(const Tensor* dy, const float lr, void* gpu) 
{
    std::memset(m_dx.m_tensor, 0, (m_dx.m_size) * sizeof(float));  // zero fill

    PC_MP2DB push_constant;
    push_constant.inH = m_height;
    push_constant.inW = m_width;
    push_constant.inC = m_ch;
    push_constant.outC = m_out_shape[3];
    push_constant.outH = m_out_shape[1];
    push_constant.outW = m_out_shape[2];
    push_constant.batch = m_out_shape[0];

    const uint32_t WGX = 16;
    const uint32_t WGY = 16;
    const uint32_t WGZ = 1;

    float* a = dy->m_tensor;
    uint32_t* b = argmax.get();
    float* c = m_dx.m_tensor;

    uint32_t gx = s_ceil_div(push_constant.outW * push_constant.outC, WGX);
    uint32_t gy = s_ceil_div(push_constant.outH, WGY);
    uint32_t gz = s_ceil_div(push_constant.batch, WGZ);

    dim3 dimBlock(WGX, WGY, WGZ);
    dim3 dimGrid(gx, gy, gz);

    // recalculate size a b c
    size_t size_a_temp = sizeof(float) * dy->m_size;
    size_t size_b_temp = m_argmax_len * sizeof(uint32_t);
    size_t size_c_temp = sizeof(float) * m_dx.m_size;

    check_mem(size_a_temp, m_size_a, m_a_gpu, "m_a_gpu", "MaxPool2D Backward GPU CUDA");
    check_mem(size_b_temp, m_size_b, m_b_gpu, "m_b_gpu", "MaxPool2D Backward GPU CUDA");
    check_mem(size_c_temp, m_size_c, m_c_gpu, "m_c_gpu", "MaxPool2D Backward GPU CUDA");

    cudaMemcpy(m_a_gpu, a, size_a_temp, cudaMemcpyHostToDevice);
    cudaMemcpy(m_b_gpu, b, size_b_temp, cudaMemcpyHostToDevice);
    
    k_mp2D_b<<<dimGrid, dimBlock>>>(m_b_gpu, m_a_gpu, m_c_gpu, push_constant);
    
    // cudaDeviceSynchronize();
    cudaMemcpy(c, m_c_gpu, size_c_temp, cudaMemcpyDeviceToHost);
    
    return &m_dx;
}

MaxPool2D_GPU::~MaxPool2D_GPU()
{
    if (m_a_gpu)
        cudaFree(m_a_gpu);
    m_a_gpu = nullptr;

    if (m_b_gpu)
        cudaFree(m_b_gpu);
    m_b_gpu = nullptr;

    if (m_c_gpu)
        cudaFree(m_c_gpu);
    m_c_gpu = nullptr;
}
