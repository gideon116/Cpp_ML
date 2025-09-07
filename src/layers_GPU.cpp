#include "useGPU.h"
#include "layers.h"

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
        for (size_t i = 0; i < size_t(px->m_shape[px->m_rank-1]) * m_units; i++) pm[i] = m_dist(m_g);

        m_num_param = m_W.m_size + (m_use_bias ? m_B.m_size : 0);

        m_out_rank = px->m_rank;
        m_out_shape = std::make_unique<size_t[]>(m_out_rank);
        std::memcpy(m_out_shape.get(), px->m_shape, m_out_rank * sizeof(size_t));
        // TODO: CATCH < 1 RANK
        m_out_shape[m_out_rank - 1] = m_units;

        m_init = true;
    }
    else
    {
        // if trying to use (reuse) the layer on a different tensor
        if (m_W.m_shape[m_W.m_rank-2] != px->m_shape[px->m_rank-1])
            throw std::invalid_argument("cannot reuse layer");
    }
    
    // copy px into m_X
    if (training) std::memcpy(m_X.m_tensor, px->m_tensor, m_X.m_size * sizeof(float));
    
    if (m_use_bias) out = wef::matmul_GPU(gpu, *px, m_W) + m_B;
    else out = wef::matmul_GPU(gpu, *px, m_W);
    return &out;

}

Tensor* Linear_GPU::backward_pass(const Tensor* dy, const float lr, void* gpu) 
{
    // gradient wrt the layer below
    m_dx = wef::matmul_GPU(gpu, *dy, wef::transpose(m_W));

    // gradient wrt weights sum everything aside from the last two axes. 
    m_dw = wef::matmul_GPU(gpu, wef::transpose(m_X), *dy);

    while (m_dw.m_rank > m_W.m_rank)
        m_dw = wef::reducesum(m_dw, /*axis*/0, /*keepkims*/false);

    m_W = wef::elemwise_GPU(gpu, m_W, m_dw * lr / dy->m_shape[0], /*operation=subtract=*/1);
    // m_W -= m_dw * lr / dy->m_shape[0];

    if (m_use_bias) 
    {
        // gradient wrt bias sum everything aside from the last axis
        m_db = *dy;
        while (m_db.m_rank > m_B.m_rank)
            m_db = wef::reducesum(m_db, /*axis*/0, /*keepkims*/false);
        m_db = wef::reducesum(m_db, 0, /*keepkims*/true); // because bias is shape = [1, bias]

        m_B = wef::elemwise_GPU(gpu, m_B, m_db * lr / dy->m_shape[0], 1);
        // m_B -= m_db * lr / dy->m_shape[0];
    }

    return &m_dx;
}

Tensor* Conv2D_GPU::forward_pass(const Tensor* px, const bool training, void* gpu) 
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
    if (training) std::memcpy(m_X.m_tensor, px->m_tensor, m_X.m_size * sizeof(float));

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

    struct PC
    {
        uint32_t inH, inW, inC;
        uint32_t kH, kW, outC;
        uint32_t outH, outW;
        uint32_t batch;
        uint32_t biasOffset;
    } push_constant;

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

    uint32_t gx = UseGPU::ceilDiv(push_constant.outW * push_constant.outC, WGX);
    uint32_t gy = UseGPU::ceilDiv(push_constant.outH, WGY);
    uint32_t gz = UseGPU::ceilDiv(push_constant.batch, WGZ);

    VkDeviceSize sizeA = sizeof(float) * px->m_size;
    VkDeviceSize sizeB = sizeof(float) * m_WB_size;
    VkDeviceSize sizeC = sizeof(float) * m_out.m_size;

    const char* spv_path = "../shaders/binaries/conv2d_f.spv";
    ((UseGPU*)gpu)->program({sizeA, sizeB}, {sizeC}, {px->m_tensor, m_WB.get()}, {m_out.m_tensor}, spv_path, &push_constant, sizeof(push_constant), gx, gy, gz);

    return &m_out;
}

Tensor* Conv2D_GPU::backward_pass(const Tensor* dy, const float lr, void* gpu) 
{   
    float* m_dx_ptr = m_dx.m_tensor;
    float* m_dw_ptr = m_dw.m_tensor;

    std::memset(m_dx_ptr, 0, (m_dx.m_size) * sizeof(float)); // zero fill
    std::memset(m_dw_ptr, 0, (m_dw.m_size) * sizeof(float)); // zero fill

    // gpu computation
    struct PC
    {
        uint32_t inH, inW, inC;
        uint32_t kH, kW, outC;
        uint32_t outH, outW;
        uint32_t batch;
    } push_constant;

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

    uint32_t gx = UseGPU::ceilDiv(m_dx.m_shape[0], WGX);
    uint32_t gy = UseGPU::ceilDiv(m_dx.m_shape[2] * m_dx.m_shape[3], WGY);
    uint32_t gz = UseGPU::ceilDiv(m_dx.m_shape[1], WGZ);
    
    VkDeviceSize sizeB = sizeof(float) * m_W.m_size;
    VkDeviceSize sizeC = sizeof(float) * dy->m_size;
    VkDeviceSize sizeA = sizeof(float) * m_dx.m_size;

    const char* spv_path = "../shaders/binaries/conv2d_b_dw.spv";
    ((UseGPU*)gpu)->program({sizeB, sizeC}, {/*output=*/sizeA}, {m_W.m_tensor, dy->m_tensor}, {/*output=*/m_dx.m_tensor}, spv_path, &push_constant, sizeof(push_constant), gx, gy, gz);

    gx = UseGPU::ceilDiv(m_dw.m_shape[3], WGX);
    gy = UseGPU::ceilDiv(m_dw.m_shape[2], WGY);
    gz = UseGPU::ceilDiv(m_dw.m_shape[0] * m_dw.m_shape[1], WGZ);

    sizeC = sizeof(float) * dy->m_size;
    sizeA = sizeof(float) * m_X.m_size;
    sizeB = sizeof(float) * m_dw.m_size;
    
    spv_path = "../shaders/binaries/conv2d_b_dx.spv";
    ((UseGPU*)gpu)->program({sizeC, sizeA}, {/*output=*/sizeB}, {dy->m_tensor, m_X.m_tensor}, {/*output=*/m_dw.m_tensor}, spv_path, &push_constant, sizeof(push_constant), gx, gy, gz);

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

    struct PC
    {
        uint32_t inH, inW, inC;
        uint32_t kH, kW, outC;
        uint32_t outH, outW;
        uint32_t batch;
        bool training;
    } push_constant;

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

    uint32_t gx = UseGPU::ceilDiv(push_constant.outW * push_constant.outC, WGX);
    uint32_t gy = UseGPU::ceilDiv(push_constant.outH, WGY);
    uint32_t gz = UseGPU::ceilDiv(push_constant.batch, WGZ);

    VkDeviceSize sizePx = sizeof(float) * px->m_size;
    VkDeviceSize sizeOut = sizeof(float) * m_out.m_size;

    const char* spv_path = "../shaders/binaries/MaxPool2D_f.spv";
    ((UseGPU*)gpu)->program({sizePx}, {m_argmax_len * sizeof(uint32_t), sizeOut}, {px->m_tensor}, {argmax.get(), m_out.m_tensor}, spv_path, &push_constant, sizeof(push_constant), gx, gy, gz);


    return &m_out;
}

Tensor* MaxPool2D_GPU::backward_pass(const Tensor* dy, const float lr, void* gpu) 
{
    std::memset(m_dx.m_tensor, 0, (m_dx.m_size) * sizeof(float));  // zero fill
    
    struct PC
    {
        uint32_t inH, inW, inC;
        uint32_t outC;
        uint32_t outH, outW;
        uint32_t batch;
    } push_constant;

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

    uint32_t gx = UseGPU::ceilDiv(push_constant.outW * push_constant.outC, WGX);
    uint32_t gy = UseGPU::ceilDiv(push_constant.outH, WGY);
    uint32_t gz = UseGPU::ceilDiv(push_constant.batch, WGZ);

    VkDeviceSize sizedy = sizeof(float) * dy->m_size;
    VkDeviceSize sizem_dx = sizeof(float) * m_dx.m_size;

    const char* spv_path = "../shaders/binaries/MaxPool2D_b.spv";
    ((UseGPU*)gpu)->program({m_argmax_len * sizeof(uint32_t), sizedy}, {sizem_dx}, {argmax.get(), dy->m_tensor}, {m_dx.m_tensor}, spv_path, &push_constant, sizeof(push_constant), gx, gy, gz);

    return &m_dx;
}
