#include "../include/useGPU.h"
#include "../include/layers.h"

Tensor* Linear_GPU::forward_pass(const Tensor& px, const bool training, void* gpu) 
{
    if (!init) 
    {   
        // initially initilize the shape of X later just copy the tensors
        X = Tensor(px);
        dist = std::normal_distribution<float>(0.0f, std::sqrt( 2.0f / (px.m_shape[px.m_rank-1])));

        size_t w_shape[2] = {px.m_shape[px.m_rank-1], units};
        size_t b_shape[2] = {1, units};
        W = Tensor::create(w_shape, 2);
        B = Tensor::create(b_shape, 2);

        float* B_ptr = B.m_tensor;
        std::fill_n(B.m_tensor, B.m_size, 0.0f); // zero fill

        float* pm = W.m_tensor;
        for (size_t i = 0; i < size_t(px.m_shape[px.m_rank-1]) * units; i++) pm[i] = dist(g);

        m_num_param = W.m_size + (m_use_bias ? B.m_size : 0);

        m_out_rank = px.m_rank;
        m_out_shape = std::make_unique<size_t[]>(m_out_rank);
        std::memcpy(m_out_shape.get(), px.m_shape, m_out_rank * sizeof(size_t));
        // TODO: CATCH < 1 RANK
        m_out_shape[m_out_rank - 1] = units;

        init = true;
    }
    else
    {
        // if trying to use (reuse) the layer on a different tensor
        if (W.m_shape[W.m_rank-2] != px.m_shape[px.m_rank-1]) throw std::invalid_argument("cannot reuse layer");
    }
    
    // copy px into X
    if (training) std::memcpy(X.m_tensor, px.m_tensor, X.m_size * sizeof(float));
    
    if (m_use_bias) out = wef::matmul_GPU(gpu, px, W) + B;
    else out = wef::matmul_GPU(gpu, px, W);
    return &out;

}

Tensor* Linear_GPU::backward_pass(const Tensor& dy, const float lr, void* gpu) 
{
    // gradient wrt the layer below
    dx = wef::matmul_GPU(gpu, dy, wef::transpose(W));

    // gradient wrt weights sum everything aside from the last two axes. 
    // CATCH rank < 2?????
    dw = wef::matmul_GPU(gpu, wef::transpose(X), dy);
    for (size_t i = 0; i < dw.m_rank - 2; i++) dw = wef::reducesum(dw, i);

    W = wef::elemwise_GPU(gpu, W, dw * lr / dy.m_shape[0], /*operation=subtract=*/1);
    // W -= dw * lr / dy.m_shape[0];

    if (m_use_bias) 
    {
        // gradient wrt bias sum everything aside from the last axis
        db = dy;
        for (size_t i = 0; i < db.m_rank - 1; i++) db = wef::reducesum(db, i);
        B = wef::elemwise_GPU(gpu, B, db * lr / dy.m_shape[0], 1);
        // B-= db * lr / dy.m_shape[0];
    }

    return &dx;
}

Tensor* Conv2D_GPU::forward_pass(const Tensor& px, const bool training, void* gpu) 
{
    if (!init) 
    {   
        // initially initilize the shape of X later just copy the tensors
        X = Tensor(px);

        // h, w, c, units
        height = px.m_shape[1]; width = px.m_shape[2]; ch = px.m_shape[3];
        dist = std::normal_distribution<float>(0.0f, std::sqrt( 2.0f / (w_height * w_width * ch))); 
        // for above now we have (2/fan_in = hwc)^0.5 good for relu we can use fan_out for tanh... which is hwu

        size_t w_shape[4] = {w_height, w_width, ch, units};
        W = Tensor::create(w_shape, 4);

        size_t B_shape[4] = {1, 1, 1, units};
        B = Tensor::create(B_shape, 4);
        std::fill_n(B.m_tensor, B.m_size, 0.0f);

        float* pm = W.m_tensor;
        for (size_t i = 0; i < W.m_size; i++) pm[i] = dist(g);

        m_out_rank = px.m_rank; // this is 4, its always 4
        m_out_shape = std::make_unique<size_t[]>(m_out_rank); // heap allocation is not the best but we only do this once pre layer so its whatever
        m_out_shape[1] = height - w_height + 1;
        m_out_shape[2] = width - w_width + 1;
        m_out_shape[3] = units;

        // gradients wrt weights and biases
        dx = Tensor(px);

        // gradient wrt weights
        dw = Tensor(W);
        db = Tensor(B);

        // weight + bias buffer
        WB_size = W.m_size + (m_use_bias ? units : 0);
        WB = std::make_unique<float[]>(WB_size);

        m_num_param = W.m_size + (m_use_bias ? B.m_size : 0);
        
        init = true;
    }
    else
    {
        // if trying to use (reuse) the layer on a different tensor
        if (px.m_shape[1] != height || 
            px.m_shape[2] != width ||
            px.m_shape[3] != ch) throw std::invalid_argument("cannot reuse layer");
    }

    // copy px into X
    if (training) std::memcpy(X.m_tensor, px.m_tensor, X.m_size * sizeof(float));

    m_out_shape[0] = px.m_shape[0]; // flexable batch 
    out = Tensor::create(m_out_shape.get(), 4);
    std::memset(out.m_tensor, 0, (out.m_size) * sizeof(float));

    std::memcpy(WB.get(), W.m_tensor, W.m_size * sizeof(float));
    uint32_t biasOffset = 0;
    if (m_use_bias)
    {
        biasOffset = W.m_size;
        std::memcpy(WB.get() + W.m_size, B.m_tensor, units * sizeof(float));
    }

    struct PC
    {
        uint32_t inH, inW, inC;
        uint32_t kH, kW, outC;
        uint32_t outH, outW;
        uint32_t batch;
        uint32_t biasOffset;
    } push_constant;

    push_constant.inH = height;
    push_constant.inW = width;
    push_constant.inC = ch;
    push_constant.kH = w_height;
    push_constant.kW = w_width;
    push_constant.outC = m_out_shape[3];
    push_constant.outH = m_out_shape[1];
    push_constant.outW = m_out_shape[2];
    push_constant.batch = m_out_shape[0];
    push_constant.biasOffset = biasOffset;
    
    const uint32_t WGX = 16;
    const uint32_t WGY = 16;
    const uint32_t WGZ = 1;

    uint32_t gx = useGPU::ceilDiv(push_constant.outW * push_constant.outC, WGX);
    uint32_t gy = useGPU::ceilDiv(push_constant.outH, WGY);
    uint32_t gz = useGPU::ceilDiv(push_constant.batch, WGZ);

    VkDeviceSize sizeA = sizeof(float) * px.m_size;
    VkDeviceSize sizeB = sizeof(float) * WB_size;
    VkDeviceSize sizeC = sizeof(float) * out.m_size;

    const char* spvPath = "shaders/binaries/conv2d_f.spv";
    ((useGPU*)gpu)->program({sizeA, sizeB}, {sizeC}, {px.m_tensor, WB.get()}, {out.m_tensor}, spvPath, &push_constant, sizeof(push_constant), gx, gy, gz);

    return &out;
}

Tensor* Conv2D_GPU::backward_pass(const Tensor& dy, const float lr, void* gpu) 
{   
    float* dx_ptr = dx.m_tensor;
    float* dw_ptr = dw.m_tensor;

    std::memset(dx_ptr, 0, (dx.m_size) * sizeof(float)); // zero fill
    std::memset(dw_ptr, 0, (dw.m_size) * sizeof(float)); // zero fill

    // gpu computation
    struct PC
    {
        uint32_t inH, inW, inC;
        uint32_t kH, kW, outC;
        uint32_t outH, outW;
        uint32_t batch;
    } push_constant;

    push_constant.inH = height;
    push_constant.inW = width;
    push_constant.inC = ch;
    push_constant.kH = w_height;
    push_constant.kW = w_width;
    push_constant.outC = units;
    push_constant.outH = dy.m_shape[1];
    push_constant.outW = dy.m_shape[2];
    push_constant.batch = dy.m_shape[0];

    const uint32_t WGX = 16;
    const uint32_t WGY = 16;
    const uint32_t WGZ = 1;

    uint32_t gx = useGPU::ceilDiv(dx.m_shape[0], WGX);
    uint32_t gy = useGPU::ceilDiv(dx.m_shape[2] * dx.m_shape[3], WGY);
    uint32_t gz = useGPU::ceilDiv(dx.m_shape[1], WGZ);
    
    VkDeviceSize sizeB = sizeof(float) * W.m_size;
    VkDeviceSize sizeC = sizeof(float) * dy.m_size;
    VkDeviceSize sizeA = sizeof(float) * dx.m_size;

    const char* spvPath = "shaders/binaries/conv2d_b_dx.spv";
    ((useGPU*)gpu)->program({sizeB, sizeC}, {/*output=*/sizeA}, {W.m_tensor, dy.m_tensor}, {/*output=*/dx.m_tensor}, spvPath, &push_constant, sizeof(push_constant), gx, gy, gz);

    gx = useGPU::ceilDiv(dw.m_shape[3], WGX);
    gy = useGPU::ceilDiv(dw.m_shape[2], WGY);
    gz = useGPU::ceilDiv(dw.m_shape[0] * dw.m_shape[1], WGZ);

    sizeC = sizeof(float) * dy.m_size;
    sizeA = sizeof(float) * X.m_size;
    sizeB = sizeof(float) * dw.m_size;
    
    spvPath ="shaders/binaries/conv2d_b_dw.spv";
    ((useGPU*)gpu)->program({sizeC, sizeA}, {/*output=*/sizeB}, {dy.m_tensor, X.m_tensor}, {/*output=*/dw.m_tensor}, spvPath, &push_constant, sizeof(push_constant), gx, gy, gz);

    float* pm_w = W.m_tensor;
    float* pm_dw = dw.m_tensor;
    // divide lr by batch size
    W = wef::elemwise_GPU(gpu, W, dw * lr / dy.m_shape[0], 1); // or
    // W -= dw * lr / dy.m_shape[0];

    if (m_use_bias)
    {
        db = dy;
        for (size_t i = 0; i < db.m_rank - 1; i++)
            db = wef::reducesum(db, i);
        B = wef::elemwise_GPU(gpu, B, db * lr / dy.m_shape[0], 1);
        // B -= db * lr / dy.m_shape[0];
    }

    return &dx;
}

Tensor* MaxPool2D_GPU::forward_pass(const Tensor& px, const bool training, void* gpu) 
{
    if (!init)
    {   
        // initially initilize the shape of X later just copy the tensors
        X = Tensor(px);
        // h, w, c, units
        height = px.m_shape[1]; width = px.m_shape[2]; ch = px.m_shape[3];

        size_t ax1 = (height + (height%k_height)) / k_height;
        size_t ax2 = (width + (width%k_width)) / k_width;

        m_argmax_len = 4 * px.m_shape[0] * ax1 * ax2 * ch;
        
        // this get the argmax in a nested for loop (2D) I made it flat for speed
        argmax = std::make_unique<uint32_t[]>(m_argmax_len);

        m_out_rank = px.m_rank; // this is 4, its always 4
        m_out_shape = std::make_unique<size_t[]>(m_out_rank); // heap allocation is not the best but we only do this once pre layer so its whatever
        m_out_shape[1] = ax1;
        m_out_shape[2] = ax2;
        m_out_shape[3] = ch;
        
        // dx is gradient wrt the layer below
        dx = Tensor(px);

        init = true;
    }
    else
    {
        // if trying to use (reuse) the layer on a different tensor
        if (px.m_shape[1] != height || 
            px.m_shape[2] != width ||
            px.m_shape[3] != ch) throw std::invalid_argument("cannot reuse layer");
    }

    // copy px into X
    if (training) std::memcpy(X.m_tensor, px.m_tensor, X.m_size * sizeof(float)); // TODO : is X even used in back prop?

    // batch is flexable
    m_out_shape[0] = px.m_shape[0];
    out = Tensor::create(m_out_shape.get(), 4);

    struct PC
    {
        uint32_t inH, inW, inC;
        uint32_t kH, kW, outC;
        uint32_t outH, outW;
        uint32_t batch;
        bool training;
    } push_constant;

    push_constant.inH = height;
    push_constant.inW = width;
    push_constant.inC = ch;
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

    uint32_t gx = useGPU::ceilDiv(push_constant.outW * push_constant.outC, WGX);
    uint32_t gy = useGPU::ceilDiv(push_constant.outH, WGY);
    uint32_t gz = useGPU::ceilDiv(push_constant.batch, WGZ);

    VkDeviceSize sizePx = sizeof(float) * px.m_size;
    VkDeviceSize sizeOut = sizeof(float) * out.m_size;

    const char* spvPath = "shaders/binaries/MaxPool2D_f.spv";
    ((useGPU*)gpu)->program({sizePx}, {m_argmax_len * sizeof(uint32_t), sizeOut}, {px.m_tensor}, {argmax.get(), out.m_tensor}, spvPath, &push_constant, sizeof(push_constant), gx, gy, gz);


    return &out;
}

Tensor* MaxPool2D_GPU::backward_pass(const Tensor& dy, const float lr, void* gpu) 
{
    std::memset(dx.m_tensor, 0, (dx.m_size) * sizeof(float));  // zero fill
    size_t ind = 0;
    size_t i1[4];

    struct PC
    {
        uint32_t inH, inW, inC;
        uint32_t outC;
        uint32_t outH, outW;
        uint32_t batch;
    } push_constant;

    push_constant.inH = height;
    push_constant.inW = width;
    push_constant.inC = ch;
    push_constant.outC = m_out_shape[3];
    push_constant.outH = m_out_shape[1];
    push_constant.outW = m_out_shape[2];
    push_constant.batch = m_out_shape[0];

    const uint32_t WGX = 16;
    const uint32_t WGY = 16;
    const uint32_t WGZ = 1;

    uint32_t gx = useGPU::ceilDiv(push_constant.outW * push_constant.outC, WGX);
    uint32_t gy = useGPU::ceilDiv(push_constant.outH, WGY);
    uint32_t gz = useGPU::ceilDiv(push_constant.batch, WGZ);

    VkDeviceSize sizedy = sizeof(float) * dy.m_size;
    VkDeviceSize sizedx = sizeof(float) * dx.m_size;

    const char* spvPath = "shaders/binaries/MaxPool2D_b.spv";
    ((useGPU*)gpu)->program({m_argmax_len * sizeof(uint32_t), sizedy}, {sizedx}, {argmax.get(), dy.m_tensor}, {dx.m_tensor}, spvPath, &push_constant, sizeof(push_constant), gx, gy, gz);

    return &dx;
}
