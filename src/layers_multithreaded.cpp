
#include <iostream>
#include <random>
#include "../include/layers.h"

Tensor* Conv2D_Fast::forward_pass(const Tensor* px, const bool training, void*) 
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
    
    float* m_out_ptr = m_out.m_tensor;
    float* px_ptr = px->m_tensor;
    float* m_W_ptr = m_W.m_tensor;
    float* m_B_ptr = m_B.m_tensor;

    std::memset(m_out_ptr, 0, (m_out.m_size) * sizeof(float));

    /*
    There is a lot of math below but the idea is to do the cov kernel math (m_W * Input) and expand 
    this to the m_units dimension by repeating the index of Input as before.

    Here what I do is index m_out m_out and m_W in contigous manner to make it faster and for Input
    I jump everytime we hit the width of the weight to the second row and use some math to do that.
    */

    size_t m_out_wo_m_units = m_out.m_size / m_units;
    size_t skip_w_help = m_ch * (m_k_width - 1);
    size_t bi_help = m_out_wo_m_units / m_out.m_shape[0];
    size_t skip_h_help = (m_k_height - 1) * px->m_shape[px->m_rank-2]*px->m_shape[px->m_rank-1];
    size_t offset = m_width - m_k_width;
    size_t id_help = m_k_width * m_ch;

    // multi thread additions
    size_t avaliable_threads = std::thread::hardware_concurrency(); // may be 0
    size_t n_threads = std::min<size_t>( m_out_wo_m_units,  avaliable_threads > 0 ? avaliable_threads : 1 );
    
    const size_t stride = m_out_wo_m_units / n_threads;
    const size_t rem = m_out_wo_m_units % n_threads;

    // spin up
    std::thread* threads = new std::thread[n_threads];
    for (size_t th = 0; th < n_threads; th++)
    {
        size_t temp = (th < n_threads - 1) ? stride : stride + rem;
        threads[th] = std::thread(

            // we dont want to capture everything in scope !
            [
                th, stride, temp, skip_w_help, bi_help, skip_h_help, px_ptr, offset, id_help,
                m_out_ptr, m_W_ptr, m_B_ptr
            ]
            (size_t m_Wsize, size_t m_outrow, size_t m_units, size_t ch, bool m_use_bias)
            {
                for (size_t m_out_i = th * stride; m_out_i < (th * stride) + temp; m_out_i++)
                {
                    size_t skip_w = skip_w_help * (m_out_i / m_outrow);
                    size_t bi = m_out_i / bi_help;
                    size_t skip_h = bi * skip_h_help;

                    for (size_t w_i = 0; w_i < m_Wsize / m_units; w_i++)
                    {
                        float temp_px = px_ptr[
                            ch * m_out_i + skip_w + skip_h
                            + 
                            w_i + ch * offset * (w_i / id_help)
                        ];

                        for (size_t u_i = 0; u_i < m_units; u_i++)
                            // TODO : data races
                            m_out_ptr[m_out_i * m_units + u_i] += temp_px * m_W_ptr[w_i * m_units + u_i];
                    }
                    if (m_use_bias)
                        for (size_t u_i = 0; u_i < m_units; u_i++)
                            m_out_ptr[m_out_i * m_units + u_i] += m_B_ptr[u_i];
                }
            },
            
            // pass these are parameters cause we dont want to copy the entire tensor
            m_W.m_size, m_out.m_shape[m_out.m_rank-2], m_units, m_ch, m_use_bias);
    }

    // free
    for (size_t i = 0; i < n_threads; i++) threads[i].join();
    
    // clean up
    delete[] threads;
    
    return &m_out;
}

Tensor* Conv2D_Fast::backward_pass(const Tensor* dy, const float lr, void*) 
{   
    float* m_dx_ptr = m_dx.m_tensor;
    float* m_dw_ptr = m_dw.m_tensor;

    std::memset(m_dx_ptr, 0, (m_dx.m_size) * sizeof(float)); // zero fill
    std::memset(m_dw_ptr, 0, (m_dw.m_size) * sizeof(float)); // zero fill

    float* dy_ptr = dy->m_tensor;
    float* m_W_ptr = m_W.m_tensor;
    float* m_X_ptr = m_X.m_tensor;

    size_t m_out_wo_m_units = dy->m_size / m_units;
    size_t skip_w_help = m_ch * (m_k_width - 1);
    size_t bi_help = m_out_wo_m_units / dy->m_shape[0];
    size_t skip_h_help = (m_k_height - 1) * m_X.m_shape[m_X.m_rank-2] * m_X.m_shape[m_X.m_rank-1];
    size_t offset = m_width - m_k_width;
    size_t id_help = m_k_width * m_ch;

    // multi thread additions
    size_t avaliable_threads = std::thread::hardware_concurrency(); // may be 0
    size_t n_threads = std::min<size_t>( m_out_wo_m_units,  avaliable_threads > 0 ? avaliable_threads : 1 );

    // make temp variables for each thread cause we should not do += at the same index in different threads
    std::unique_ptr<Tensor[]> m_dx_accum_pre_thread = std::make_unique<Tensor[]>(n_threads);
    std::unique_ptr<Tensor[]> m_dw_accum_pre_thread = std::make_unique<Tensor[]>(n_threads);
    for (size_t i = 0; i < n_threads; i++)
    {
        m_dx_accum_pre_thread[i] = m_dx; m_dw_accum_pre_thread[i] = m_dw;
    }
    
    const size_t stride = m_out_wo_m_units / n_threads;
    const size_t rem = m_out_wo_m_units % n_threads;

    // spin up
    std::thread* threads = new std::thread[n_threads];
    for (size_t th = 0; th < n_threads; th++)
    {
        size_t temp = (th < n_threads - 1) ? stride : stride + rem;
        threads[th] = std::thread(
            // we dont want to capture everything in scope !
            [
                th, stride, temp, skip_w_help, bi_help, skip_h_help, id_help, offset, 
                dy_ptr, m_X_ptr, m_W_ptr
            ]
            (size_t m_Wsize, size_t dyrow, size_t m_units, size_t ch, float* m_dx_i, float* m_dw_i)
            {
                for (size_t dy_i = th * stride; dy_i < (th * stride) + temp; dy_i++)
                {
                    size_t skip_w = skip_w_help * (dy_i / dyrow);
                    size_t bi = dy_i / bi_help;
                    size_t skip_h = bi * skip_h_help;

                    for (size_t w_i = 0; w_i < m_Wsize / m_units; w_i++)
                    {
                        size_t id1 = 
                            ch * dy_i + skip_w + skip_h
                            + 
                            w_i + ch*offset * (w_i / id_help);

                        for (size_t u_i = 0; u_i < m_units; u_i++)
                        {
                            float grad = dy_ptr[dy_i * m_units + u_i];
                            // data races TODO
                            m_dx_i[id1] += grad * m_W_ptr[w_i * m_units + u_i];
                            m_dw_i[w_i * m_units + u_i] += grad * m_X_ptr[id1];
                        }
                    }
                }
            },
        // pass these are parameters cause we dont want to copy the entire tensor
        m_W.m_size, dy->m_shape[dy->m_rank-2], m_units, m_ch, m_dx_accum_pre_thread[th].m_tensor, m_dw_accum_pre_thread[th].m_tensor);
    }

    // free
    for (size_t i = 0; i < n_threads; i++) threads[i].join();

    // clean up
    delete[] threads;

    // aggrigate the temp variables from each thread
    for (size_t i = 0; i < n_threads; i++)
    {
        for (size_t v = 0; v < m_dw.m_size; v++) m_dw_ptr[v] += m_dw_accum_pre_thread[i].m_tensor[v];
        for (size_t v = 0; v < m_dx.m_size; v++) m_dx_ptr[v] += m_dx_accum_pre_thread[i].m_tensor[v];
    }

    // divide lr by batch size
    m_W -= m_dw * lr /dy->m_shape[0];

    if (m_use_bias)
    {
        m_db = *dy;
        for (size_t i = 0; i < m_db.m_rank - 1; i++)
            m_db = wef::reducesum(m_db, i, /*keepkims*/true);
        m_B -= m_db * lr / dy->m_shape[0];
    }

    return &m_dx;
}

Tensor* Linear_Fast::forward_pass(const Tensor* px, const bool training, void*) 
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

        float* m_B_ptr = m_B.m_tensor;
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

    if (m_use_bias) m_out = wef::matmul(*px, m_W, true) + m_B;
    else m_out = wef::matmul(*px, m_W, true);
    return &m_out;

}

Tensor* Linear_Fast::backward_pass(const Tensor* dy, const float lr, void*) 
{
    // gradient wrt the layer below
    m_dx = wef::matmul(*dy, wef::transpose(m_W), true);

    // gradient wrt weights sum everything aside from the last two axes. 
    m_dw = wef::matmul(wef::transpose(m_X), *dy, /*threads=*/true);

    while (m_dw.m_rank > m_W.m_rank)
        m_dw = wef::reducesum(m_dw, /*axis*/0, /*keepkims*/false);

    m_W -= m_dw * lr / dy->m_shape[0];

    if (m_use_bias) 
    {
        // gradient wrt bias sum everything aside from the last axis
        m_db = *dy;
        while (m_db.m_rank > m_B.m_rank)
            m_db = wef::reducesum(m_db, /*axis*/0, /*keepkims*/false);
        m_db = wef::reducesum(m_db, 0, /*keepkims*/true); // because bias is shape = [1, bias]
        m_B -= m_db * lr / dy->m_shape[0];
    }

    return &m_dx;
}
