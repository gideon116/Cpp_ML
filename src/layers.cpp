#include <iostream>
#include <random>
#include "layers.h"

Tensor* Linear::forward_pass(const Tensor* px, const bool training, void*) 
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


    if (m_use_bias)
        m_out = wef::matmul(*px, m_W, true) + m_B;
    else
        m_out = wef::matmul(*px, m_W, true);
    return &m_out;

    // if (m_use_bias) return &wef::matmul(px, m_W) + m_B;
    // return &wef::matmul(px, m_W);
}

Tensor* Linear::backward_pass(const Tensor* dy, const float lr, void*) 
{
    // gradient wrt the layer below
    m_dx = wef::matmul(*dy, wef::transpose(m_W));

    // gradient wrt weights sum everything aside from the last two axes. 
    // CATCH rank < 2?????
    m_dw = wef::matmul(wef::transpose(m_X), *dy);

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
        // or
        // for (size_t i = 0; i < m_db.m_rank - 1; i++) m_db = wef::reducesum(m_db, i);
        m_B -= m_db * lr / dy->m_shape[0];
    }

    return &m_dx;
}

Tensor* Conv2D::forward_pass(const Tensor* px, const bool training, void*) 
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

        // gradient wrt the layer below
        m_dx = Tensor(*px);

        // gradients wrt weights and biases
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
    I jump everytime we hit the m_width of the weight to the second row and use some math to do that.
    */

    size_t m_out_wo_m_units = m_out.m_size / m_units;
    size_t skip_w_help = m_ch * (m_k_width - 1);
    size_t bi_help = m_out_wo_m_units / m_out.m_shape[0];
    size_t skip_h_help = (m_k_height - 1) * px->m_shape[px->m_rank-2] * px->m_shape[px->m_rank-1];
    size_t offset = m_width - m_k_width;
    size_t id_help = m_k_width * m_ch;

    #pragma omp parallel for schedule(static)
    for (size_t m_out_i = 0; m_out_i < m_out_wo_m_units; m_out_i++)
    {
        size_t skip_w = skip_w_help * (m_out_i / m_out.m_shape[m_out.m_rank-2]);
        size_t bi = m_out_i / bi_help;
        size_t skip_h = bi * skip_h_help;

        for (size_t w_i = 0; w_i < m_W.m_size / m_units; w_i++)
        {
            float temp_px = px_ptr[
                m_ch * m_out_i + skip_w + skip_h
                + 
                w_i + m_ch * offset * (w_i / id_help)
            ];

            for (size_t u_i = 0; u_i < m_units; u_i++)
                m_out_ptr[m_out_i * m_units + u_i] += temp_px * m_W_ptr[w_i * m_units + u_i];
        }
        if (m_use_bias)
            for (size_t u_i = 0; u_i < m_units; u_i++)
                m_out_ptr[m_out_i * m_units + u_i] += m_B_ptr[u_i];
    }

    return &m_out;
}

Tensor* Conv2D::backward_pass(const Tensor* dy, const float lr, void*) 
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

    #pragma omp parallel for schedule(static)
    for (size_t dy_i = 0; dy_i < m_out_wo_m_units; dy_i++)
    {
        size_t skip_w = skip_w_help * (dy_i / dy->m_shape[dy->m_rank-2]);
        size_t bi = dy_i / bi_help;
        size_t skip_h = bi * skip_h_help;

        for (size_t w_i = 0; w_i < m_W.m_size / m_units; w_i++)
        {
            size_t id1 = 
                m_ch * dy_i + skip_w + skip_h
                + 
                w_i + m_ch * offset * (w_i / id_help);

            for (size_t u_i = 0; u_i < m_units; u_i++)
            {
                float grad = dy_ptr[dy_i * m_units + u_i];
                m_dx_ptr[id1] += grad * m_W_ptr[w_i * m_units + u_i];
                m_dw_ptr[w_i * m_units + u_i] += grad * m_X_ptr[id1];
            }
        }
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

Tensor* Conv2D_legacy::forward_pass(const Tensor* px, const bool training, void*) 
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

        // gradient wrt the layer below
        m_dx = Tensor(*px);
        
        // gradients wrt weights and biases
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
    float* pm_out = m_out.m_tensor;
    float* pm_b = m_B.m_tensor;

    size_t ind = 0;
    size_t i1[4];
    size_t i2[4];

    #pragma omp parallel for collapse(4) schedule(static)
    for (size_t b1 = 0; b1 < m_out.m_shape[0]; b1++)
    {
        for (size_t h1 = 0; h1 < m_out.m_shape[1]; h1++)
        {
            for (size_t w1 = 0; w1 < m_out.m_shape[2]; w1++)
            {
                for (size_t oc = 0; oc < m_out.m_shape[3]; oc++)    
                {
                    float temp = 0.0f;
                    for (size_t h2 = 0; h2 < m_k_height; h2++)
                    {
                        for (size_t w2 = 0; w2 < m_k_width; w2++)
                        {
                            for (size_t c2 = 0; c2 < m_ch; c2++)
                            {
                                // doing this to ensure cstyle array for indexing
                                // not making a new array and just rewriting
                                i1[0] = b1; i1[1] = h2 + h1; i1[2] = w2 + w1; i1[3] = c2;
                                i2[0] = h2; i2[1] = w2; i2[2] = c2; i2[3] = oc; 

                                temp += (*px)[i1] * m_W[i2];
                            }
                        }
                    }
                    pm_out[ind++] = temp + (m_use_bias ? pm_b[oc] : 0.0); // TODO: m_check m_BIAS
                }
            }
        }
    }

    return &m_out;
}

Tensor* Conv2D_legacy::backward_pass(const Tensor* dy, const float lr, void*) 
{
    std::memset(m_dx.m_tensor, 0, (m_dx.m_size) * sizeof(float)); // zero fill
    std::memset(m_dw.m_tensor, 0, (m_dw.m_size) * sizeof(float)); // zero fill

    size_t ind = 0;
    size_t i1[4];
    size_t i2[4];

    #pragma omp parallel for collapse(4) schedule(static)
    for (size_t b1 = 0; b1 < dy->m_shape[0]; b1++)
    {
        for (size_t h1 = 0; h1 < dy->m_shape[1]; h1++)
        {
            for (size_t w1 = 0; w1 < dy->m_shape[2]; w1++)
            {
                for (size_t oc = 0; oc < dy->m_shape[3]; oc++)
                {
                    float grad = dy->m_tensor[ind++];
                    for (size_t h2 = 0; h2 < m_k_height; h2++)
                    {
                        for (size_t w2 = 0; w2 < m_k_width; w2++)
                        {
                            for (size_t c2 = 0; c2 < m_ch; c2++)
                            {
                                // doing this to ensure cstyle array for indexing
                                // not making a new array and just rewriting
                                i1[0] = b1; i1[1] = h2 + h1; i1[2] = w2 + w1; i1[3] = c2;
                                i2[0] = h2; i2[1] = w2; i2[2] = c2; i2[3] = oc; 

                                m_dx[i1] += grad * m_W[i2];
                                m_dw[i2] += grad * m_X[i1];
                            }
                        }
                    }
                }
            }
        }
    }
    
    float* pm_w = m_W.m_tensor;
    float* pm_m_dw = m_dw.m_tensor;
    // divide lr by batch size
    m_W -= m_dw * lr / dy->m_shape[0];

    if (m_use_bias)
    {
        m_db = *dy;
        for (size_t i = 0; i < m_db.m_rank - 1; i++) m_db = wef::reducesum(m_db, i, /*keepkims*/true);
        m_B -= m_db * lr / dy->m_shape[0];
    }

    return &m_dx;
}

Tensor* MaxPool2D::forward_pass(const Tensor* px, const bool training, void*) 
{
    if (!m_init)
    {   
        // initially initilize the shape of m_X later just copy the tensors
        m_X = Tensor(*px);
        // h, w, c, m_units
        m_height = px->m_shape[1]; m_width = px->m_shape[2]; m_ch = px->m_shape[3];

        size_t ax1 = (m_height + (m_height%k_height)) / k_height;
        size_t ax2 = (m_width + (m_width%k_width)) / k_width;

        size_t o_size = px->m_shape[0] * ax1 * ax2 * m_ch;
        
        // this get the argmax in a nested for loop (2D) I made it flat for speed
        argmax = std::make_unique<size_t[]>(o_size * 4);

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

    size_t ind = 0;
    size_t i1[4];

    #pragma omp parallel for collapse(4) schedule(static)
    for (size_t b1 = 0; b1 < m_out.m_shape[0]; b1++)
    {
        for (size_t h1 = 0; h1 < m_out.m_shape[1]; h1++)
        {
            for (size_t w1 = 0; w1 < m_out.m_shape[2]; w1++)
            {
                for (size_t c = 0; c < m_out.m_shape[3]; c++)
                {
                    float temp_val = -1e19f;
                    size_t temp_ind[4];
                    for (size_t h2 = h1 * k_height; h2 < h1 * k_height + k_height; h2++)
                    {
                        if (h2 >= m_height) break;
                        for (size_t w2 = w1 * k_width; w2 < w1 * k_width + k_width; w2++)
                        {
                            if (w2 >= m_width) break;

                            i1[0] = b1; i1[1] = h2; i1[2] = w2; i1[3] = c;
                            float val = (*px)[i1];

                            if (val > temp_val)
                            {
                                temp_val = val;
                                temp_ind[0] = b1;
                                temp_ind[1] = h2;
                                temp_ind[2] = w2;
                                temp_ind[3] = c;
                            }
                        }
                    }
                    m_out.m_tensor[ind] = temp_val;

                    // only populate argmax during training for speed, idt it affects values if we keep it tho
                    if (training)
                        for (size_t ii = 0; ii < 4; ii++)
                            argmax[ind * 4 + ii] = temp_ind[ii];
                    ind++;
                }
            }
        }
    }
    return &m_out;
}

Tensor* MaxPool2D::backward_pass(const Tensor* dy, const float lr, void*) 
{
    std::memset(m_dx.m_tensor, 0, (m_dx.m_size) * sizeof(float));  // zero fill
    size_t ind = 0;
    size_t i1[4];

    #pragma omp parallel for collapse(4) schedule(static)
    for (size_t b1 = 0; b1 < dy->m_shape[0]; b1++)
    {
        for (size_t h1 = 0; h1 < dy->m_shape[1]; h1++)
        {
            for (size_t w1 = 0; w1 < dy->m_shape[2]; w1++)
            {
                for (size_t c = 0; c < dy->m_shape[3]; c++)    
                {
                    i1[0] = argmax[ind * 4 + 0]; i1[1] = argmax[ind * 4 + 1]; 
                    i1[2] = argmax[ind * 4 + 2]; i1[3] = argmax[ind * 4 + 3];
                    m_dx[i1] = dy->m_tensor[ind];
                    ind++; 
                }
            }
        }
    }
    return &m_dx;
}

Tensor* ReduceSum::forward_pass(const Tensor* px, const bool training, void*) 
{ 

    if (!m_init) 
    {
        // initially initilize the shape of m_X later just copy the tensors
        m_X = Tensor(*px);

        if (m_ax >= px->m_rank)
            throw std::invalid_argument("axis m_outside shape");
        const size_t* shape = px->m_shape; // [b, h, w, c]

        m_out_rank = keepdims ? px->m_rank : px->m_rank - 1;
        m_out_shape = std::make_unique<size_t[]>(m_out_rank);  // [b, h, w] or  // [b, h, w, 1]
        
        // index 0 will be reassigned if ax > 0 so as to be flexable with the batch
        size_t j = 0;
        for (size_t i = 0; i < px->m_rank; i++)
        {
            if (i != m_ax) m_out_shape[j++] = shape[i]; 
            else if (keepdims) m_out_shape[j++] = 1;
        }

        m_dx = Tensor(*px);
        
        m_init = true;
    }

    // copy px into m_X
    if (training) std::memcpy(m_X.m_tensor, px->m_tensor, m_X.m_size * sizeof(float));

    // batch flexability only applies if ax != 0
    if (m_ax != 0)
        m_out_shape[0] = px->m_shape[0];

    m_out = Tensor::create(m_out_shape.get(), m_out_rank);

    const float* pm = px->m_tensor;
    float* pm_out = m_out.m_tensor;

    size_t eaa = 1; // everything after axis i.e. b, h w, axis, x1, x2 -> eaa = x1 * x2
    for (size_t i = m_ax + 1; i < px->m_rank; i++) eaa *= px->m_shape[i];

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < m_out.m_size; i++)
    {
        float temp = 0.0f;
        size_t mult = (i/eaa) * (1 - px->m_shape[m_ax]);

        for (size_t j = 0; j < px->m_shape[m_ax]; j++)
            temp += pm[i  + eaa * (j - mult)];

        pm_out[i] = temp;
    }

    return &m_out;
}

Tensor* ReduceSum::backward_pass(const Tensor* dy, float, void*) 
{
    if (!m_init)
        throw std::invalid_argument("layer not initilized");

    const float* pdy = dy->m_tensor;
    float* pm_dx = m_dx.m_tensor; // no need to zero fill cause we dont do +=

    size_t eaa = 1;
    for (size_t i = m_ax + 1; i < m_dx.m_rank; i++) eaa *= m_dx.m_shape[i];

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < dy->m_size; i++)
    {
        size_t mult = (i/eaa) * (1 - m_dx.m_shape[m_ax]) ;
        for (size_t j = 0; j < m_dx.m_shape[m_ax]; j++)
        {
            pm_dx[i  + eaa * (j - mult)] = pdy[i];
        }
    }
    return &m_dx;
}

Tensor* LayerNorm::forward_pass(const Tensor* px, const bool training, void*)
{
    if (!m_init) 
    {
        // initially initilize the shape of m_X later just copy the tensors
        m_X = Tensor(*px);

        m_ax_val = (float)(px->m_shape[m_axis]);
        std::unique_ptr<size_t[]> beta_shape = std::make_unique<size_t[]>(px->m_rank);
        std::unique_ptr<size_t[]> gamma_shape = std::make_unique<size_t[]>(px->m_rank);

        // fill with 1s
        std::fill_n(beta_shape.get(), px->m_rank, 1);
        std::fill_n(gamma_shape.get(), px->m_rank, 1);

        beta_shape[m_axis] = px->m_shape[m_axis];
        gamma_shape[m_axis] = px->m_shape[m_axis];

        m_beta = Tensor::create(beta_shape.get(), px->m_rank);
        m_gamma = Tensor::create(gamma_shape.get(), px->m_rank);

        // initilize beta and gamma
        memset(m_beta.m_tensor, 0, m_beta.m_size * sizeof(float));
        memset(m_gamma.m_tensor, 0, m_gamma.m_size * sizeof(float));
        m_gamma += 1;

        m_num_param = m_beta.m_size + m_gamma.m_size;

        m_out_rank = px->m_rank;
        m_out_shape = std::make_unique<size_t[]>(m_out_rank);
        std::memcpy(m_out_shape.get(), px->m_shape, m_out_rank * sizeof(size_t));
        
        m_init = true;
    }
    if (px->m_shape[m_axis] != (size_t)m_ax_val)
        throw std::invalid_argument("cannot reuse layer [LayerNorm]");

    // copy px into m_X
    if (training)
        std::memcpy(m_X.m_tensor, px->m_tensor, m_X.m_size * sizeof(float));

    // following m_Ba et al. 2016
    m_mu = wef::reducesum(*px, /*axis=*/m_axis, /*keepkims*/true) / m_ax_val;
    m_x_mu = (*px) - m_mu;
    m_var = wef::reducesum(m_x_mu * m_x_mu, /*axis=*/m_axis, /*keepkims*/true) / m_ax_val;
    m_inv_std = wef::pow(m_var + m_eps, -0.5f);
    m_x_i_hat = m_x_mu * m_inv_std;
    m_y_i = m_x_i_hat * m_gamma + m_beta;

    return &m_y_i;
}

Tensor* LayerNorm::backward_pass(const Tensor* dy, const float lr, void*)
{
    if (!m_init)
        throw std::invalid_argument("layer not initilized");

    m_d_gamma = (*dy) * m_x_i_hat;
    m_d_beta = *dy;
    for (size_t i = 0; i < dy->m_rank; i++)
    {
        if (i != m_axis)
        {
            m_d_gamma = wef::reducesum(m_d_gamma, i, /*keepkims*/true);
            m_d_beta = wef::reducesum(m_d_beta, i, /*keepkims*/true);
        }
    }

    m_dx = m_inv_std * (1.0 / m_ax_val) * 
    (
        m_gamma * (*dy) * m_ax_val 
        - wef::reducesum(m_gamma * (*dy), m_axis, /*keepkims*/true)
        - m_x_i_hat * (wef::reducesum(m_gamma * (*dy) * m_x_i_hat, m_axis, /*keepkims*/true))
    );

    m_gamma -= m_d_gamma * lr / dy->m_shape[0];
    m_beta -= m_d_beta * lr / dy->m_shape[0];

    return &m_dx;
}

Tensor* Flatten::forward_pass(const Tensor* px, const bool training, void*) 
{
    if (!m_init)
    {
        m_dx = Tensor::create(px->m_shape, px->m_rank); // TODO : set shape only once, locked in after
        
        size_t flat = 1;
        for (size_t i = 1; i < px->m_rank; i++) flat *= px->m_shape[i];
        
        m_out_rank = 2; // TODO : hard code??
        m_out_shape = std::make_unique<size_t[]>(m_out_rank);
        m_out_shape[1] = flat;
        m_init = true;
    }
    else
    {
        // if trying to use (reuse) the layer on a different tensor
        if (training)
            if (m_dx.m_size != px->m_size) // TODO : better to check shape but this is faster
                throw std::invalid_argument("cannot reuse layer");
    }

    m_out_shape[0] = px->m_shape[0];
    m_out = Tensor::create(m_out_shape.get(), 2);
    memcpy(m_out.m_tensor, px->m_tensor, m_out.m_size * sizeof(float));

    return &m_out;
}

Tensor* Flatten::backward_pass(const Tensor* dy, float, void*) 
{
    memcpy(m_dx.m_tensor, dy->m_tensor, m_dx.m_size * sizeof(float));
    return &m_dx;
}

Tensor* ReLU::forward_pass(const Tensor* px, const bool training, void*)
{
    if (!m_init)
    {
        m_out_rank = px->m_rank;
        m_out_shape = std::make_unique<size_t[]>(m_out_rank);
        std::memcpy(m_out_shape.get(), px->m_shape, m_out_rank * sizeof(size_t));
        m_init= true;
    }

    m_X = wef::relu(*px);
    return &m_X;
}

Tensor* ReLU::backward_pass(const Tensor* dy, float, void*)
{
    m_dx = wef::d_relu(m_X) * (*dy);
    return &m_dx;
}

Tensor* Sigmoid::forward_pass(const Tensor* px, const bool training, void*) 
{ 
    if (!m_init)
    {
        m_out_rank = px->m_rank;
        m_out_shape = std::make_unique<size_t[]>(m_out_rank);
        std::memcpy(m_out_shape.get(), px->m_shape, m_out_rank * sizeof(size_t));
        m_init= true;
    }

    m_X = wef::sigmoid(*px);
    return &m_X;
}

Tensor* Sigmoid::backward_pass(const Tensor* dy, float, void*)
{
    m_dx = wef::d_sigmoid(m_X) * (*dy);
    return &m_dx;
}

Tensor* Conv2D_NR::forward_pass(const Tensor* px, const bool training, void*) 
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

        // gradient wrt the layer below
        m_dx = Tensor(*px);
        
        // gradients wrt weights and biases
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
    float* pm_out = m_out.m_tensor;
    float* pm_b = m_B.m_tensor;

    size_t ind = 0;
    size_t i1[4];
    size_t i2[4];

    #pragma omp parallel for collapse(4) schedule(static)
    for (size_t b1 = 0; b1 < m_out.m_shape[0]; b1++)
    {
        for (size_t h1 = 0; h1 < m_out.m_shape[1]; h1++)
        {
            for (size_t w1 = 0; w1 < m_out.m_shape[2]; w1++)
            {
                for (size_t oc = 0; oc < m_out.m_shape[3]; oc++)    
                {
                    float temp = 0.0f;
                    for (size_t h2 = 0; h2 < m_k_height; h2++)
                    {
                        for (size_t w2 = 0; w2 < m_k_width; w2++)
                        {
                            for (size_t c2 = 0; c2 < m_ch; c2++)
                            {
                                // doing this to ensure cstyle array for indexing
                                // not making a new array and just rewriting
                                i1[0] = b1; i1[1] = h2 + h1; i1[2] = w2 + w1; i1[3] = c2;
                                i2[0] = h2; i2[1] = w2; i2[2] = c2; i2[3] = oc; 

                                temp += (*px)[i1] * m_W[i2];
                            }
                        }
                    }
                    pm_out[ind++] = temp + (m_use_bias ? pm_b[oc] : 0.0); // TODO: m_check m_BIAS
                }
            }
        }
    }

    return &m_out;
}

Tensor* Conv2D_NR::backward_pass(const Tensor* dy, const float lr, void*) 
{
    std::memset(m_dx.m_tensor, 0, (m_dx.m_size) * sizeof(float)); // zero fill
    std::memset(m_dw.m_tensor, 0, (m_dw.m_size) * sizeof(float)); // zero fill

    #pragma omp parallel for collapse(4) schedule(static)
    for (size_t bin_hin = 0; bin_hin < m_dx.m_shape[0] * m_dx.m_shape[1]; bin_hin++)
    {
        size_t bin = bin_hin / m_dx.m_shape[1];
        size_t hin = bin_hin % m_dx.m_shape[1];
        
        for (size_t win_cin = 0; win_cin < m_dx.m_shape[2] * m_dx.m_shape[3]; win_cin++)  
        {
            size_t win = win_cin / m_dx.m_shape[3];
            size_t cin = win_cin % m_dx.m_shape[3];

            float temp = 0.0;
            for (size_t hk = 0; hk < m_k_height; hk++)            
            {
                if (hin < hk) continue;
                size_t ho = hin - hk; // move m_out

                for (size_t wk = 0; wk < m_k_width; wk++)                
                {
                    if (win < wk) continue;
                    size_t wo = win - wk; // move m_out

                    for (size_t co = 0; co < m_units; co++)
                    {
                        if (ho >= dy->m_shape[1] || wo >= dy->m_shape[2]) continue;
                        
                        size_t ik[4] = {hk, wk, cin, co};
                        size_t io[4] = {bin, ho, wo, co};

                        temp += (*dy)[io] * m_W[ik];
                    }
                }
            }
            size_t iin[4] = {bin, hin, win, cin};
            m_dx[iin] = temp;
        }
    }
    
    #pragma omp parallel for collapse(4) schedule(static)
    for (size_t hk_wk = 0; hk_wk < m_dw.m_shape[0] * m_dw.m_shape[1]; hk_wk++)
    {
        size_t hk = hk_wk / m_dw.m_shape[1];
        size_t wk = hk_wk % m_dw.m_shape[1];
    
        for (size_t cin_co = 0; cin_co < m_dw.m_shape[2] * m_dw.m_shape[3]; cin_co++)
        {
            size_t cin = cin_co / m_dw.m_shape[3];
            size_t co = cin_co % m_dw.m_shape[3];
            
            float temp = 0.0;
            for (size_t bin = 0; bin < dy->m_shape[0]; bin++)
            {
                for (size_t ho = 0; ho < dy->m_shape[1]; ho++)
                {
                    // I m_chose to do this rather than doing the inner 3 loops
                    // with m_dx.m_shape[...] b/c otherwise I'd have to add three 
                    // bounds m_checks like "if (win < wk) continue;"
                    size_t hin = ho + hk;

                    for (size_t wo = 0; wo < dy->m_shape[2]; wo++)  
                    {
                        
                        size_t win = wo + wk;
                        
                        size_t iin[4] = {bin, hin, win, cin};
                        size_t io[4] = {bin, ho, wo, co};

                        temp += (*dy)[io] * m_X[iin];
                    }
                }
            }
            size_t ik[4] = {hk, wk, cin, co};
            m_dw[ik] = temp;
            
        }
    }


    float* pm_w = m_W.m_tensor;
    float* pm_m_dw = m_dw.m_tensor;
    // divide lr by batch size
    m_W -= m_dw * lr / dy->m_shape[0];

    if (m_use_bias)
    {
        m_db = *dy;
        for (size_t i = 0; i < m_db.m_rank - 1; i++)
            m_db = wef::reducesum(m_db, i, /*keepkims*/true);
        m_B -= m_db * lr / dy->m_shape[0];
    }

    return &m_dx;
}

Tensor MHA::scaled_dot_product_attention(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor* mask, void* gpu)
{
    Tensor product;
    if (m_use_gpu)
        product = wef::matmul_GPU(gpu, q, wef::transpose(k));
    else
        product = wef::matmul(q, wef::transpose(k));

    m_keys_dim = (float)k.m_shape[k.m_rank-1];

    Tensor eij = product / sqrt(m_keys_dim);

    if (m_use_mask) // TODO or mask->tensor == nullptr
        eij = eij + (*mask * -1e9f); // TODO : add += with broadcast

    m_aij = wef::softmax(eij);
    Tensor z;
    if (m_use_gpu)
        z = wef::matmul_GPU(gpu, m_aij, v);
    else
        z = wef::matmul(m_aij, v);
     
    return z;
}

void MHA::split_heads(Tensor& x, size_t seq_len)
{ 
    size_t shape[4] = {m_batch, seq_len, m_num_heads, m_depth};
    x.reshape(shape, 4); // (batch_size, seq_len_q, d_model)

    size_t prem[4] = {0, 2, 1, 3};
    x = wef::transpose(x, prem);
}

Tensor* MHA::forward_pass(const Tensor* qkv_mask, const bool training, void* gpu)
{
    if (!m_init)
    {
        m_init = true;
    }
    else
    {
        // TODO : add shape m_check after init
    }

    m_q = qkv_mask[0];
    m_k = qkv_mask[1];
    m_v = qkv_mask[2];
    m_mask = m_use_mask ? qkv_mask[3] : Tensor();
    
    m_batch = m_q.m_shape[0];
    m_seq_len_q = m_q.m_shape[1];
    m_seq_len_k = m_k.m_shape[1];
    m_seq_len_v = m_v.m_shape[1];

    m_q = *m_wq->forward_pass(&m_q, training, gpu);  // (batch_size, seq_len, d_model)
    m_k = *m_wk->forward_pass(&m_k, training, gpu);  // (batch_size, seq_len, d_model)
    m_v = *m_wv->forward_pass(&m_v, training, gpu);  // (batch_size, seq_len, d_model)

    split_heads(m_q, m_seq_len_q);  // (batch_size, num_heads, seq_len_q, depth)
    split_heads(m_k, m_seq_len_k);  // (batch_size, num_heads, seq_len_k, depth)
    split_heads(m_v, m_seq_len_v);  // (batch_size, num_heads, seq_len_v, depth)

    Tensor attention = scaled_dot_product_attention(m_q, m_k, m_v, &m_mask, gpu);
    
    size_t prem[4] = {0, 2, 1, 3};
    attention = wef::transpose(attention, prem); // (batch_size, seq_len_q, num_heads, depth)
    
    size_t reshape[3] = {attention.m_shape[0], attention.m_shape[1], attention.m_shape[2] * attention.m_shape[3]};
    attention.reshape(reshape, 3); // (batch_size, seq_len_q, d_model)

    return m_out_layer->forward_pass(&attention, training, gpu);  // (batch_size, seq_len_q, d_model)
}

void MHA::merge_heads(Tensor& x, size_t seq_len)
{
    size_t perm[4] = {0, 2, 1, 3};
    x = wef::transpose(x, perm);

    size_t shape[3] = {m_batch, seq_len, m_d_model};
    x.reshape(shape, 3);
}

Tensor* MHA::backward_pass(const Tensor* dy, const float lr, void* gpu)
{
    m_temp = m_out_layer->backward_pass(dy, lr, gpu);
    size_t shape[4] = {m_batch, m_seq_len_q, m_num_heads, m_depth};
    m_temp->reshape(shape, 4);


    size_t prem[4] = {0, 2, 1, 3};
    *m_temp = wef::transpose(*m_temp, prem);

    // undo "Tensor z = wef::matmul(aij, v);"
    if (m_use_gpu)
    {
        m_dv = wef::matmul_GPU(gpu, wef::transpose(m_aij), *m_temp);
        m_daij = wef::matmul_GPU(gpu, *m_temp, wef::transpose(m_v));
    }
    else
    {
         m_dv = wef::matmul(wef::transpose(m_aij), *m_temp);
         m_daij = wef::matmul(*m_temp, wef::transpose(m_v));
    }

    // undo softmax
    m_de = (m_daij - wef::reducesum(m_daij * m_aij, 3, /*keepkims*/true)) * m_aij;

    // undo "product = wef::matmul(q, wef::transpose(k));"
    const float scale = 1.0f / std::sqrt(m_keys_dim);
    if (m_use_gpu)
    {
        m_dq = wef::matmul_GPU(gpu, m_de, m_k) * scale;
        m_dk = wef::matmul_GPU(gpu, wef::transpose(m_de), m_q) * scale;
    }
    else
    {
         m_dq = wef::matmul(m_de, m_k) * scale;
         m_dk = wef::matmul(wef::transpose(m_de), m_q) * scale;
    }

    // undo split
    merge_heads(m_dq, m_seq_len_q);
    merge_heads(m_dk, m_seq_len_k);
    merge_heads(m_dv, m_seq_len_v);

    m_dq_in = m_wq->backward_pass(&m_dq, lr, gpu);
    m_dk_in = m_wk->backward_pass(&m_dk, lr, gpu);
    m_dv_in = m_wv->backward_pass(&m_dv, lr, gpu);
    
    if (m_self_attention)
    {
        m_output = *m_dq_in + *m_dk_in + *m_dv_in;
        return &m_output;
    }
    else
    {
        m_output = *((Tensor[3]){*m_dq_in, *m_dk_in, *m_dv_in}); // TODO CHECK
        return &m_output;
    }
}

Tensor* Embedding::forward_pass(const Tensor* px, const bool training, void*) 
{
    if (!m_init) 
    {   
        // initially initilize the shape of m_X later just copy the tensors
        m_X = Tensor(*px);

        m_dist = std::normal_distribution<float>(0.0f, std::sqrt( 2.0f / m_vocab_size));

        size_t w_shape[2] = {m_vocab_size, m_d_model};
        m_W = Tensor::create(w_shape, 2);

        float* pm = m_W.m_tensor;
        for (size_t i = 0; i < m_W.m_size; i++) pm[i] = m_dist(m_g);

        m_num_param = m_W.m_size;

        // in: [b, ml, 1(token)] or [b, ml]
        // m_out: [b, ml, d_model]
        if (px->m_shape[px->m_rank - 1] == 1)
            m_out_rank = px->m_rank;
        else
            m_out_rank = px->m_rank + 1;

        m_out_shape = std::make_unique<size_t[]>(m_out_rank);

        std::memcpy(m_out_shape.get(), px->m_shape, px->m_rank * sizeof(size_t)); // either px shape option works here cause we modify the last axis later
        m_out_shape[m_out_rank - 1] = m_d_model;

        // gradient wrt the layer below
        m_dx = Tensor(*px);

        // gradients wrt weight
        m_dw = Tensor(m_W);

        m_init = true;
    }
    else
    {
        // TODO : add resuse guard
    }

    // TODO : catch if m_X shape m_changes during training
    
    if (training)
        std::memcpy(m_X.m_tensor, px->m_tensor, m_X.m_size * sizeof(float));

    m_out_shape[0] = px->m_shape[0]; // flexable batch
    m_out = Tensor::create(m_out_shape.get(), m_out_rank);

    float* pin = px->m_tensor;
    float* pout = m_out.m_tensor;
    float* pm_W = m_W.m_tensor;
    size_t base;
    for (size_t i = 0; i < px->m_size; i++)
    {
        // cast the index (vocab) into an int and use it to index from the embedding table
        // pin[i] is the embedding row we want to index m_out
        base = (size_t)pin[i] * m_d_model;
        // end = (size_t)pin[i] * m_d_model + m_d_model;
        memcpy(pout + i * m_d_model, pm_W + base, sizeof(float) * m_d_model);
    }

    return &m_out;

}

Tensor* Embedding::backward_pass(const Tensor* dy, const float lr, void*) 
{
    float* m_dx_ptr = m_dx.m_tensor;
    float* m_dw_ptr = m_dw.m_tensor;
    float* dy_ptr = dy->m_tensor;
    float* m_X_ptr = m_X.m_tensor;

    std::memset(m_dx_ptr, 0, (m_dx.m_size) * sizeof(float)); // zero fill
    std::memset(m_dw_ptr, 0, (m_dw.m_size) * sizeof(float)); // zero fill

    float* temp_m_dw = 0;
    float* temp_dy = 0;
    for (size_t i = 0; i < m_X.m_size; i++)
    {
        temp_m_dw = m_dw_ptr + (size_t)m_X_ptr[i] * m_d_model;
        temp_dy = dy_ptr + i * m_d_model;
        for (size_t j = 0; j < m_d_model; j++)
            temp_m_dw[j] += temp_dy[j];
    }

    m_W -= m_dw * lr / dy->m_shape[0];

    return &m_dx;
}

