#include <iostream>
#include <random>
#include "../include/layers.h"

Tensor* Linear::forward_pass(const Tensor& px, const bool training, void*) 
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


        if (m_use_bias) out = wef::matmul(px, W, true) + B;
        else out = wef::matmul(px, W, true);
        return &out;

        // if (m_use_bias) return &wef::matmul(px, W) + B;
        // return &wef::matmul(px, W);
    }

Tensor* Linear::backward_pass(const Tensor& dy, const float lr, void*) 
    {
        // gradient wrt the layer below
        dx = wef::matmul(dy, wef::transpose(W));

        // gradient wrt weights sum everything aside from the last two axes. 
        // CATCH rank < 2?????
        dw = wef::matmul(wef::transpose(X), dy);
        for (size_t i = 0; i < dw.m_rank - 2; i++) dw = wef::reducesum(dw, i);

        W -= dw * lr / dy.m_shape[0];

        if (m_use_bias) 
        {
            // gradient wrt bias sum everything aside from the last axis
            db = dy;
            for (size_t i = 0; i < db.m_rank - 1; i++) db = wef::reducesum(db, i);
            B -= db * lr / dy.m_shape[0];
        }

        return &dx;
    }

Tensor* Conv2D::forward_pass(const Tensor& px, const bool training, void*) 
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

        // gradient wrt the layer below
        dx = Tensor(px);

        // gradients wrt weights and biases
        dw = Tensor(W);
        db = Tensor(B);
        
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
    
    float* out_ptr = out.m_tensor;
    float* px_ptr = px.m_tensor;
    float* W_ptr = W.m_tensor;
    float* B_ptr = B.m_tensor;

    std::memset(out_ptr, 0, (out.m_size) * sizeof(float));

    /*
    There is a lot of math below but the idea is to do the cov kernel math (W * Input) and expand 
    this to the units dimension by repeating the index of Input as before.

    Here what I do is index out out and W in contigous manner to make it faster and for Input
    I jump everytime we hit the width of the weight to the second row and use some math to do that.
    */

    size_t out_wo_units = out.m_size / units;
    size_t skip_w_help = ch * (w_width - 1);
    size_t bi_help = out_wo_units / out.m_shape[0];
    size_t skip_h_help = (w_height - 1) * px.m_shape[px.m_rank-2] * px.m_shape[px.m_rank-1];
    size_t offset = width - w_width;
    size_t id_help = w_width * ch;

    #pragma omp parallel for schedule(static)
    for (size_t out_i = 0; out_i < out_wo_units; out_i++)
    {
        size_t skip_w = skip_w_help * (out_i / out.m_shape[out.m_rank-2]);
        size_t bi = out_i / bi_help;
        size_t skip_h = bi * skip_h_help;

        for (size_t w_i = 0; w_i < W.m_size / units; w_i++)
        {
            float temp_px = px_ptr[
                ch * out_i + skip_w + skip_h
                + 
                w_i + ch * offset * (w_i / id_help)
            ];

            for (size_t u_i = 0; u_i < units; u_i++)
                out_ptr[out_i * units + u_i] += temp_px * W_ptr[w_i * units + u_i];
        }
        if (m_use_bias)
            for (size_t u_i = 0; u_i < units; u_i++)
                out_ptr[out_i * units + u_i] += B_ptr[u_i];
    }

    return &out;
    }

Tensor* Conv2D::backward_pass(const Tensor& dy, const float lr, void*) 
    {   
        float* dx_ptr = dx.m_tensor;
        float* dw_ptr = dw.m_tensor;

        std::memset(dx_ptr, 0, (dx.m_size) * sizeof(float)); // zero fill
        std::memset(dw_ptr, 0, (dw.m_size) * sizeof(float)); // zero fill

        float* dy_ptr = dy.m_tensor;
        float* W_ptr = W.m_tensor;
        float* X_ptr = X.m_tensor;

        size_t out_wo_units = dy.m_size / units;
        size_t skip_w_help = ch * (w_width - 1);
        size_t bi_help = out_wo_units / dy.m_shape[0];
        size_t skip_h_help = (w_height - 1) * X.m_shape[X.m_rank-2] * X.m_shape[X.m_rank-1];
        size_t offset = width - w_width;
        size_t id_help = w_width * ch;

        #pragma omp parallel for schedule(static)
        for (size_t dy_i = 0; dy_i < out_wo_units; dy_i++)
        {
            size_t skip_w = skip_w_help * (dy_i / dy.m_shape[dy.m_rank-2]);
            size_t bi = dy_i / bi_help;
            size_t skip_h = bi * skip_h_help;

            for (size_t w_i = 0; w_i < W.m_size / units; w_i++)
            {
                size_t id1 = 
                    ch * dy_i + skip_w + skip_h
                    + 
                    w_i + ch * offset * (w_i / id_help);

                for (size_t u_i = 0; u_i < units; u_i++)
                {
                    float grad = dy_ptr[dy_i * units + u_i];
                    dx_ptr[id1] += grad * W_ptr[w_i * units + u_i];
                    dw_ptr[w_i * units + u_i] += grad * X_ptr[id1];
                }
            }
        }

        // divide lr by batch size
       W -= dw * lr /dy.m_shape[0];

        if (m_use_bias)
        {
            db = dy;
            for (size_t i = 0; i < db.m_rank - 1; i++) db = wef::reducesum(db, i);
            B -= db * lr / dy.m_shape[0];
        }

        return &dx;
    }

Tensor* Conv2D_legacy::forward_pass(const Tensor& px, const bool training, void*) 
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

        // gradient wrt the layer below
        dx = Tensor(px);
        
        // gradients wrt weights and biases
        dw = Tensor(W);
        db = Tensor(B);

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
    float* pm_out = out.m_tensor;
    float* pm_b = B.m_tensor;

    size_t ind = 0;
    size_t i1[4];
    size_t i2[4];

    #pragma omp parallel for collapse(4) schedule(static)
    for (size_t b1 = 0; b1 < out.m_shape[0]; b1++)
    {
        for (size_t h1 = 0; h1 < out.m_shape[1]; h1++)
        {
            for (size_t w1 = 0; w1 < out.m_shape[2]; w1++)
            {
                for (size_t oc = 0; oc < out.m_shape[3]; oc++)    
                {
                    float temp = 0.0f;
                    for (size_t h2 = 0; h2 < w_height; h2++)
                    {
                        for (size_t w2 = 0; w2 < w_width; w2++)
                        {
                            for (size_t c2 = 0; c2 < ch; c2++)
                            {
                                // doing this to ensure cstyle array for indexing
                                // not making a new array and just rewriting
                                i1[0] = b1; i1[1] = h2 + h1; i1[2] = w2 + w1; i1[3] = c2;
                                i2[0] = h2; i2[1] = w2; i2[2] = c2; i2[3] = oc; 

                                temp += px[i1] * W[i2];
                            }
                        }
                    }
                    pm_out[ind++] = temp + (m_use_bias ? pm_b[oc] : 0.0); // TODO: check BIAS
                }
            }
        }
    }

    return &out;
    }

Tensor* Conv2D_legacy::backward_pass(const Tensor& dy, const float lr, void*) 
    {
        std::memset(dx.m_tensor, 0, (dx.m_size) * sizeof(float)); // zero fill
        std::memset(dw.m_tensor, 0, (dw.m_size) * sizeof(float)); // zero fill

        size_t ind = 0;
        size_t i1[4];
        size_t i2[4];

        #pragma omp parallel for collapse(4) schedule(static)
        for (size_t b1 = 0; b1 < dy.m_shape[0]; b1++)
        {
            for (size_t h1 = 0; h1 < dy.m_shape[1]; h1++)
            {
                for (size_t w1 = 0; w1 < dy.m_shape[2]; w1++)
                {
                    for (size_t oc = 0; oc < dy.m_shape[3]; oc++)
                    {
                        float grad = dy.m_tensor[ind++];
                        for (size_t h2 = 0; h2 < w_height; h2++)
                        {
                            for (size_t w2 = 0; w2 < w_width; w2++)
                            {
                                for (size_t c2 = 0; c2 < ch; c2++)
                                {
                                    // doing this to ensure cstyle array for indexing
                                    // not making a new array and just rewriting
                                    i1[0] = b1; i1[1] = h2 + h1; i1[2] = w2 + w1; i1[3] = c2;
                                    i2[0] = h2; i2[1] = w2; i2[2] = c2; i2[3] = oc; 

                                    dx[i1] += grad * W[i2];
                                    dw[i2] += grad * X[i1];
                                }
                            }
                        }
                    }
                }
            }
        }
        
        float* pm_w = W.m_tensor;
        float* pm_dw = dw.m_tensor;
        // divide lr by batch size
        W -= dw * lr / dy.m_shape[0];

        if (m_use_bias)
        {
            db = dy;
            for (size_t i = 0; i < db.m_rank - 1; i++) db = wef::reducesum(db, i);
            B -= db * lr / dy.m_shape[0];
        }

        return &dx;
    }

Tensor* MaxPool2D::forward_pass(const Tensor& px, const bool training, void*) 
    {
        if (!init)
        {   
            // initially initilize the shape of X later just copy the tensors
            X = Tensor(px);
            // h, w, c, units
            height = px.m_shape[1]; width = px.m_shape[2]; ch = px.m_shape[3];

            size_t ax1 = (height + (height%k_height)) / k_height;
            size_t ax2 = (width + (width%k_width)) / k_width;

            size_t o_size = px.m_shape[0] * ax1 * ax2 * ch;
            
            // this get the argmax in a nested for loop (2D) I made it flat for speed
            argmax = std::make_unique<size_t[]>(o_size * 4);

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

        size_t ind = 0;
        size_t i1[4];

        #pragma omp parallel for collapse(4) schedule(static)
        for (size_t b1 = 0; b1 < out.m_shape[0]; b1++)
        {
            for (size_t h1 = 0; h1 < out.m_shape[1]; h1++)
            {
                for (size_t w1 = 0; w1 < out.m_shape[2]; w1++)
                {
                    for (size_t c = 0; c < out.m_shape[3]; c++)
                    {
                        float temp_val = -1e19f;
                        size_t temp_ind[4];
                        for (size_t h2 = h1 * k_height; h2 < h1 * k_height + k_height; h2++)
                        {
                            if (h2 >= height) break;
                            for (size_t w2 = w1 * k_width; w2 < w1 * k_width + k_width; w2++)
                            {
                                if (w2 >= width) break;

                                i1[0] = b1; i1[1] = h2; i1[2] = w2; i1[3] = c;
                                float val = px[i1];

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
                        out.m_tensor[ind] = temp_val;

                        // only populate argmax during training for speed, idt it affects values if we keep it tho
                        if (training) for (size_t ii = 0; ii < 4; ii++) argmax[ind * 4 + ii] = temp_ind[ii];
                        ind++;
                    }
                }
            }
        }
        return &out;
    }

Tensor* MaxPool2D::backward_pass(const Tensor& dy, const float lr, void*) 
    {
        std::memset(dx.m_tensor, 0, (dx.m_size) * sizeof(float));  // zero fill
        size_t ind = 0;
        size_t i1[4];

        #pragma omp parallel for collapse(4) schedule(static)
        for (size_t b1 = 0; b1 < dy.m_shape[0]; b1++)
        {
            for (size_t h1 = 0; h1 < dy.m_shape[1]; h1++)
            {
                for (size_t w1 = 0; w1 < dy.m_shape[2]; w1++)
                {
                    for (size_t c = 0; c < dy.m_shape[3]; c++)    
                    {
                        i1[0] = argmax[ind * 4 + 0]; i1[1] = argmax[ind * 4 + 1]; 
                        i1[2] = argmax[ind * 4 + 2]; i1[3] = argmax[ind * 4 + 3];
                        dx[i1] = dy.m_tensor[ind];
                        ind++; 
                    }
                }
            }
        }
        return &dx;
    }

Tensor* ReduceSum::forward_pass(const Tensor& px, const bool training, void*) 
    { 

        if (!init) 
        {
            // initially initilize the shape of X later just copy the tensors
            X = Tensor(px);

            if (ax >= px.m_rank) throw std::invalid_argument("axis outside shape");
            const size_t* shape = px.m_shape; // [b, h, w, c]

            m_out_rank = keepdims ? px.m_rank : px.m_rank - 1;
            m_out_shape = std::make_unique<size_t[]>(m_out_rank);  // [b, h, w] or  // [b, h, w, 1]
            
            // index 0 will be reassigned if ax > 0 so as to be flexable with the batch
            size_t j = 0;
            for (size_t i = 0; i < px.m_rank; i++)
            {
                if (i != ax) m_out_shape[j++] = shape[i]; 
                else if (keepdims) m_out_shape[j++] = 1;
            }

            dx = Tensor(px);
            
            init = true;
        }

        // copy px into X
        if (training) std::memcpy(X.m_tensor, px.m_tensor, X.m_size * sizeof(float));

        // batch flexability only applies if ax != 0
        if (ax != 0)
            m_out_shape[0] = px.m_shape[0];

        out = Tensor::create(m_out_shape.get(), m_out_rank);

        const float* pm = px.m_tensor;
        float* pm_out = out.m_tensor;

        size_t eaa = 1; // everything after axis i.e. b, h w, axis, x1, x2 -> eaa = x1 * x2
        for (size_t i = ax + 1; i < px.m_rank; i++) eaa *= px.m_shape[i];

        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < out.m_size; i++)
        {
            float temp = 0.0f;
            size_t mult = (i/eaa) * (1 - px.m_shape[ax]);

            for (size_t j = 0; j < px.m_shape[ax]; j++)
                temp += pm[i  + eaa * (j - mult)];

            pm_out[i] = temp;
        }

        return &out;
    }

Tensor* ReduceSum::backward_pass(const Tensor& dy, float, void*) 
    {
        if (!init) throw std::invalid_argument("layer not initilized");

        const float* pdy = dy.m_tensor;
        float* pdx = dx.m_tensor; // no need to zero fill cause we dont do +=

        size_t eaa = 1;
        for (size_t i = ax + 1; i < dx.m_rank; i++) eaa *= dx.m_shape[i];

        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < dy.m_size; i++)
        {
            size_t mult = (i/eaa) * (1 - dx.m_shape[ax]) ;
            for (size_t j = 0; j < dx.m_shape[ax]; j++)
            {
                pdx[i  + eaa * (j - mult)] = pdy[i];
            }
        }
        return &dx;
    }

Tensor* LayerNorm::forward_pass(const Tensor& px, const bool training, void*)
{
    if (!init) 
        {
            // initially initilize the shape of X later just copy the tensors
            X = Tensor(px);

            ax_val = px.m_shape[axis];
            std::unique_ptr<size_t[]> beta_shape = std::make_unique<size_t[]>(px.m_rank);
            std::unique_ptr<size_t[]> gamma_shape = std::make_unique<size_t[]>(px.m_rank);

            // fill with 1s
            std::fill_n(beta_shape.get(), px.m_rank, 1);
            std::fill_n(gamma_shape.get(), px.m_rank, 1);

            beta_shape[axis] = px.m_shape[axis];
            gamma_shape[axis] = px.m_shape[axis];

            beta = Tensor::create(beta_shape.get(), px.m_rank);
            gamma = Tensor::create(gamma_shape.get(), px.m_rank);

            // initilize beta and gamma
            std::fill_n(beta.m_tensor, ax_val, 0.01f);
            std::fill_n(gamma.m_tensor, ax_val, 0.99f);

            m_num_param = beta.m_size + gamma.m_size;

            m_out_rank = px.m_rank;
            m_out_shape = std::make_unique<size_t[]>(m_out_rank);
            std::memcpy(m_out_shape.get(), px.m_shape, m_out_rank * sizeof(size_t));
            
            init = true;
        }
        if (px.m_shape[axis] != ax_val) throw std::invalid_argument("cannot reuse layer [LayerNorm]");

        // copy px into X
        if (training) std::memcpy(X.m_tensor, px.m_tensor, X.m_size * sizeof(float));

        // follwoing Ba et al. 2016
        mu = wef::reducesum(px, /*axis=*/axis) / ax_val;
        x_mu = px - mu;
        var = wef::reducesum(x_mu * x_mu, /*axis=*/axis) / ax_val;
        inv_std = wef::pow(var + eps, -0.5f);
        x_i_hat = x_mu * inv_std;
        y_i = x_i_hat * gamma + beta;

        return &y_i;
}

Tensor* LayerNorm::backward_pass(const Tensor& dy, const float lr, void*)
{
    if (!init) throw std::invalid_argument("layer not initilized");

    d_gamma = dy * x_i_hat;
    d_beta = dy;
    for (size_t i = 0; i < dy.m_rank; i++)
    {
        if (i != axis)
        {
            d_gamma = wef::reducesum(d_gamma, i);
            d_beta = wef::reducesum(d_beta, i);
        }
    }

    dx = inv_std * (1.0 / ax_val) * 
    (
        gamma * dy * ax_val 
        - wef::reducesum(gamma * dy, axis)
        - x_i_hat * (wef::reducesum(gamma * dy * x_i_hat, axis))
    );

    gamma -= d_gamma * lr / dy.m_shape[0];
    beta -= d_beta * lr / dy.m_shape[0];

    return &dx;
}

Tensor* Conv2D_NR::forward_pass(const Tensor& px, const bool training, void*) 
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

        // gradient wrt the layer below
        dx = Tensor(px);
        
        // gradients wrt weights and biases
        dw = Tensor(W);
        db = Tensor(B);

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
    float* pm_out = out.m_tensor;
    float* pm_b = B.m_tensor;

    size_t ind = 0;
    size_t i1[4];
    size_t i2[4];

    #pragma omp parallel for collapse(4) schedule(static)
    for (size_t b1 = 0; b1 < out.m_shape[0]; b1++)
    {
        for (size_t h1 = 0; h1 < out.m_shape[1]; h1++)
        {
            for (size_t w1 = 0; w1 < out.m_shape[2]; w1++)
            {
                for (size_t oc = 0; oc < out.m_shape[3]; oc++)    
                {
                    float temp = 0.0f;
                    for (size_t h2 = 0; h2 < w_height; h2++)
                    {
                        for (size_t w2 = 0; w2 < w_width; w2++)
                        {
                            for (size_t c2 = 0; c2 < ch; c2++)
                            {
                                // doing this to ensure cstyle array for indexing
                                // not making a new array and just rewriting
                                i1[0] = b1; i1[1] = h2 + h1; i1[2] = w2 + w1; i1[3] = c2;
                                i2[0] = h2; i2[1] = w2; i2[2] = c2; i2[3] = oc; 

                                temp += px[i1] * W[i2];
                            }
                        }
                    }
                    pm_out[ind++] = temp + (m_use_bias ? pm_b[oc] : 0.0); // TODO: check BIAS
                }
            }
        }
    }

    return &out;
}

Tensor* Conv2D_NR::backward_pass(const Tensor& dy, const float lr, void*) 
{
    std::memset(dx.m_tensor, 0, (dx.m_size) * sizeof(float)); // zero fill
    std::memset(dw.m_tensor, 0, (dw.m_size) * sizeof(float)); // zero fill

    #pragma omp parallel for collapse(4) schedule(static)
    for (size_t bin_hin = 0; bin_hin < dx.m_shape[0] * dx.m_shape[1]; bin_hin++)
        {
            size_t bin = bin_hin / dx.m_shape[1];
            size_t hin = bin_hin % dx.m_shape[1];
            
            for (size_t win_cin = 0; win_cin < dx.m_shape[2] * dx.m_shape[3]; win_cin++)  
                {
                    size_t win = win_cin / dx.m_shape[3];
                    size_t cin = win_cin % dx.m_shape[3];

                    float temp = 0.0;
                    for (size_t hk = 0; hk < w_height; hk++)            
                        {
                            if (hin < hk) continue;
                            size_t ho = hin - hk; // move out

                            for (size_t wk = 0; wk < w_width; wk++)                
                            {
                                if (win < wk) continue;
                                size_t wo = win - wk; // move out

                                for (size_t co = 0; co < units; co++)
                            
                                {
                                    if (ho >= dy.m_shape[1] || wo >= dy.m_shape[2]) continue;
                                    
                                    size_t ik[4] = {hk, wk, cin, co};
                                    size_t io[4] = {bin, ho, wo, co};

                                    temp += dy[io] * W[ik];
                                }
                            }
                        }
                    size_t iin[4] = {bin, hin, win, cin};
                    dx[iin] = temp;
                }
        }
    
    #pragma omp parallel for collapse(4) schedule(static)
    for (size_t hk_wk = 0; hk_wk < dw.m_shape[0] * dw.m_shape[1]; hk_wk++)
        {
            size_t hk = hk_wk / dw.m_shape[1];
            size_t wk = hk_wk % dw.m_shape[1];
        
            for (size_t cin_co = 0; cin_co < dw.m_shape[2] * dw.m_shape[3]; cin_co++)
            {
                size_t cin = cin_co / dw.m_shape[3];
                size_t co = cin_co % dw.m_shape[3];
                
                float temp = 0.0;
                for (size_t bin = 0; bin < dy.m_shape[0]; bin++)
                    {
                        for (size_t ho = 0; ho < dy.m_shape[1]; ho++)
                        {
                            // I chose to do this rather than doing the inner 3 loops
                            // with dx.m_shape[...] b/c otherwise I'd have to add three 
                            // bounds checks like "if (win < wk) continue;"
                            size_t hin = ho + hk;

                            for (size_t wo = 0; wo < dy.m_shape[2]; wo++)  
                            {
                                
                                size_t win = wo + wk;
                                
                                size_t iin[4] = {bin, hin, win, cin};
                                size_t io[4] = {bin, ho, wo, co};

                                temp += dy[io] * X[iin];
                            }
                        }
                    }
                size_t ik[4] = {hk, wk, cin, co};
                dw[ik] = temp;
                
            }
        }


    float* pm_w = W.m_tensor;
    float* pm_dw = dw.m_tensor;
    // divide lr by batch size
    W -= dw * lr / dy.m_shape[0];

    if (m_use_bias)
    {
        db = dy;
        for (size_t i = 0; i < db.m_rank - 1; i++)
            db = wef::reducesum(db, i);
        B -= db * lr / dy.m_shape[0];
    }

    return &dx;
}

Tensor MHA::scaled_dot_product_attention(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor* mask)
{
    Tensor product = wef::matmul(q, wef::transpose(k));
    keys_dim = (float)k.m_shape[k.m_rank-1];

    Tensor eij = product / sqrt(keys_dim);

    if (mask)
        eij += (*mask * -1e9f); // TODO : add tensor += overload

    m_aij = wef::softmax(eij);
    Tensor z = wef::matmul(m_aij, v);
    return z;
}

void MHA::split_heads(Tensor& x, size_t seq_len)
{
    size_t shape[4] = {m_batch, seq_len, m_num_heads, m_depth};
 
    delete[] x.m_shape;
    x.m_shape = new size_t[4];
    memcpy(x.m_shape, shape, sizeof(size_t) * 4);
    x.m_rank = 4; // (batch_size, seq_len_q, d_model)

    size_t prem[4] = {0, 2, 1, 3};
    x = wef::transpose(x, prem);
}

Tensor* MHA::forward_pass(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor* mask, const bool training, void* gpu)
{
    if (!init)
    {
        m_batch = q.m_shape[0];
        m_seq_len_q = q.m_shape[1];
        m_seq_len_k = k.m_shape[1];
        m_seq_len_v = v.m_shape[1];
        wq = std::make_unique<Linear>(m_d_model, m_use_bias);
        wk = std::make_unique<Linear>(m_d_model, m_use_bias);
        wv = std::make_unique<Linear>(m_d_model, m_use_bias);
        out_layer = std::make_unique<Linear>(m_d_model, m_use_bias);
        init = true;
    }

    // TODO : add shape check after init

    m_q = *wq->forward_pass(q, training, gpu);  // (batch_size, seq_len, d_model)
    m_k = *wk->forward_pass(k, training, gpu);  // (batch_size, seq_len, d_model)
    m_v = *wv->forward_pass(v, training, gpu);  // (batch_size, seq_len, d_model)

    split_heads(m_q, m_seq_len_q);  // (batch_size, num_heads, seq_len_q, depth)
    split_heads(m_k, m_seq_len_k);  // (batch_size, num_heads, seq_len_k, depth)
    split_heads(m_v, m_seq_len_v);  // (batch_size, num_heads, seq_len_v, depth)

    Tensor attention = scaled_dot_product_attention(m_q, m_k, m_v, mask);

    size_t prem[4] = {0, 2, 1, 3};
    attention = wef::transpose(attention, prem); // (batch_size, seq_len_q, num_heads, depth)
    
    size_t reshape[3] = {attention.m_shape[0], attention.m_shape[1], m_d_model};
    delete[] attention.m_shape;
    attention.m_shape = new size_t[3];
    memcpy(attention.m_shape, reshape, sizeof(size_t) * 3);
    attention.m_rank = 3; // (batch_size, seq_len_q, d_model)

    return out_layer->forward_pass(attention, training, gpu);  // (batch_size, seq_len_q, d_model)
}

void MHA::merge_heads(Tensor& x, size_t seq_len)
{
    size_t perm[4] = {0, 2, 1, 3};
    x = wef::transpose(x, perm);
    size_t shape[3] = {m_batch, seq_len, m_d_model};
    delete[] x.m_shape;
    x.m_shape = new size_t[3];
    memcpy(x.m_shape, shape, sizeof(size_t) * 3);
    x.m_rank = 3;
}

Tensor* MHA::backward_pass(const Tensor& dy, const float lr, void* gpu)
{
    Tensor* dy_ptr = out_layer->backward_pass(dy, lr, gpu);

    size_t shape[4] = {m_batch, m_seq_len_q, m_num_heads, m_depth};
    delete[] dy_ptr->m_shape;
    dy_ptr->m_shape = new size_t[4];
    memcpy(dy_ptr->m_shape, shape, sizeof(size_t) * 4);
    dy_ptr->m_rank = 4;

    size_t prem[4] = {0, 2, 1, 3};
    *dy_ptr = wef::transpose(*dy_ptr, prem);

    // undo "Tensor z = wef::matmul(aij, v);"
    Tensor dv = wef::matmul(wef::transpose(m_aij), *dy_ptr);
    Tensor daij = wef::matmul(*dy_ptr, wef::transpose(m_v));

    // undo softmax
    Tensor de = (daij - wef::reducesum(daij * m_aij, 3)) * m_aij;

    // undo "Tensor eij = product / sqrt(keys_dim);"
    const float scale = 1.0f / std::sqrt(keys_dim);
    Tensor dq = wef::matmul(de, m_k) * scale;
    Tensor dk = wef::matmul(wef::transpose(de), m_q) * scale;
    
    // undo split
    merge_heads(dq, m_seq_len_q);
    merge_heads(dk, m_seq_len_k);
    merge_heads(dv, m_seq_len_v);

    Tensor* dq_in = wq->backward_pass(dq, lr, gpu);
    Tensor* dk_in = wk->backward_pass(dk, lr, gpu);
    Tensor* dv_in = wv->backward_pass(dv, lr, gpu);
    
    if (m_self_attention)
    {
        m_output = *dq_in + *dk_in + *dv_in;
        return &m_output;
    }
    else
        return dq_in;
}

Tensor* Embedding::forward_pass(const Tensor& px, const bool training, void*) 
    {
        if (!init) 
        {   
            // initially initilize the shape of X later just copy the tensors
            X = Tensor(px);

            dist = std::normal_distribution<float>(0.0f, std::sqrt( 2.0f / m_vocab_size));

            size_t w_shape[2] = {m_vocab_size, m_d_model};
            W = Tensor::create(w_shape, 2);

            float* pm = W.m_tensor;
            for (size_t i = 0; i < W.m_size; i++) pm[i] = dist(g);

            m_num_param = W.m_size;

            // in: [b, ml, 1(token)] or [b, ml]
            // out: [b, ml, d_model]
            if (px.m_shape[px.m_rank - 1] == 1)
                m_out_rank = px.m_rank;
            else
                m_out_rank = px.m_rank + 1;

            m_out_shape = std::make_unique<size_t[]>(m_out_rank);

            std::memcpy(m_out_shape.get(), px.m_shape, px.m_rank * sizeof(size_t)); // either px shape option works here cause we modify the last axis later
            m_out_shape[m_out_rank - 1] = m_d_model;

            // gradient wrt the layer below
            dx = Tensor(px);

            // gradients wrt weight
            dw = Tensor(W);

            init = true;
        }
        else
        {
            // TODO : add resuse guard
        }

        // copy px into X
        // TODO : catch if X shape changes during training
        if (training) std::memcpy(X.m_tensor, px.m_tensor, X.m_size * sizeof(float));

        m_out_shape[0] = px.m_shape[0]; // flexable batch
        out = Tensor::create(m_out_shape.get(), m_out_rank);


        // cast the index (vocab) into an int and use it to index from the embedding table
        float* pin = px.m_tensor;
        float* pout = out.m_tensor;
        float* pW = W.m_tensor;
        size_t base;
        for (size_t i = 0; i < px.m_size; i++)
        {
            // pin[i] is the embedding row we want to index out
            base = (size_t)pin[i] * m_d_model;
            // end = (size_t)pin[i] * m_d_model + m_d_model;
            memcpy(pout + i * m_d_model, pW + base, sizeof(float) * m_d_model);
        }

        return &out;

    }

Tensor* Embedding::backward_pass(const Tensor& dy, const float lr, void*) 
    {
        float* dx_ptr = dx.m_tensor;
        float* dw_ptr = dw.m_tensor;
        float* dy_ptr = dy.m_tensor;
        float* X_ptr = X.m_tensor;

        std::memset(dx_ptr, 0, (dx.m_size) * sizeof(float)); // zero fill
        std::memset(dw_ptr, 0, (dw.m_size) * sizeof(float)); // zero fill

        float* temp_dw = 0;
        float* temp_dy = 0;
        for (size_t i = 0; i < X.m_size; i++)
        {
            temp_dw = dw_ptr + (size_t)X_ptr[i] * m_d_model;
            temp_dy = dy_ptr + i * m_d_model;
            for (size_t j = 0; j < m_d_model; j++)
                temp_dw[j] += temp_dy[j];
        }

        W -= dw * lr / dy.m_shape[0];

        return &dx;
    }

