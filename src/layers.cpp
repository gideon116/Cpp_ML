#include <iostream>
#include <random>
#include "../include/layers.h"

Tensor*  Linear::forward_pass(const Tensor& px, const bool training) 
    {
        if (!init) 
        {   
            // initially initilize the shape of X later just copy the tensors
            X = Tensor(px);

            size_t w_shape[2] = {px.col, units};
            size_t b_shape[2] = {1, units};
            W = Tensor::create(w_shape, 2);
            B = Tensor::create(b_shape, 2);

            float* B_ptr = B.tensor.get();
            std::fill_n(B.tensor.get(), B.tot_size, 0.01f); // zero fill

            float* pm = W.tensor.get();
            for (size_t i = 0; i < size_t(px.col) * units; i++) pm[i] = dist(g);

            m_num_param = W.tot_size + (usebias ? B.tot_size : 0);
            
            m_out_rank = px.rank;
            m_out_shape = std::make_unique<size_t[]>(m_out_rank);
            std::memcpy(m_out_shape.get(), px.shape.get(), m_out_rank * sizeof(size_t));
            // TO DO: CATCH < 1 RANK
            m_out_shape[m_out_rank - 1] = units;

            init = true;
        }
        else
        {
            // if trying to use (reuse) the layer on a different tensor
            if (W.row != px.col) throw std::invalid_argument("cannot reuse layer");
        }

        // copy px into X
        if (training) std::memcpy(X.tensor.get(), px.tensor.get(), X.tot_size * sizeof(float));


        if (usebias) out = wef::matmul(px, W, true) + B;
        else out = wef::matmul(px, W, true);
        return &out;

        // if (usebias) return &wef::matmul(px, W) + B;
        // return &wef::matmul(px, W);
    }

Tensor*  Linear::backward_pass(const Tensor& dy, const float lr) 
    {
        // gradient wrt the layer below
        dx = wef::matmul(dy, wef::transpose(W));

        // gradient wrt weights sum everything aside from the last two axes. 
        // CATCH rank < 2?????
        dw = wef::matmul(wef::transpose(X), dy);
        for (size_t i = 0; i < dw.rank - 2; i++) dw = wef::reducesum(dw, i);

        W = W - dw * lr / dy.shape[0];

        if (usebias) 
        {
            // gradient wrt bias sum everything aside from the last axis
            db = dy;
            for (size_t i = 0; i < db.rank - 1; i++) db = wef::reducesum(db, i);
            B = B - db * lr / dy.shape[0];
        }

        return &dx;
    }

Tensor*  Conv2D::forward_pass(const Tensor& px, const bool training) 
    {
    if (!init) 
    {   
        // initially initilize the shape of X later just copy the tensors
        X = Tensor(px);
        
        // h, w, c, units
        height = px.shape[1]; width = px.shape[2]; ch = px.shape[3];
        dist = std::normal_distribution<float>(0.0f, std::sqrt( 2.0f / (w_height * w_width * ch)));
        // for above now we have (2/fan_in = hwc)^0.5 good for relu we can use fan_out for tanh... which is hwu

        size_t w_shape[4] = {w_height, w_width, ch, units};
        W = Tensor::create(w_shape, 4);

        size_t B_shape[4] = {1, 1, 1, units};
        B = Tensor::create(B_shape, 4);
        std::fill_n(B.tensor.get(), B.tot_size, 0.01f);

        float* pm = W.tensor.get();
        for (size_t i = 0; i < W.tot_size; i++) pm[i] = dist(g);

        m_out_rank = px.rank; // this is 4, its always 4
        m_out_shape = std::make_unique<size_t[]>(m_out_rank); // heap allocation is not the best but we only do this once pre layer so its whatever
        m_out_shape[1] = height - w_height + 1;
        m_out_shape[2] = width - w_width + 1;
        m_out_shape[3] = units;

        // gradient wrt the layer below
        dx = Tensor(px);

        // gradients wrt weights and biases
        dw = Tensor(W);
        db = Tensor(B);
        
        m_num_param = W.tot_size + (usebias ? B.tot_size : 0);
        
        init = true;
    }
    else
    {
        // if trying to use (reuse) the layer on a different tensor
        if (px.shape[1] != height || 
            px.shape[2] != width ||
            px.shape[3] != ch) throw std::invalid_argument("cannot reuse layer");
    }

    // copy px into X
    if (training) std::memcpy(X.tensor.get(), px.tensor.get(), X.tot_size * sizeof(float));

    m_out_shape[0] = px.shape[0]; // flexable batch 
    out = Tensor::create(m_out_shape.get(), 4);
    
    float* out_ptr = out.tensor.get();
    float* px_ptr = px.tensor.get();
    float* W_ptr = W.tensor.get();
    float* B_ptr = B.tensor.get();

    std::memset(out_ptr, 0, (out.tot_size) * sizeof(float));

    /*
    There is a lot of math below but the idea is to do the cov kernel math (W * Input) and expand 
    this to the units dimension by repeating the index of Input as before.

    Here what I do is index out out and W in contigous manner to make it faster and for Input
    I jump everytime we hit the width of the weight to the second row and use some math to do that.
    */

    size_t out_wo_units = out.tot_size / units;
    size_t skip_w_help = ch * (w_width - 1);
    size_t bi_help = out_wo_units / out.shape[0];
    size_t skip_h_help = (w_height - 1) * px.row*px.col;
    size_t offset = width - w_width;
    size_t id_help = w_width * ch;

    #pragma omp parallel for schedule(static)
    for (size_t out_i = 0; out_i < out_wo_units; out_i++)
    {
        size_t skip_w = skip_w_help * (out_i / out.row);
        size_t bi = out_i / bi_help;
        size_t skip_h = bi * skip_h_help;

        for (size_t w_i = 0; w_i < W.tot_size / units; w_i++)
        {
            float temp_px = px_ptr[
                ch * out_i + skip_w + skip_h
                + 
                w_i + ch*offset * (w_i / id_help)
            ];

            for (size_t u_i = 0; u_i < units; u_i++)
                out_ptr[out_i * units + u_i] += temp_px * W_ptr[w_i * units + u_i];
        }
        if (usebias)
            for (size_t u_i = 0; u_i < units; u_i++)
                out_ptr[out_i * units + u_i] += B_ptr[u_i];
    }

    return &out;
    }

Tensor*  Conv2D::backward_pass(const Tensor& dy, const float lr) 
    {   
        float* dx_ptr = dx.tensor.get();
        float* dw_ptr = dw.tensor.get();

        std::memset(dx_ptr, 0, (dx.tot_size) * sizeof(float)); // zero fill
        std::memset(dw_ptr, 0, (dw.tot_size) * sizeof(float)); // zero fill

        float* dy_ptr = dy.tensor.get();
        float* W_ptr = W.tensor.get();
        float* X_ptr = X.tensor.get();

        size_t out_wo_units = dy.tot_size / units;
        size_t skip_w_help = ch * (w_width - 1);
        size_t bi_help = out_wo_units / out.shape[0];
        size_t skip_h_help = (w_height - 1) * X.row*X.col;
        size_t offset = width - w_width;
        size_t id_help = w_width * ch;

        #pragma omp parallel for schedule(static)
        for (size_t dy_i = 0; dy_i < out_wo_units; dy_i++)
        {
            size_t skip_w = skip_w_help * (dy_i / dy.row);
            size_t bi = dy_i / bi_help;
            size_t skip_h = bi * skip_h_help;

            for (size_t w_i = 0; w_i < W.tot_size / units; w_i++)
            {
                size_t id1 = 
                    ch * dy_i + skip_w + skip_h
                    + 
                    w_i + ch*offset * (w_i / id_help);

                for (size_t u_i = 0; u_i < units; u_i++)
                {
                    float grad = dy_ptr[dy_i * units + u_i];
                    dx_ptr[id1] += grad * W_ptr[w_i * units + u_i];
                    dw_ptr[w_i * units + u_i] += grad * X_ptr[id1];
                }
            }
        }

        // divide lr by batch size
        for (size_t i = 0; i < W.tot_size; i++) W_ptr[i] -= dw_ptr[i] * lr /dy.shape[0];

        if (usebias)
        {
            db = dy;
            for (size_t i = 0; i < db.rank - 1; i++) db = wef::reducesum(db, i);
            B = B - db * lr / dy.shape[0];
        }

        return &dx;
    }

Tensor*  Conv2D::forward_pass_legacy(const Tensor& px, const bool training) 
    {
    if (!init) 
    {   
        // initially initilize the shape of X later just copy the tensors
        X = Tensor(px);
        
        // h, w, c, units
        height = px.shape[1]; width = px.shape[2]; ch = px.shape[3];
        dist = std::normal_distribution<float>(0.0f, std::sqrt( 2.0f / (w_height * w_width * ch)));
        // for above now we have (2/fan_in = hwc)^0.5 good for relu we can use fan_out for tanh... which is hwu

        size_t w_shape[4] = {w_height, w_width, ch, units};
        W = Tensor::create(w_shape, 4);

        size_t B_shape[4] = {1, 1, 1, units};
        B = Tensor::create(B_shape, 4);
        std::fill_n(B.tensor.get(), B.tot_size, 0.01f);

        float* pm = W.tensor.get();
        for (size_t i = 0; i < W.tot_size; i++) pm[i] = dist(g);

        m_out_rank = px.rank; // this is 4, its always 4
        m_out_shape = std::make_unique<size_t[]>(m_out_rank); // heap allocation is not the best but we only do this once pre layer so its whatever
        m_out_shape[1] = height - w_height + 1;
        m_out_shape[2] = width - w_width + 1;
        m_out_shape[3] = units;

        // gradient wrt the layer below
        dx = Tensor(px);
        
        // gradients wrt weights and biases
        dw = Tensor(W);
        db = Tensor(B);

        m_num_param = W.tot_size + (usebias ? B.tot_size : 0);

        init = true;
    }
    else
    {
        // if trying to use (reuse) the layer on a different tensor
        if (px.shape[1] != height || 
            px.shape[2] != width ||
            px.shape[3] != ch) throw std::invalid_argument("cannot reuse layer");
    }

    // copy px into X
    if (training) std::memcpy(X.tensor.get(), px.tensor.get(), X.tot_size * sizeof(float));

    m_out_shape[0] = px.shape[0]; // flexable batch 
    out = Tensor::create(m_out_shape.get(), 4);
    float* pm_out = out.tensor.get();
    float* pm_b = B.tensor.get();

    size_t ind = 0;
    size_t i1[4];
    size_t i2[4];

    #pragma omp parallel for collapse(4) schedule(static)
    for (size_t b1 = 0; b1 < out.shape[0]; b1++)
    {
        for (size_t h1 = 0; h1 < out.shape[1]; h1++)
        {
            for (size_t w1 = 0; w1 < out.shape[2]; w1++)
            {
                for (size_t oc = 0; oc < out.shape[3]; oc++)    
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

                                temp += px.index(i1) * W.index(i2);
                            }
                        }
                    }
                    pm_out[ind++] = temp + pm_b[oc]; // TO DO: check BIAS
                }
            }
        }
    }

    return &out;
    }

Tensor*  Conv2D::backward_pass_legacy(const Tensor& dy, const float lr) 
    {
        std::memset(dx.tensor.get(), 0, (dx.tot_size) * sizeof(float)); // zero fill
        std::memset(dw.tensor.get(), 0, (dw.tot_size) * sizeof(float)); // zero fill

        size_t ind = 0;
        size_t i1[4];
        size_t i2[4];

        #pragma omp parallel for collapse(4) schedule(static)
        for (size_t b1 = 0; b1 < dy.shape[0]; b1++)
        {
            for (size_t h1 = 0; h1 < dy.shape[1]; h1++)
            {
                for (size_t w1 = 0; w1 < dy.shape[2]; w1++)
                {
                    for (size_t oc = 0; oc < dy.shape[3]; oc++)
                    {
                        float grad = dy.tensor[ind++];
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

                                    dx.index(i1) += grad * W.index(i2);
                                    dw.index(i2) += grad * X.index(i1);
                                }
                            }
                        }
                    }
                }
            }
        }
        
        float* pm_w = W.tensor.get();
        float* pm_dw = dw.tensor.get();
        // divide lr by batch size
        for (size_t i = 0; i < W.tot_size; i++) pm_w[i] = pm_w[i] - (pm_dw[i] * lr / dy.shape[0]);

        if (usebias)
        {
            db = dy;
            for (size_t i = 0; i < db.rank - 1; i++) db = wef::reducesum(db, i);
            B = B - db * lr / dy.shape[0];
        }

        return &dx;
    }

Tensor*  MaxPool2D::forward_pass(const Tensor& px, const bool training) 
    {
        if (!init)
        {   
            // initially initilize the shape of X later just copy the tensors
            X = Tensor(px);
            // h, w, c, units
            height = px.shape[1]; width = px.shape[2]; ch = px.shape[3];

            size_t ax1 = (height + (height%k_height)) / k_height;
            size_t ax2 = (width + (width%k_width)) / k_width;

            size_t o_size = px.shape[0] * ax1 * ax2 * ch;
            
            // this get the argmax in a nested for loop (2D) I made it flat for speed
            argmax = std::make_unique<size_t[]>(o_size * 4);

            m_out_rank = px.rank; // this is 4, its always 4
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
            if (px.shape[1] != height || 
                px.shape[2] != width ||
                px.shape[3] != ch) throw std::invalid_argument("cannot reuse layer");
        }

        // copy px into X
        if (training) std::memcpy(X.tensor.get(), px.tensor.get(), X.tot_size * sizeof(float));

        // batch is flexable
        m_out_shape[0] = px.shape[0];
        out = Tensor::create(m_out_shape.get(), 4);

        size_t ind = 0;
        size_t i1[4];

        #pragma omp parallel for collapse(4) schedule(static)
        for (size_t b1 = 0; b1 < out.shape[0]; b1++)
        {
            for (size_t h1 = 0; h1 < out.shape[1]; h1++)
            {
                for (size_t w1 = 0; w1 < out.shape[2]; w1++)
                {
                    for (size_t c = 0; c < out.shape[3]; c++)    
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
                                float val = px.index(i1);

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
                        out.tensor[ind] = temp_val;

                        // only populate argmax during training for speed, idt it affects values if we keep it tho
                        if (training) for (size_t ii = 0; ii < 4; ii++) argmax[ind * 4 + ii] = temp_ind[ii];
                        ind++;
                    }
                }
            }
        }
        return &out;
    }

Tensor*  MaxPool2D::backward_pass(const Tensor& dy, const float lr) 
    {
        std::memset(dx.tensor.get(), 0, (dx.tot_size) * sizeof(float));  // zero fill
        size_t ind = 0;
        size_t i1[4];

        #pragma omp parallel for collapse(4) schedule(static)
        for (size_t b1 = 0; b1 < dy.shape[0]; b1++)
        {
            for (size_t h1 = 0; h1 < dy.shape[1]; h1++)
            {
                for (size_t w1 = 0; w1 < dy.shape[2]; w1++)
                {
                    for (size_t c = 0; c < dy.shape[3]; c++)    
                    {
                        i1[0] = argmax[ind * 4 + 0]; i1[1] = argmax[ind * 4 + 1]; 
                        i1[2] = argmax[ind * 4 + 2]; i1[3] = argmax[ind * 4 + 3];
                        dx.index(i1) = dy.tensor[ind];
                        ind++; 
                    }
                }
            }
        }
        return &dx;
    }

Tensor*  ReduceSum::forward_pass(const Tensor& px, const bool training) 
    { 

        if (!init) 
        {
            // initially initilize the shape of X later just copy the tensors
            X = Tensor(px);

            if (ax >= px.rank) throw std::invalid_argument("axis outside shape");
            const size_t* shape = px.shape.get(); // [b, h, w, c]

            m_out_rank = keepdims ? px.rank : px.rank - 1;
            m_out_shape = std::make_unique<size_t[]>(m_out_rank);  // [b, h, w] or  // [b, h, w, 1]
            
            // index 0 will be reassigned if ax > 0 so as to be flexable with the batch
            size_t j = 0;
            for (size_t i = 0; i < px.rank; i++)
            {
                if (i != ax) m_out_shape[j++] = shape[i]; 
                else if (keepdims) m_out_shape[j++] = 1;
            }

            dx = Tensor(px);
            
            init = true;
        }

        // copy px into X
        if (training) std::memcpy(X.tensor.get(), px.tensor.get(), X.tot_size * sizeof(float));

        // batch flexability only applies if ax != 0
        if (ax != 0)
            m_out_shape[0] = px.shape[0];

        out = Tensor::create(m_out_shape.get(), m_out_rank);

        const float* pm = px.tensor.get();
        float* pm_out = out.tensor.get();

        size_t eaa = 1; // everything after axis i.e. b, h w, axis, x1, x2 -> eaa = x1 * x2
        for (size_t i = ax + 1; i < px.rank; i++) eaa *= px.shape[i];

        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < out.tot_size; i++)
        {
            float temp = 0.0f;
            size_t mult = (i/eaa) * (1 - px.shape[ax]);

            for (size_t j = 0; j < px.shape[ax]; j++)
                temp += pm[i  + eaa * (j - mult)];

            pm_out[i] = temp;
        }

        return &out;
    }

Tensor*  ReduceSum::backward_pass(const Tensor& dy, float) 
    {
        if (!init) throw std::invalid_argument("layer not initilized");

        const float* pdy = dy.tensor.get();
        float* pdx = dx.tensor.get(); // no need to zero fill cause we dont do +=

        size_t eaa = 1;
        for (size_t i = ax + 1; i < dx.rank; i++) eaa *= dx.shape[i];

        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < dy.tot_size; i++)
        {
            size_t mult = (i/eaa) * (1 - dx.shape[ax]) ;
            for (size_t j = 0; j < dx.shape[ax]; j++)
            {
                pdx[i  + eaa * (j - mult)] = pdy[i];
            }
        }
        return &dx;
    }

Tensor*  LayerNorm::forward_pass(const Tensor& px, const bool training)
{
    if (!init) 
        {
            // initially initilize the shape of X later just copy the tensors
            X = Tensor(px);

            ax_val = px.shape[axis];
            std::unique_ptr<size_t[]> beta_shape = std::make_unique<size_t[]>(px.rank);
            std::unique_ptr<size_t[]> gamma_shape = std::make_unique<size_t[]>(px.rank);

            // fill with 1s
            std::fill_n(beta_shape.get(), px.rank, 1);
            std::fill_n(gamma_shape.get(), px.rank, 1);

            beta_shape[axis] = px.shape[axis];
            gamma_shape[axis] = px.shape[axis];

            beta = Tensor::create(beta_shape.get(), px.rank);
            gamma = Tensor::create(gamma_shape.get(), px.rank);

            // initilize beta and gamma
            std::fill_n(beta.tensor.get(), ax_val, 0.01f);
            std::fill_n(gamma.tensor.get(), ax_val, 0.99f);

            m_num_param = beta.tot_size + gamma.tot_size;

            m_out_rank = px.rank;
            m_out_shape = std::make_unique<size_t[]>(m_out_rank);
            std::memcpy(m_out_shape.get(), px.shape.get(), m_out_rank * sizeof(size_t));
            
            init = true;
        }
        if (px.shape[axis] != ax_val) throw std::invalid_argument("cannot reuse layer [LayerNorm]");

        // copy px into X
        if (training) std::memcpy(X.tensor.get(), px.tensor.get(), X.tot_size * sizeof(float));

        // follwoing Ba et al. 2016
        mu = wef::reducesum(px, /*axis=*/axis) / ax_val;
        x_mu = px - mu;
        var = wef::reducesum(x_mu * x_mu, /*axis=*/axis) / ax_val;
        inv_std = wef::pow(var + eps, -0.5f);
        x_i_hat = x_mu * inv_std;
        y_i = x_i_hat * gamma + beta;

        return &y_i;
}

Tensor*  LayerNorm::backward_pass(const Tensor& dy, const float lr)
{
    if (!init) throw std::invalid_argument("layer not initilized");

    d_gamma = dy * x_i_hat;
    d_beta = dy;
    for (size_t i = 0; i < dy.rank; i++)
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

    gamma = gamma - d_gamma * lr / dy.shape[0];
    beta = beta - d_beta * lr / dy.shape[0];

    return &dx;
}