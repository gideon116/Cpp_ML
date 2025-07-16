
#include <iostream>
#include <random>
#include "layers.h"


Tensor Linear::forward_pass(const Tensor& px) 
    {
        if (!init) 
        {   
            int w_shape[2] = {px.col, units};
            int b_shape[2] = {1, units};
            W = Tensor::create(w_shape, 2);
            B = Tensor::create(b_shape, 2);

            double* B_ptr = B.tensor.get();
            std::memset(B_ptr, 0, (B.tot_size) * sizeof(double)); // zero fill

            double* pm = W.tensor.get();
            for (size_t i = 0; i < size_t(px.col) * units; i++) pm[i] = dist(g);
            init = true;
        }
        else
        {
            // if trying to use (reuse) the layer on a different tensor
            if (W.row != px.col) throw std::invalid_argument("cannot reuse layer");
        }
        X = Tensor(px);
        return wef::matmul(px, W, true) + B;
    }

Tensor Linear::backward_pass(const Tensor& dy, const double lr) 
    {
        // gradient wrt the layer below
        dx = wef::matmul(dy, wef::transpose(W), true);

        // gradient wrt weights
        dw = wef::batchsum(wef::matmul(wef::transpose(X), dy, /*threads=*/true));

        // gradient wrt bias
        db = wef::reducesum(wef::batchsum((dy)), /*axis=*/0);

        W = W - dw * lr / dy.shape[0];
        B = B - db * lr / dy.shape[0];

        return dx;
    }

Tensor Conv2D::forward_pass_multi(const Tensor& px) 
    {
    if (!init) 
    {   
        // h, w, c, units
        height = px.shape[1]; width = px.shape[2]; ch = px.shape[3];
        dist = std::normal_distribution<double>(0.0, 1.0/std::sqrt(w_height * w_width * ch));
        
        int w_shape[4] = {w_height, w_width, ch, units};
        W = Tensor::create(w_shape, 4);
        double* pm = W.tensor.get();
        for (size_t i = 0; i < W.tot_size; i++) pm[i] = dist(g);

        int out_shape[4] = {px.shape[0], height - w_height + 1, width - w_width + 1, units};
        out = Tensor::create(out_shape, 4);

        // gradient wrt the layer below
        dx = Tensor(px);
        
        // gradient wrt weights
        dw = Tensor(W);

        init = true;
    }
    else
    {
        // if trying to use (reuse) the layer on a different tensor
        if (px.shape[1] != height || 
            px.shape[2] != width ||
            px.shape[3] != ch) throw std::invalid_argument("cannot reuse layer");
    }

    X = Tensor(px);

    size_t ind = 0;
    size_t i1[4];
    size_t i2[4];
    for (size_t b1 = 0; b1 < out.shape[0]; b1++)
    {
        for (size_t h1 = 0; h1 < out.shape[1]; h1++)
        {
            for (size_t w1 = 0; w1 < out.shape[2]; w1++)
            {
                for (size_t oc = 0; oc < out.shape[3]; oc++)    
                {
                    double temp = 0;
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
                    out.tensor[ind++] = temp;
                }
            }
        }
    }

    return out;
    }

Tensor Conv2D::forward_pass/*_multi*/(const Tensor& px) 
    {
    if (!init) 
    {   
        // h, w, c, units
        height = px.shape[1]; width = px.shape[2]; ch = px.shape[3];
        dist = std::normal_distribution<double>(0.0, 1.0/std::sqrt(w_height * w_width * ch));

        int w_shape[4] = {w_height, w_width, ch, units};
        W = Tensor::create(w_shape, 4);

        double* pm = W.tensor.get();
        for (size_t i = 0; i < W.tot_size; i++) pm[i] = dist(g);

        int out_shape[4] = {px.shape[0], height - w_height + 1, width - w_width + 1, units};
        out = Tensor::create(out_shape, 4);

        // gradient wrt the layer below
        dx = Tensor(px);

        // gradient wrt weights
        dw = Tensor(W);
        
        init = true;
    }
    else
    {
        // if trying to use (reuse) the layer on a different tensor
        if (px.shape[1] != height || 
            px.shape[2] != width ||
            px.shape[3] != ch) throw std::invalid_argument("cannot reuse layer");
    }

    X = Tensor(px);
    
    double* out_ptr = out.tensor.get();
    double* px_ptr = px.tensor.get();
    double* W_ptr = W.tensor.get();

    std::memset(out_ptr, 0, (out.tot_size) * sizeof(double));

    /*
    There is a lot of math below but the idea is to do the cov kernel math (W * Input) and expand 
    this to the units dimension by repeating the index of Input as before.

    Here what I do is index out out and W in contigous manner to make it faster and for Input
    I jump everytime we hit the width of the weight to the second row and use some math to do that.
    */

    int out_wo_units = out.tot_size / units;
    int skip_w_help = ch * (w_width - 1);
    int bi_help = out_wo_units / out.shape[0];
    int skip_h_help = (w_height - 1) * px.row*px.col;
    int offset = width - w_width;
    int id_help = w_width * ch;

    for (int out_i = 0; out_i < out_wo_units; out_i++)
    {
        int skip_w = skip_w_help * (out_i / out.row);
        int bi = out_i / bi_help;
        int skip_h = bi * skip_h_help;

        for (int w_i = 0; w_i < W.tot_size / units; w_i++)
        {
            double temp_px = px_ptr[
                ch * out_i + skip_w + skip_h
                + 
                w_i + ch*offset * (w_i / id_help)
            ];

            for (int u_i = 0; u_i < units; u_i++)
                out_ptr[out_i * units + u_i] += temp_px * W_ptr[w_i * units + u_i];
        }
    }
        

    /*
    
    // multi thread additions
    int avaliable_threads = std::thread::hardware_concurrency(); // may be 0
    int n_threads = std::min<int>( out.tot_size,  avaliable_threads > 0 ? avaliable_threads : 1 );
    
    const int stride = out.tot_size / n_threads;
    const int rem = out.tot_size % n_threads;

    // spin up
    std::thread* threads = new std::thread[n_threads];
    */

    return out;
    }

Tensor Conv2D::backward_pass/*_multi*/(const Tensor& dy, const double lr) 
    {   
        double* dx_ptr = dx.tensor.get();
        double* dw_ptr = dw.tensor.get();

        std::memset(dx_ptr, 0, (dx.tot_size) * sizeof(double)); // zero fill
        std::memset(dw_ptr, 0, (dw.tot_size) * sizeof(double)); // zero fill

        double* dy_ptr = dy.tensor.get();
        double* W_ptr = W.tensor.get();
        double* X_ptr = X.tensor.get();

        int out_wo_units = dy.tot_size / units;
        int skip_w_help = ch * (w_width - 1);
        int bi_help = out_wo_units / out.shape[0];
        int skip_h_help = (w_height - 1) * X.row*X.col;
        int offset = width - w_width;
        int id_help = w_width * ch;


        for (int dy_i = 0; dy_i < out_wo_units; dy_i++)
        {
            int skip_w = skip_w_help * (dy_i / dy.row);
            int bi = dy_i / bi_help;
            int skip_h = bi * skip_h_help;

            for (int w_i = 0; w_i < W.tot_size / units; w_i++)
            {
                int id1 = 
                    ch * dy_i + skip_w + skip_h
                    + 
                    w_i + ch*offset * (w_i / id_help);

                for (int u_i = 0; u_i < units; u_i++)
                {
                    double grad = dy_ptr[dy_i * units + u_i];
                    dx_ptr[id1] += grad * W_ptr[w_i * units + u_i];
                    dw_ptr[w_i * units + u_i] += grad * X_ptr[id1];
                }
            }
        }

        // divide lr by batch size
        for (size_t i = 0; i < W.tot_size; i++) W_ptr[i] = W_ptr[i] - (dw_ptr[i] * lr /dy.shape[0]);

        return dx;
    }

Tensor Conv2D::backward_pass_multi(const Tensor& dy, const double lr) 
    {
        std::memset(dx.tensor.get(), 0, (dx.tot_size) * sizeof(double)); // zero fill
        std::memset(dw.tensor.get(), 0, (dw.tot_size) * sizeof(double)); // zero fill

        size_t ind = 0;
        size_t i1[4];
        size_t i2[4];
        for (size_t b1 = 0; b1 < dy.shape[0]; b1++)
        {
            for (size_t h1 = 0; h1 < dy.shape[1]; h1++)
            {
                for (size_t w1 = 0; w1 < dy.shape[2]; w1++)
                {
                    for (size_t oc = 0; oc < dy.shape[3]; oc++)
                    {
                        double grad = dy.tensor[ind++];
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
        // divide lr by batch size
        for (size_t i = 0; i < W.tot_size; i++) W.tensor[i] = W.tensor[i] - (dw.tensor[i] * lr / dy.shape[0]);

        return dx;
    }


Tensor MaxPool2D::forward_pass(const Tensor& px) 
    {
        if (!init)
        {   
            // h, w, c, units
            height = px.shape[1]; width = px.shape[2]; ch = px.shape[3];

            size_t o_size = (px.shape[0]) * ((height + (height%k_height)) / k_height) * ((width + (width%k_width)) / k_width) * (ch);
            
            // this get the argmax in a nested for loop (2D) I made it flat for speed
            argmax = std::make_unique<size_t[]>(o_size * 4);

            int out_shape[4] = {px.shape[0], (height + (height%k_height)) / k_height, (width + (width%k_width)) / k_width, ch};
            out = Tensor::create(out_shape, 4);
            
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

        X = Tensor(px);

        size_t ind = 0;
        size_t i1[4];
        for (size_t b1 = 0; b1 < out.shape[0]; b1++)
        {
            for (size_t h1 = 0; h1 < out.shape[1]; h1++)
            {
                for (size_t w1 = 0; w1 < out.shape[2]; w1++)
                {
                    for (size_t c = 0; c < out.shape[3]; c++)    
                    {
                        double temp_val = -1e19;
                        size_t temp_ind[4];
                        for (size_t h2 = h1 * k_height; h2 < h1 * k_height + k_height; h2++)
                        {
                            if (h2 >= height) break;
                            for (size_t w2 = w1 * k_width; w2 < w1 * k_width + k_width; w2++)
                            {
                                if (w2 >= width) break;

                                i1[0] = b1; i1[1] = h2; i1[2] = w2; i1[3] = c;
                                double val = px.index(i1);

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
                        for (int ii = 0; ii < 4; ii++) argmax[ind * 4 + ii] = temp_ind[ii];
                        ind++;
                    }
                }
            }
        }
        return out;
    }

Tensor MaxPool2D::backward_pass(const Tensor& dy, const double lr) 
    {
        std::memset(dx.tensor.get(), 0, (dx.tot_size) * sizeof(double));  // zero fill
        size_t ind = 0;
        size_t i1[4];
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
        return dx;
    }


Tensor ReduceSum::forward_pass(const Tensor& px) 
    { 

        if (!init) 
        {
            if (ax >= px.rank) throw std::invalid_argument("axis outside shape");
            const int* shape = px.shape.get(); // [b, h, w, c]
            reshape_shape = std::make_unique<int[]>(px.rank-2); // [h, w]
            keepdims_shape = std::make_unique<int[]>(px.rank-1);  // [h, w, 1]

            keepdims_rank = px.rank;

            int j = 0;
            for (int i = 1; i < px.rank; i++)
            {
                if (i != ax) { reshape_shape[j++] = shape[i]; keepdims_shape[i - 1] = shape[i]; }
                else keepdims_shape[i - 1] = 1;
            }

            // the whole point of curr_shape is to be flexable with the batch but strict with the other dims
            int curr_shape_kd[keepdims_rank];
            curr_shape_kd[0] = px.shape[0];
            for (int i = 1; i < px.rank; i++) curr_shape_kd[i] = keepdims_shape.get()[i - 1];

            out_keepdims = Tensor::create(curr_shape_kd, keepdims_rank);

            if (!keepdims)
            {
                int curr_shape[keepdims_rank - 1];
                curr_shape[0] = px.shape[0];
                for (int i = 1; i < px.rank - 1; i++) curr_shape[i] = reshape_shape.get()[i - 1];
                out = Tensor::create(curr_shape, keepdims_rank - 1);
            }
            
            dx = Tensor(px);
            
            init = true;
        }

        X = Tensor(px);

        const double* pm = px.tensor.get();
        double* pm_okd = out_keepdims.tensor.get();

        int eaa = 1; // everything after axis i.e. b, h w, axis, x1, x2 -> eaa = x1 * x2
        for (int i = ax + 1; i < px.rank; i++) eaa *= px.shape[i];

        for (size_t i = 0; i < out_keepdims.tot_size; i++)
        {
            double temp = 0;
            int mult = (i/eaa) * (1 - px.shape[ax]) ;
            for (int j = 0; j < px.shape[ax]; j++)
            {
                temp += pm[i  + eaa * (j - mult)];
            }
            pm_okd[i] = temp;
        }

        if (!keepdims)
        {
            double* p_out = out.tensor.get();
            for (size_t i = 0; i < out_keepdims.tot_size; i++) p_out[i] = pm_okd[i];
            return out;
        }

        return out_keepdims;
    }

Tensor ReduceSum::backward_pass(const Tensor& dy, double) 
    {
        if (!init) throw std::invalid_argument("layer not initilized");

        const double* pdy = dy.tensor.get();
        double* pdx = dx.tensor.get();

        int eaa = 1;
        for (int i = ax + 1; i < dx.rank; i++) eaa *= dx.shape[i];

        for (size_t i = 0; i < dy.tot_size; i++)
        {
            int mult = (i/eaa) * (1 - dx.shape[ax]) ;
            for (int j = 0; j < dx.shape[ax]; j++)
            {
                pdx[i  + eaa * (j - mult)] = pdy[i];
            }
        }
        return dx;
    }

Tensor LayerNorm::forward_pass(const Tensor& px)
{
    if (!init) 
        {
            ax_val = px.shape[axis];
            std::unique_ptr<int[]> beta_shape = std::make_unique<int[]>(px.rank);
            std::unique_ptr<int[]> gamma_shape = std::make_unique<int[]>(px.rank);

            // fill with 1s
            std::fill_n(beta_shape.get(), px.rank, 1);
            std::fill_n(gamma_shape.get(), px.rank, 1);

            beta_shape[axis] = px.shape[axis];
            gamma_shape[axis] = px.shape[axis];

            beta = Tensor::create(beta_shape.get(), px.rank);
            gamma = Tensor::create(gamma_shape.get(), px.rank);

            // initilize beta and gamma
            std::fill_n(beta.tensor.get(), ax_val, 0.01);
            std::fill_n(gamma.tensor.get(), ax_val, 0.99);
            
            init = true;
        }
        if (px.shape[axis] != ax_val) throw std::invalid_argument("cannot reuse layer [LayerNorm]");

        X = Tensor(px);

        // follwoing Ba et al. 2016
        mu = wef::reducesum(px, /*axis=*/axis) / ax_val;
        x_mu = px - mu;
        var = wef::reducesum(x_mu * x_mu, /*axis=*/axis) / ax_val;
        inv_std = wef::pow(var + eps, -0.5);
        x_i_hat = x_mu * inv_std;
        y_i = x_i_hat * gamma + beta;

        return y_i;
}

Tensor LayerNorm::backward_pass(const Tensor& dy, const double lr)
{
    if (!init) throw std::invalid_argument("layer not initilized");

    d_gamma = dy * x_i_hat;
    d_beta = dy;
    for (int i = 0; i < dy.rank; i++)
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

    return dx;
}