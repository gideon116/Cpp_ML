
#include <iostream>
#include <random>
#include "../include/layers.h"

Tensor* Conv2D_Fast::forward_pass(const Tensor& px, const bool training, void*) 
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
    size_t skip_h_help = (w_height - 1) * px.m_shape[px.m_rank-2]*px.m_shape[px.m_rank-1];
    size_t offset = width - w_width;
    size_t id_help = w_width * ch;

    // multi thread additions
    size_t avaliable_threads = std::thread::hardware_concurrency(); // may be 0
    size_t n_threads = std::min<size_t>( out_wo_units,  avaliable_threads > 0 ? avaliable_threads : 1 );
    
    const size_t stride = out_wo_units / n_threads;
    const size_t rem = out_wo_units % n_threads;

    // spin up
    std::thread* threads = new std::thread[n_threads];
    for (size_t th = 0; th < n_threads; th++)
    {
        size_t temp = (th < n_threads - 1) ? stride : stride + rem;
        threads[th] = std::thread(

            // we dont want to capture everything in scope !
            [
                th, stride, temp, skip_w_help, bi_help, skip_h_help, px_ptr, offset, id_help,
                out_ptr, W_ptr, B_ptr
            ]
            (size_t Wsize, size_t outrow, size_t units, size_t ch, bool m_use_bias)
            {
                for (size_t out_i = th * stride; out_i < (th * stride) + temp; out_i++)
                {
                    size_t skip_w = skip_w_help * (out_i / outrow);
                    size_t bi = out_i / bi_help;
                    size_t skip_h = bi * skip_h_help;

                    for (size_t w_i = 0; w_i < Wsize / units; w_i++)
                    {
                        float temp_px = px_ptr[
                            ch * out_i + skip_w + skip_h
                            + 
                            w_i + ch * offset * (w_i / id_help)
                        ];

                        for (size_t u_i = 0; u_i < units; u_i++)
                            // TODO : data races
                            out_ptr[out_i * units + u_i] += temp_px * W_ptr[w_i * units + u_i];
                    }
                    if (m_use_bias)
                        for (size_t u_i = 0; u_i < units; u_i++)
                            out_ptr[out_i * units + u_i] += B_ptr[u_i];
                }
            },
            
            // pass these are parameters cause we dont want to copy the entire tensor
            W.m_size, out.m_shape[out.m_rank-2], units, ch, m_use_bias);
    }

    // free
    for (size_t i = 0; i < n_threads; i++) threads[i].join();
    
    // clean up
    delete[] threads;
    
    return &out;
}

Tensor* Conv2D_Fast::backward_pass(const Tensor& dy, const float lr, void*) 
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

    // multi thread additions
    size_t avaliable_threads = std::thread::hardware_concurrency(); // may be 0
    size_t n_threads = std::min<size_t>( out_wo_units,  avaliable_threads > 0 ? avaliable_threads : 1 );

    // make temp variables for each thread cause we should not do += at the same index in different threads
    std::unique_ptr<Tensor[]> dx_accum_pre_thread = std::make_unique<Tensor[]>(n_threads);
    std::unique_ptr<Tensor[]> dw_accum_pre_thread = std::make_unique<Tensor[]>(n_threads);
    for (size_t i = 0; i < n_threads; i++)
    {
        dx_accum_pre_thread[i] = dx; dw_accum_pre_thread[i] = dw;
    }
    
    const size_t stride = out_wo_units / n_threads;
    const size_t rem = out_wo_units % n_threads;

    // spin up
    std::thread* threads = new std::thread[n_threads];
    for (size_t th = 0; th < n_threads; th++)
    {
        size_t temp = (th < n_threads - 1) ? stride : stride + rem;
        threads[th] = std::thread(
            // we dont want to capture everything in scope !
            [
                th, stride, temp, skip_w_help, bi_help, skip_h_help, id_help, offset, 
                dy_ptr, X_ptr, W_ptr
            ]
            (size_t Wsize, size_t dyrow, size_t units, size_t ch, float* dx_i, float* dw_i)
            {
                for (size_t dy_i = th * stride; dy_i < (th * stride) + temp; dy_i++)
                {
                    size_t skip_w = skip_w_help * (dy_i / dyrow);
                    size_t bi = dy_i / bi_help;
                    size_t skip_h = bi * skip_h_help;

                    for (size_t w_i = 0; w_i < Wsize / units; w_i++)
                    {
                        size_t id1 = 
                            ch * dy_i + skip_w + skip_h
                            + 
                            w_i + ch*offset * (w_i / id_help);

                        for (size_t u_i = 0; u_i < units; u_i++)
                        {
                            float grad = dy_ptr[dy_i * units + u_i];
                            // data races TODO
                            dx_i[id1] += grad * W_ptr[w_i * units + u_i];
                            dw_i[w_i * units + u_i] += grad * X_ptr[id1];
                        }
                    }
                }
            },
        // pass these are parameters cause we dont want to copy the entire tensor
        W.m_size, dy.m_shape[dy.m_rank-2], units, ch, dx_accum_pre_thread[th].m_tensor, dw_accum_pre_thread[th].m_tensor);
    }

    // free
    for (size_t i = 0; i < n_threads; i++) threads[i].join();

    // clean up
    delete[] threads;

    // aggrigate the temp variables from each thread
    for (size_t i = 0; i < n_threads; i++)
    {
        for (size_t v = 0; v < dw.m_size; v++) dw_ptr[v] += dw_accum_pre_thread[i].m_tensor[v];
        for (size_t v = 0; v < dx.m_size; v++) dx_ptr[v] += dx_accum_pre_thread[i].m_tensor[v];
    }

    // divide lr by batch size
    W -= dw * lr /dy.m_shape[0];

    if (m_use_bias)
    {
        db = dy;
        for (size_t i = 0; i < db.m_rank - 1; i++)
            db = wef::reducesum(db, i);
        B -= db * lr / dy.m_shape[0];
    }

    return &dx;
}

Tensor* Linear_Fast::forward_pass(const Tensor& px, const bool training, void*) 
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

    }

Tensor* Linear_Fast::backward_pass(const Tensor& dy, const float lr, void*) 
    {
        // gradient wrt the layer below
        dx = wef::matmul(dy, wef::transpose(W), true);

        // gradient wrt weights sum everything aside from the last two axes. 
        // CATCH rank < 2?????
        dw = wef::matmul(wef::transpose(X), dy, /*threads=*/true);
        for (size_t i = 0; i < dw.m_rank - 2; i++)
            dw = wef::reducesum(dw, i);

        W -= dw * lr / dy.m_shape[0];

        if (m_use_bias) 
        {
            // gradient wrt bias sum everything aside from the last axis
            db = dy;
            for (size_t i = 0; i < db.m_rank - 1; i++)
                db = wef::reducesum(db, i);
            B -= db * lr / dy.m_shape[0];
        }

        return &dx;
    }
