
#include <iostream>
#include <random>
#include "layers.h"

Tensor Conv2D_Fast::forward_pass(const Tensor& px, const bool training) 
{
    if (!init) 
    {   
        // initially initilize the shape of X later just copy the tensors
        X = Tensor(px);

        // h, w, c, units
        height = px.shape[1]; width = px.shape[2]; ch = px.shape[3];
        dist = std::normal_distribution<double>(0.0, std::sqrt( 2.0 / (w_height * w_width * ch))); 
        // for above now we have (2/fan_in = hwc)^0.5 good for relu we can use fan_out for tanh... which is hwu

        int w_shape[4] = {w_height, w_width, ch, units};
        W = Tensor::create(w_shape, 4);

        int B_shape[4] = {1, 1, 1, units};
        B = Tensor::create(B_shape, 4);
        std::fill_n(B.tensor.get(), B.tot_size, 0.01);

        double* pm = W.tensor.get();
        for (size_t i = 0; i < W.tot_size; i++) pm[i] = dist(g);

        int out_shape[4] = {px.shape[0], height - w_height + 1, width - w_width + 1, units};
        out = Tensor::create(out_shape, 4);

        // gradient wrt the layer below
        dx = Tensor(px);

        // gradient wrt weights
        dw = Tensor(W);

        db = Tensor(B);
        m_num_param = W.tot_size + (usebias ? B.tot_size : 0);
        // for (int i=0; i < 4; i++) m_out_shape[i] = 0;
        std::memcpy(m_out_shape, out_shape, 4 * sizeof(int));
        m_out_rank = 4;
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
    if (training) std::memcpy(X.tensor.get(), px.tensor.get(), X.tot_size * sizeof(double));
    
    double* out_ptr = out.tensor.get();
    double* px_ptr = px.tensor.get();
    double* W_ptr = W.tensor.get();
    double* B_ptr = B.tensor.get();

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

    // multi thread additions
    int avaliable_threads = std::thread::hardware_concurrency(); // may be 0
    int n_threads = std::min<int>( out_wo_units,  avaliable_threads > 0 ? avaliable_threads : 1 );
    
    const int stride = out_wo_units / n_threads;
    const int rem = out_wo_units % n_threads;

    // spin up
    std::thread* threads = new std::thread[n_threads];
    for (int th = 0; th < n_threads; th++)
    {
        int temp = (th < n_threads - 1) ? stride : stride + rem;
        threads[th] = std::thread(

            // we dont want to capture everything in scope !
            [
                th, stride, temp, skip_w_help, bi_help, skip_h_help, px_ptr, offset, id_help,
                out_ptr, W_ptr, B_ptr
            ]
            (size_t Wtot_size, size_t outrow, int units, int ch, bool usebias)
            {
                for (int out_i = th * stride; out_i < (th * stride) + temp; out_i++)
                {
                    int skip_w = skip_w_help * (out_i / outrow);
                    int bi = out_i / bi_help;
                    int skip_h = bi * skip_h_help;

                    for (int w_i = 0; w_i < Wtot_size / units; w_i++)
                    {
                        double temp_px = px_ptr[
                            ch * out_i + skip_w + skip_h
                            + 
                            w_i + ch*offset * (w_i / id_help)
                        ];

                        for (int u_i = 0; u_i < units; u_i++)
                            out_ptr[out_i * units + u_i] += temp_px * W_ptr[w_i * units + u_i];
                    }
                    if (usebias)
                        for (int u_i = 0; u_i < units; u_i++)
                            out_ptr[out_i * units + u_i] += B_ptr[u_i];
                }
            },
            
            // pass these are parameters cause we dont want to copy the entire tensor
            W.tot_size, out.row, units, ch, usebias);
    }

    // free
    for (int i = 0; i < n_threads; i++) threads[i].join();
    

    // clean up
    delete[] threads;
    
    return out;
}

Tensor Conv2D_Fast::backward_pass(const Tensor& dy, const double lr) 
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
    int bi_help = out_wo_units / dy.shape[0];
    int skip_h_help = (w_height - 1) * X.row*X.col;
    int offset = width - w_width;
    int id_help = w_width * ch;

    // multi thread additions
    int avaliable_threads = std::thread::hardware_concurrency(); // may be 0
    int n_threads = std::min<int>( out_wo_units,  avaliable_threads > 0 ? avaliable_threads : 1 );

    // make temp variables for each thread cause we should not do += at the same index in different threads
    std::unique_ptr<Tensor[]> dx_accum_pre_thread = std::make_unique<Tensor[]>(n_threads);
    std::unique_ptr<Tensor[]> dw_accum_pre_thread = std::make_unique<Tensor[]>(n_threads);
    for (int i = 0; i < n_threads; i++)
    {
        dx_accum_pre_thread[i] = dx; dw_accum_pre_thread[i] = dw;
    }
    
    const int stride = out_wo_units / n_threads;
    const int rem = out_wo_units % n_threads;

    // spin up
    std::thread* threads = new std::thread[n_threads];
    for (int th = 0; th < n_threads; th++)
    {
        int temp = (th < n_threads - 1) ? stride : stride + rem;
        threads[th] = std::thread(
            // we dont want to capture everything in scope !
            [
                th, stride, temp, skip_w_help, bi_help, skip_h_help, id_help, offset, 
                dy_ptr, X_ptr, W_ptr
            ]
            (size_t Wtot_size, size_t dyrow, int units, int ch, double* dx_i, double* dw_i)
            {
                for (int dy_i = th * stride; dy_i < (th * stride) + temp; dy_i++)
                {
                    int skip_w = skip_w_help * (dy_i / dyrow);
                    int bi = dy_i / bi_help;
                    int skip_h = bi * skip_h_help;

                    for (int w_i = 0; w_i < Wtot_size / units; w_i++)
                    {
                        int id1 = 
                            ch * dy_i + skip_w + skip_h
                            + 
                            w_i + ch*offset * (w_i / id_help);

                        for (int u_i = 0; u_i < units; u_i++)
                        {
                            double grad = dy_ptr[dy_i * units + u_i];
                            dx_i[id1] += grad * W_ptr[w_i * units + u_i];
                            dw_i[w_i * units + u_i] += grad * X_ptr[id1];
                        }
                    }
                }
            },
        // pass these are parameters cause we dont want to copy the entire tensor
        W.tot_size, dy.row, units, ch, dx_accum_pre_thread[th].tensor.get(), dw_accum_pre_thread[th].tensor.get());
    }

    // free
    for (int i = 0; i < n_threads; i++) threads[i].join();

    // clean up
    delete[] threads;

    // aggrigate the temp variables from each thread
    for (int i = 0; i < n_threads; i++)
    {
        for (size_t v = 0; v < dw.tot_size; v++) dw_ptr[v] += dw_accum_pre_thread[i].tensor.get()[v];
        for (size_t v = 0; v < dx.tot_size; v++) dx_ptr[v] += dx_accum_pre_thread[i].tensor.get()[v];
    }

    // divide lr by batch size
    for (size_t i = 0; i < W.tot_size; i++) W_ptr[i] -= dw_ptr[i] * lr /dy.shape[0];

    if (usebias)
    {
        db = dy;
        for (int i = 0; i < db.rank - 1; i++) db = wef::reducesum(db, i);
        B = B - db * lr / dy.shape[0];
    }

    return dx;
}

Tensor Linear_Fast::forward_pass(const Tensor& px, const bool training) 
    {
        if (!init) 
        {   
            // initially initilize the shape of X later just copy the tensors
            X = Tensor(px);

            int w_shape[2] = {px.col, units};
            int b_shape[2] = {1, units};
            W = Tensor::create(w_shape, 2);
            B = Tensor::create(b_shape, 2);

            double* B_ptr = B.tensor.get();
            std::fill_n(B.tensor.get(), B.tot_size, 0.01); // zero fill

            double* pm = W.tensor.get();
            for (size_t i = 0; i < size_t(px.col) * units; i++) pm[i] = dist(g);

            m_num_param = W.tot_size + (usebias ? B.tot_size : 0);
            //////// REMOVE
            std::memcpy(m_out_shape,  wef::matmul(px, W).shape.get(), wef::matmul(px, W).rank * sizeof(int));
            m_out_rank = wef::matmul(px, W).rank;
            init = true;
        }
        else
        {
            // if trying to use (reuse) the layer on a different tensor
            if (W.row != px.col) throw std::invalid_argument("cannot reuse layer");
        }
        
        // copy px into X
        if (training) std::memcpy(X.tensor.get(), px.tensor.get(), X.tot_size * sizeof(double));

        if (usebias) return wef::matmul(px, W, true);
        return wef::matmul(px, W, true);
    }

Tensor Linear_Fast::backward_pass(const Tensor& dy, const double lr) 
    {
        // gradient wrt the layer below
        dx = wef::matmul(dy, wef::transpose(W), true);

        // gradient wrt weights sum everything aside from the last two axes. 
        // CATCH rank < 2?????
        dw = wef::matmul(wef::transpose(X), dy, /*threads=*/true);
        for (int i = 0; i < dw.rank - 2; i++) dw = wef::reducesum(dw, i);

        W = W - dw * lr / dy.shape[0];

        if (usebias) 
        {
            // gradient wrt bias sum everything aside from the last axis
            db = dy;
            for (int i = 0; i < db.rank - 1; i++) db = wef::reducesum(db, i);
            B = B - db * lr / dy.shape[0];
        }
        return dx;
    }


