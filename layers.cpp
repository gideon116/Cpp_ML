
#include <iostream>
#include <random>
#include "layers.h"


Tensor Linear::forward_pass(const Tensor& px, matrixOperations& wf) 
    {
        if (!init) 
        {   
            W = Tensor::create({px.col, units});
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
        return wf.matmul(px, W);
    }

Tensor Linear::backward_pass(const Tensor& dy, const double lr, matrixOperations& wf) 
    {
        // gradient wrt the layer below
        Tensor dx = wf.matmul(dy, wf.transpose(W));

        // gradient wrt weights
        Tensor dw = wf.batchsum(wf.matmul(wf.transpose(X), dy));

        W = W - dw * lr;

        return dx;
    }




Tensor Conv2D::forward_pass(const Tensor& px, matrixOperations& wf) 
    {
    if (!init) 
    {   
        // h, w, c, units
        height = px.shape[1]; width = px.shape[2]; ch = px.shape[3];
        dist = std::normal_distribution<double>(0.0, 1.0/std::sqrt(w_height * w_width * ch));
        
        W = Tensor::create({w_height, w_width, ch, units});
        double* pm = W.tensor.get();
        for (size_t i = 0; i < W.tot_size; i++) pm[i] = dist(g);
        out = Tensor::create({px.shape[0], height - w_height + 1, width - w_width + 1, units});
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
                                temp += px.index({b1, h2 + h1, w2 + w1, c2}) * W.index({h2, w2, c2, oc});
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

Tensor Conv2D::backward_pass(const Tensor& dy, const double lr, matrixOperations& wf) 
    {
        // gradient wrt the layer below
        Tensor dx = Tensor::create({X.shape[0], X.shape[1], X.shape[2], X.shape[3]});
        for (size_t i = 0; i < dx.tot_size; i++) dx.tensor[i] = 0;
        
        // gradient wrt weights
        Tensor dw = Tensor::create({W.shape[0], W.shape[1], W.shape[2], W.shape[3]});
        for (size_t i = 0; i < dw.tot_size; i++) dw.tensor[i] = 0;


        size_t ind = 0;
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
                                    dx.index({b1, h2 + h1, w2 + w1, c2}) += grad * W.index({h2, w2, c2, oc});
                                    dw.index({h2, w2, c2, oc}) += grad * X.index({b1, h2 + h1, w2 + w1, c2});
                                }
                            }
                        }
                    }
                }
            }
        }
        // divide lr by batch size
        for (size_t i = 0; i < W.tot_size; i++) W.tensor[i] = W.tensor[i] - (lr * dw.tensor[i]/dy.shape[0]);

        return dx;
    }


Tensor MaxPool2D::forward_pass(const Tensor& px, matrixOperations& wf) 
    {
        if (!init)
        {   
            // h, w, c, units
            height = px.shape[1]; width = px.shape[2]; ch = px.shape[3];

            size_t o_size = (px.shape[0]) * ((height + (height%k_height)) / k_height) * ((width + (width%k_width)) / k_width) * (ch);
            argmax = std::make_unique<size_t[][4]>(o_size);
            out = Tensor::create({px.shape[0], (height + (height%k_height)) / k_height, (width + (width%k_width)) / k_width, ch});
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
                                double val = px.index({b1, h2, w2, c});
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
                        for (int ii = 0; ii < 4; ii++) argmax[ind][ii] = temp_ind[ii];
                        ind++;
                    }
                }
            }
        }
        return out;
    }

Tensor MaxPool2D::backward_pass(const Tensor& dy, const double lr, matrixOperations& wf) 
    {

        // gradient wrt the layer below
        Tensor dx = Tensor::create({X.shape[0], X.shape[1], X.shape[2], X.shape[3]});
        for (size_t i = 0; i < dx.tot_size; i++) dx.tensor[i] = 0;


        size_t ind = 0;
        for (size_t b1 = 0; b1 < dy.shape[0]; b1++)
        {
            for (size_t h1 = 0; h1 < dy.shape[1]; h1++)
            {
                for (size_t w1 = 0; w1 < dy.shape[2]; w1++)
                {
                    for (size_t c = 0; c < dy.shape[3]; c++)    
                    {
                        dx.index({argmax[ind][0], argmax[ind][1], argmax[ind][2], argmax[ind][3]})
                                    = dy.tensor[ind];
                        ind++; 
                    }
                }
            }
        }
        return dx;
    }


Tensor ReduceSum::forward_pass(const Tensor& px, matrixOperations& wf) 
    { 

        if (!init) 
        {
            if (ax >= px.rank) throw std::invalid_argument("axis outside shape");
            const int* shape = px.shape.get(); // [b, h, w, c]
            reshape_shape = std::make_unique<int[]>(px.rank-1); // [b, w, c]
            keepdims_shape = std::make_unique<int[]>(px.rank);  // [b, 1, w, c]


            keepdims_rank = px.rank;

            int j = 0;
            for (int i = 0; i < px.rank; i++)
            {
                if (i != ax) { reshape_shape[j++] = shape[i]; keepdims_shape[i] = shape[i]; }
                else keepdims_shape[i] = 1;
            }
            
            init = true;
        }

        X = Tensor(px);

        Tensor out_keepdims = Tensor::create(keepdims_shape.get(), keepdims_rank);
        for (size_t i = 0; i < out_keepdims.tot_size; i++) out_keepdims.tensor[i] = 0;

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
            Tensor out = Tensor::create(reshape_shape.get(), keepdims_rank - 1);
            double* p_out = out.tensor.get();

            for (size_t i = 0; i < out_keepdims.tot_size; i++) p_out[i] = pm_okd[i];
            return out;
        }

        return out_keepdims;
    }

Tensor ReduceSum::backward_pass(const Tensor& dy, double, matrixOperations& wf) 
    {
        if (!init) throw std::invalid_argument("layer not initilized");

        Tensor dx = Tensor(X);
        for (size_t i = 0; i < dx.tot_size; i++) dx.tensor[i] = 0;


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
