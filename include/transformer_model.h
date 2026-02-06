#pragma once

#include "layers.h"
#include "tensor.h"
#include "model.h"
#include "matrix_operations.h"
#include <iostream>
#include <memory>
#include <cmath>
#include <cstring>
#include <algorithm>

// ── Mask utilities ───────────────────────────────────────

inline Tensor transf_create_padding_mask(const Tensor& seq)
{
    Tensor mask = seq;
    for (size_t i = 0; i < mask.m_size; i++)
        if (mask.m_tensor[i] == 0)
            mask.m_tensor[i] = 1;
        else
            mask.m_tensor[i] = 0;

    size_t newshape[4] = {mask.m_shape[0], 1, 1, mask.m_shape[1]};
    mask.reshape(newshape, 4);
    return mask;
}

inline Tensor transf_create_look_ahead_mask(const Tensor& seq)
{
    size_t seq_len = seq.m_shape[1];
    size_t newshape[4] = {1, 1, seq_len, seq_len};
    Tensor look_ahead_mask = Tensor::create(newshape, 4);

    for (size_t i = 0; i < seq_len; i++)
        for (size_t j = 0; j < seq_len; j++)
            look_ahead_mask.m_tensor[i*seq_len + j] = (j > i) ? 1.0f : 0.0f;

    return look_ahead_mask;
}

inline Tensor transf_create_dec_mask(Tensor dec_inputs)
{
    Tensor dec_mask_1 = transf_create_padding_mask(dec_inputs);
    Tensor dec_mask_2 = transf_create_look_ahead_mask(dec_inputs);

    size_t batch = dec_mask_1.m_shape[0];
    size_t ml = dec_mask_1.m_shape[3];

    size_t newshape[4] = {batch, 1, ml, ml};
    Tensor dec_mask = Tensor::create(newshape, 4);

    for (size_t b = 0; b < batch; b++)
        for (size_t i = 0; i < ml; i++)
            for (size_t j = 0; j < ml; j++)
                dec_mask.m_tensor[(b * ml + i) * ml + j] =
                    std::max(dec_mask_1.m_tensor[b * ml + j],
                             dec_mask_2.m_tensor[i * ml + j]);
    return dec_mask;
}

// ── TransformerModel ─────────────────────────────────────

class TransformerModel
{
private:
    size_t m_d_model;
    size_t m_num_heads;
    size_t m_vocab_size;
    size_t m_max_len = 0;
    size_t m_batch_size;
    bool   m_use_gpu;
    float  m_lr = 0.0f;
    float  m_loss = 0.0f;
    Tensor m_dy;

    // ── layers (owned) ──
    std::unique_ptr<Layer> m_out;
    std::unique_ptr<Layer> m_ffn1;
    ReLU m_relu1;
    LayerNorm m_norm{2}, m_norm2{2}, m_norm3{2}, m_norm4{2}, m_norm5{2};
    Embedding m_embedding;
    Embedding m_embedding_out;
    MHA m_mha_input;
    MHA m_mha_output;
    MHA m_mha_cross;

    // ── GPU backend ──
#ifdef LEARNN_HAS_VULKAN
    UseGPU* m_gpu_backend = nullptr;
#endif
    void* m_gpu = nullptr;

public:
    TransformerModel(size_t vocab_size,
                     size_t d_model     = 128,
                     size_t num_heads   = 4,
                     size_t batch_size  = 50,
                     bool   use_gpu     = false)
        : m_d_model(d_model),
          m_num_heads(num_heads),
          m_vocab_size(vocab_size + 3),
          m_batch_size(batch_size),
          m_use_gpu(use_gpu),
          m_embedding(m_vocab_size, m_d_model),
          m_embedding_out(m_vocab_size, m_d_model),
          m_mha_input (m_d_model, /*self_attention*/true,  m_num_heads, /*use_bias*/true, /*use_mask*/true, use_gpu),
          m_mha_output(m_d_model, /*self_attention*/true,  m_num_heads, /*use_bias*/true, /*use_mask*/true, use_gpu),
          m_mha_cross (m_d_model, /*self_attention*/false, m_num_heads, /*use_bias*/true, /*use_mask*/true, use_gpu)
    {
        if (use_gpu)
        {
#if defined(LEARNN_HAS_VULKAN) || defined(LEARNN_HAS_CUDA)
            m_out  = std::make_unique<Linear_GPU>(m_vocab_size, true, 7);
            m_ffn1 = std::make_unique<Linear_GPU>(m_d_model, true, 8);
#ifdef LEARNN_HAS_VULKAN
            m_gpu_backend = new UseGPU;
            m_gpu = (void*)m_gpu_backend;
#endif
#else
            throw std::runtime_error("GPU support not available — rebuild with Vulkan SDK or CUDA toolkit");
#endif
        }
        else
        {
            m_out  = std::make_unique<Linear_Fast>(m_vocab_size, true, 7);
            m_ffn1 = std::make_unique<Linear_Fast>(m_d_model, true, 8);
        }
    }

    ~TransformerModel()
    {
#ifdef LEARNN_HAS_VULKAN
        delete m_gpu_backend;
#endif
    }

    // non-copyable
    TransformerModel(const TransformerModel&) = delete;
    TransformerModel& operator=(const TransformerModel&) = delete;

    // ── public API ──────────────────────────────────

    void train(const Tensor& enc_input,     const Tensor& dec_input,     const Tensor& dec_target,
               const Tensor& val_enc_input, const Tensor& val_dec_input, const Tensor& val_dec_target,
               int epochs, float lr)
    {
        if (enc_input.m_rank != 2 || dec_target.m_rank != 2)
            throw std::invalid_argument("encoder input and decoder target must be tensors of rank 2");

        Timer timer;
        std::cout << "\n____________________________________________";
        std::cout << "\nBeginning training\n\n";

        m_lr = lr;

        Tensor enc_mask     = transf_create_padding_mask(enc_input);
        Tensor dec_mask     = transf_create_dec_mask(dec_input);
        Tensor target_mask  = 1.0f - transf_create_padding_mask(dec_target);

        Tensor val_enc_mask    = transf_create_padding_mask(val_enc_input);
        Tensor val_dec_mask    = transf_create_dec_mask(val_dec_input);
        Tensor val_target_mask = 1.0f - transf_create_padding_mask(val_dec_target);

        m_max_len = enc_input.m_shape[1];

        size_t batch_size = m_batch_size;
        if (!batch_size)
            batch_size = enc_input.m_shape[0];
        size_t num_batches = enc_input.m_shape[0] / batch_size;

        Tensor min_enc_input   = create_minibatch(enc_input, batch_size);
        Tensor min_dec_input   = create_minibatch(dec_input, batch_size);
        Tensor min_dec_target  = create_minibatch(dec_target, batch_size);
        Tensor min_enc_mask    = create_minibatch(enc_mask, batch_size);
        Tensor min_dec_mask    = create_minibatch(dec_mask, batch_size);
        Tensor min_target_mask = create_minibatch(target_mask, batch_size);

        for (int epoch = 0; epoch < epochs; epoch++)
        {
            Timer epoch_timer;
            for (size_t b = 0; b < num_batches; b++)
            {
                std::memcpy(min_enc_input.m_tensor,   enc_input.m_tensor   + b * min_enc_input.m_size,   sizeof(float) * min_enc_input.m_size);
                std::memcpy(min_dec_input.m_tensor,   dec_input.m_tensor   + b * min_dec_input.m_size,   sizeof(float) * min_dec_input.m_size);
                std::memcpy(min_dec_target.m_tensor,  dec_target.m_tensor  + b * min_dec_target.m_size,  sizeof(float) * min_dec_target.m_size);
                std::memcpy(min_enc_mask.m_tensor,    enc_mask.m_tensor    + b * min_enc_mask.m_size,    sizeof(float) * min_enc_mask.m_size);
                std::memcpy(min_dec_mask.m_tensor,    dec_mask.m_tensor    + b * min_dec_mask.m_size,    sizeof(float) * min_dec_mask.m_size);
                std::memcpy(min_target_mask.m_tensor, target_mask.m_tensor + b * min_target_mask.m_size, sizeof(float) * min_target_mask.m_size);

                ValLayer pred = forward(min_enc_input, min_dec_input, true, min_enc_mask, min_dec_mask);
                if (epoch == 0 && b == 0)
                    m_dy = *(pred.val);
                backward(min_dec_target, pred, &min_target_mask);
            }
            std::cout << "epoch: " << epoch + 1 << "\n\tloss = " << m_loss << "\n";
            validate(val_enc_input, val_dec_input, val_dec_target, val_enc_mask, val_dec_mask, &val_target_mask);
        }

        std::cout << "\n____________________________________________";
        std::cout << "\nTraining complete";
        std::cout << "\nTotal training time = ";
    }

    Tensor generate(const Tensor& enc_input, size_t start_token, size_t end_token, size_t max_tokens = 10)
    {
        if (enc_input.m_rank != 2 || enc_input.m_shape[0] != 1)
            throw std::invalid_argument("generate() expects a single sample of shape [1, max_len]");

        if (!m_max_len)
            m_max_len = enc_input.m_shape[1];

        Tensor dec_input = enc_input;
        std::memset(dec_input.m_tensor, 0, sizeof(float) * dec_input.m_size);
        dec_input.m_tensor[0] = (float)start_token;

        Tensor enc_mask = transf_create_padding_mask(enc_input);

        for (size_t i = 0; i < max_tokens; i++)
        {
            Tensor dec_mask = transf_create_dec_mask(dec_input);
            ValLayer curr_pred = forward(enc_input, dec_input, false, enc_mask, dec_mask);
            dec_input.m_tensor[i + 1] = wef::argmax(wef::softmax(*(curr_pred.val))).m_tensor[i];
            if (dec_input.m_tensor[i + 1] == (float)end_token)
                break;
        }
        return dec_input;
    }

    float loss() const { return m_loss; }

private:

    ValLayer forward(const Tensor& enc_input, const Tensor& dec_input,
                     bool training, const Tensor& enc_mask, const Tensor& dec_mask)
    {
        // ── encoder ──
        ValLayer x = {nullptr, &enc_input};
        x = m_embedding.call(x, training, m_gpu);
        Tensor temp = *x.val;
        temp *= std::sqrt((float)m_d_model);
        temp = temp + wef::positional_encoding(m_max_len, m_d_model);
        x.val = &temp;

        ValLayer c = m_mha_input.call(x, x, x, training, m_gpu, {nullptr, &enc_mask});
        temp = *c.val + *x.val;
        x.val = &temp;
        x = m_norm.call(x, training, m_gpu);

        c = m_ffn1->call(x, training, m_gpu);
        c = m_relu1.call(c, training, m_gpu);
        temp = *c.val + *x.val;
        x.val = &temp;
        x = m_norm2.call(x, training, m_gpu);

        // ── decoder ──
        ValLayer x_out = {nullptr, &dec_input};
        x_out = m_embedding_out.call(x_out, training, m_gpu);
        temp = *x_out.val;
        temp *= std::sqrt((float)m_d_model);
        temp = temp + wef::positional_encoding(m_max_len, m_d_model);
        x_out.val = &temp;

        ValLayer d = m_mha_output.call(x_out, x_out, x_out, training, m_gpu, {nullptr, &dec_mask});
        temp = *d.val + *x_out.val;
        x_out.val = &temp;
        x_out = m_norm3.call(x_out, training, m_gpu);

        ValLayer e = m_mha_cross.call(x_out, x, x, training, m_gpu, {nullptr, &enc_mask});
        temp = *e.val + *x_out.val;
        x_out.val = &temp;
        x_out = m_norm4.call(x_out, training, m_gpu);

        e = m_ffn1->call(x_out, training, m_gpu);
        e = m_relu1.call(e, training, m_gpu);
        temp = *e.val + *x_out.val;
        x_out.val = &temp;
        x_out = m_norm5.call(x_out, training, m_gpu);

        e = m_out->call(x_out, training, m_gpu);
        return e;
    }

    void backward(const Tensor& dec_target, const ValLayer& pred, Tensor* mask)
    {
        m_loss = wef::categoricalcrossentropy(dec_target, *(pred.val), m_dy, mask);
        ((Layer*)pred.layer)->rev(&m_dy, m_lr, m_gpu);
    }

    void validate(const Tensor& val_enc_input, const Tensor& val_dec_input, const Tensor& val_dec_target,
                  const Tensor& val_enc_mask, const Tensor& val_dec_mask, Tensor* val_target_mask)
    {
        ValLayer val_pred = forward(val_enc_input, val_dec_input, false, val_enc_mask, val_dec_mask);
        float val_loss = wef::categoricalcrossentropy(val_dec_target, *(val_pred.val), val_target_mask);
        std::cout << "\tvalid_loss = " << val_loss << "\n";
        std::cout << "\ttime per epoch = ";
    }

    Tensor create_minibatch(const Tensor& original, size_t mini_batch_size)
    {
        size_t* shape = new size_t[original.m_rank];
        std::memcpy(shape, original.m_shape, sizeof(size_t) * original.m_rank);
        shape[0] = mini_batch_size;
        Tensor mb = Tensor::create(shape, original.m_rank);
        delete[] shape;
        return mb;
    }
};
