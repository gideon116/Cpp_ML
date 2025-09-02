#if 1
#include "include/tokenizer.h"
#include "include/layers.h"
#include "include/tensor.h"
#include "include/model.h"
#include <iostream>


Tensor create_padding_mask(const Tensor& seq)
{
    Tensor mask = seq;
    for (size_t i = 0; i < mask.m_size; i++)
        if (mask.m_tensor[i] == 0)
            mask.m_tensor[i] = 1;
        else
            mask.m_tensor[i] = 0;
    mask.reshape((size_t[4]){mask.m_shape[0], 1, 1, mask.m_shape[1]}, 4);
    return mask;
}

Tensor create_look_ahead_mask(const Tensor& seq)
{
    size_t seq_len = seq.m_shape[1];
    Tensor look_ahead_mask = Tensor::create((size_t[4]){1, 1, seq_len, seq_len}, 4);

    for (size_t i = 0; i < seq_len; i++)
        for (size_t j = 0; j < seq_len; j++)
            look_ahead_mask.m_tensor[i*seq_len + j] = (j > i) ? 1.0f : 0.0f;

    return look_ahead_mask;
}

Tensor create_dec_mask(Tensor dec_inputs)
{
    Tensor dec_mask_1 = create_padding_mask(dec_inputs); // b, 1, 1, ml
    Tensor dec_mask_2 = create_look_ahead_mask(dec_inputs); // 1, 1, ml, ml

    size_t batch = dec_mask_1.m_shape[0];
    size_t ml = dec_mask_1.m_shape[3];

    Tensor dec_mask = Tensor::create((size_t[4]){batch, 1, ml, ml}, 4); // b, 1, ml, ml

    for (size_t b = 0; b < batch; b++)
    {
        for (size_t i = 0; i < ml; i++)
        {
            for (size_t j = 0; j < ml; j++)
            {
                dec_mask.m_tensor[(b * ml + i) * ml + j] = std::max(dec_mask_1.m_tensor[b * ml + j], dec_mask_2.m_tensor[i * ml + j]);
            }
        }
    }
    return dec_mask;
}

class functional_model
{
private:
    size_t m_d_model = 128;
    size_t m_vocab_size = 32;
    size_t m_max_len;
    float m_lr = 0.0f;
    float m_loss = 0.0f;
    Tensor m_dy;
    
public:

    functional_model(size_t vocab_size) : m_vocab_size(vocab_size + 3) {}

    void train(const Tensor& enc_input, const Tensor& dec_input, const Tensor& dec_target, const Tensor& val_enc_input, const Tensor& val_dec_input, const Tensor& val_dec_target, int epochs, float lr)
    {
        if (enc_input.m_rank != 2 || dec_target.m_rank != 2)
            throw std::invalid_argument("encoder input and decoder input must be tensors of rank 2");

        Timer timer;
        std::cout << "\n____________________________________________";
        std::cout << "\nBeginning training\n\n";

        m_lr = lr;

        Tensor enc_mask = create_padding_mask(enc_input);
        Tensor dec_mask = create_dec_mask(dec_input);
        Tensor target_mask = 1.0f - create_padding_mask(dec_target);

        Tensor val_enc_mask = create_padding_mask(val_enc_input);
        Tensor val_dec_mask = create_dec_mask(val_dec_input);
        Tensor val_target_mask = 1.0f - create_padding_mask(val_dec_target);

        // input is of shape [batch, max_len]
        m_max_len = enc_input.m_shape[1];

        for (int epoch = 0; epoch < epochs; epoch++)
        {
            Timer timer;
            ValLayer pred = call(enc_input, dec_input, true, enc_mask, dec_mask);
            
            if (!epoch)
                m_dy = *(pred.val); // only set m_dy during epoch 0
            
            backward(dec_target, pred, &target_mask);

            std::cout << "epoch: " << epoch + 1 << "\n\tloss = " << m_loss << "\n";
            valid(val_enc_input, val_dec_input, val_dec_target, val_enc_mask, val_dec_mask, &val_target_mask);
        }

        std::cout << "\n____________________________________________";
        std::cout << "\nTraining complete";
        std::cout << "\nTotal training time = ";
    }

    Tensor generate(const Tensor& enc_input, const size_t& start_token, const size_t& end_token)
    {
        if (enc_input.m_rank != 2 && enc_input.m_shape[0] != 1)
            throw std::invalid_argument("generates only takes in one sample of shape [1, max_len]");

        Tensor dec_input = enc_input;
        memset(dec_input.m_tensor, 0, sizeof(float) * dec_input.m_size);
        dec_input.m_tensor[0] = start_token;

        Tensor enc_mask = create_padding_mask(enc_input);

        for (size_t i = 0; i < 10; i++) // generate 10 only
        {
            Tensor dec_mask = create_dec_mask(dec_input);

            ValLayer curr_pred = call(enc_input, dec_input, false, enc_mask, dec_mask);
            dec_input.m_tensor[i + 1] = wef::argmax(wef::softmax(*(curr_pred.val))).m_tensor[i];
            if (dec_input.m_tensor[i + 1] == (float)end_token)
                break;
        }
        return dec_input;
    }

private:
    
    Linear_GPU out{m_vocab_size, true, 7}, ffn1{m_d_model, true, 8}, ffn2{m_d_model, true, 8};
    ReLU relu1, relu2, relu3;
    LayerNorm norm{2}, norm2{2}, norm3{2}, norm4{2}, norm5{2};
    
    Embedding embedding{/*vocab*/m_vocab_size, m_d_model}, embedding_out{m_vocab_size, m_d_model};

    MHA mha_input{m_d_model, /*self_attention*/true, /*num_heads*/2, /*use_bias*/true, /*use_mask*/true, /*use_gpu*/true};
    MHA mha_output{m_d_model, true, 2, true, true, true};
    MHA mha_cross{m_d_model, false, 2, true, true, true};

    UseGPU gpu;


private:

    ValLayer call(const Tensor& enc_input, const Tensor& dec_input, const bool& training, const Tensor& enc_mask, const Tensor& dec_mask)
    {

        ValLayer x = {nullptr, &enc_input};
        x = embedding.call(x, training, &gpu);
        // according to tensorflow : "This factor sets the relative scale of the embedding and positonal_encoding."
        Tensor temp = *x.val;
        temp *= std::sqrt((float)m_d_model);
        temp = temp + wef::positional_encoding(m_max_len, m_d_model);
        x.val = &temp;

        // MHA
        ValLayer c = mha_input.call(x, x, x, training, &gpu, {nullptr, &enc_mask});

        // Add and norm
        temp = *c.val + *x.val;
        x.val = &temp;
        x = norm.call(x, training, &gpu);

        // ffn then add and norm
        c = ffn1.call(x, training, &gpu);
        c = relu1.call(c, training, &gpu);
        temp = *c.val + *x.val;
        x.val = &temp;
        x = norm2.call(x, training, &gpu);

        // output
        ValLayer x_out = {nullptr, &dec_input};
        x_out = embedding_out.call(x_out, training, &gpu);
        temp = *x_out.val;
        temp *= std::sqrt((float)m_d_model);
        temp = temp + wef::positional_encoding(m_max_len, m_d_model);
        x_out.val = &temp;

        // masked MHA
        ValLayer d = mha_output.call(x_out, x_out, x_out, training, &gpu, {nullptr, &dec_mask});
        
        // Add and norm
        temp = *d.val + *x_out.val;
        x_out.val = &temp;
        x_out = norm3.call(x_out, training, &gpu);

        // cross attn
        ValLayer e = mha_cross.call(x_out, x, x, training, &gpu, {nullptr, &enc_mask});

        // Add and norm
        temp = *e.val + *x_out.val;
        x_out.val = &temp;
        x_out = norm4.call(x_out, training, &gpu);

        // ffn then add and norm
        e = ffn1.call(x_out, training, &gpu);
        e = relu1.call(e, training, &gpu);
        temp = *e.val + *x_out.val;
        x_out.val = &temp;
        x_out = norm5.call(x_out, training, &gpu);

        e = out.call(x_out, training, &gpu);

        return e;
    }
    
    void backward(const Tensor& dec_target, const ValLayer& pred, Tensor* mask)
    {
        m_loss = wef::categoricalcrossentropy(dec_target, *(pred.val), m_dy, mask);
        ((Layer*)pred.layer)->rev(&m_dy, m_lr, &gpu);
    }

    void valid(const Tensor& val_enc_input, const Tensor& val_dec_input, const Tensor& val_dec_target, const Tensor& val_enc_mask, const Tensor& val_dec_mask, Tensor* val_target_mask)
    {
        // validation
        ValLayer val_pred_ptr = call(val_enc_input, val_dec_input, false, val_enc_mask, val_dec_mask);
        float val_loss = wef::categoricalcrossentropy(val_dec_target, *(val_pred_ptr.val), val_target_mask);
        std::cout << "\tvalid_loss = " << val_loss << "\n";

        
        std::cout << "\ttime per epoch = ";
    }

};

int main()
{
    Tokenizer tokenizer;
    tokenizer.process("english_spanish_tab.txt", 1000);
    std::cout << tokenizer.english_sen.size() << std::endl;
    std::cout << tokenizer.spanish_sen.size() << std::endl;
    
    size_t batch = (size_t)tokenizer.english_sen.size();
    size_t de_batch = (size_t)tokenizer.spanish_sen.size();
    size_t val_share = 50;

    if (batch != de_batch)
        throw std::invalid_argument("enc and dec batchs must match");

    float start_token = std::max(tokenizer.english_vsize, tokenizer.spanish_vsize) + 1;
    float end_token = start_token + 1;


    Tensor inp = Tensor::create((size_t[2]){batch - val_share, tokenizer.maxlen + 1}, 2);
    memset(inp.m_tensor, 0, sizeof(float) * inp.m_size);

    Tensor dec = Tensor::create((size_t[2]){batch - val_share, tokenizer.maxlen + 1}, 2);
    memset(dec.m_tensor, 0, sizeof(float) * dec.m_size);

    Tensor tar = Tensor::create((size_t[2]){batch - val_share, tokenizer.maxlen + 1}, 2);
    memset(tar.m_tensor, 0, sizeof(float) * tar.m_size);

   for (size_t i = 0; i < batch - val_share; i++)
    {
        float* temp = inp.m_tensor + i * (tokenizer.maxlen + 1);
        temp[tokenizer.english_sen[i].size()] = end_token;
        for (size_t j = 0; j < tokenizer.english_sen[i].size(); j++)
            temp[j] = (float)(tokenizer.english_sen[i][j]);
    }

    for (size_t i = 0; i < batch - val_share; i++)
    {
        float* inp_temp = dec.m_tensor + i * (tokenizer.maxlen + 1);
        float* tar_temp = tar.m_tensor + i * (tokenizer.maxlen + 1);

        inp_temp[0] = start_token;
        tar_temp[tokenizer.spanish_sen[i].size()] = end_token;

        for (size_t j = 0; j < tokenizer.spanish_sen[i].size(); j++)
        {
            inp_temp[j + 1] = (float)(tokenizer.spanish_sen[i][j]);
            tar_temp[j] = (float)(tokenizer.spanish_sen[i ][j]);
        }
    }

    Tensor val_inp = Tensor::create((size_t[2]){val_share, tokenizer.maxlen + 1}, 2);
    memset(val_inp.m_tensor, 0, sizeof(float) * val_inp.m_size);

    Tensor val_dec = Tensor::create((size_t[2]){val_share, tokenizer.maxlen + 1}, 2);
    memset(val_dec.m_tensor, 0, sizeof(float) * val_dec.m_size);

    Tensor val_tar = Tensor::create((size_t[2]){val_share, tokenizer.maxlen + 1}, 2);
    memset(val_tar.m_tensor, 0, sizeof(float) * val_tar.m_size);

    for (size_t i = 0; i < val_share; i++)
    {
        float* temp = val_inp.m_tensor + i * (tokenizer.maxlen + 1);
        temp[tokenizer.english_sen[i + (batch - val_share)].size()] = end_token;
        for (size_t j = 0; j < tokenizer.english_sen[i + (batch - val_share)].size(); j++)
            temp[j] = (float)(tokenizer.english_sen[i + (batch - val_share)][j]);
    }

    for (size_t i = 0; i < val_share; i++)
    {
        float* inp_temp = val_dec.m_tensor + i * (tokenizer.maxlen + 1);
        float* tar_temp = val_tar.m_tensor + i * (tokenizer.maxlen + 1);

        inp_temp[0] = start_token;
        tar_temp[tokenizer.spanish_sen[i + (batch - val_share)].size()] = end_token;

        for (size_t j = 0; j < tokenizer.spanish_sen[i + (batch - val_share)].size(); j++)
        {
            inp_temp[j + 1] = (float)(tokenizer.spanish_sen[i + (batch - val_share)][j]);
            tar_temp[j] = (float)(tokenizer.spanish_sen[i + (batch - val_share)][j]);
        }
    }

    functional_model model(std::max(tokenizer.english_vsize, tokenizer.spanish_vsize));
    model.train(inp, dec, tar, val_inp, val_dec, val_tar, 5, 0.01f);

    Tensor test_enc_input = Tensor::create((size_t[2]){1, (tokenizer.maxlen + 1)}, 2);
    memset(test_enc_input.m_tensor, 0, sizeof(float) * test_enc_input.m_size);
    for (size_t i = 0; i < test_enc_input.m_size; i++)
        test_enc_input.m_tensor[i] = val_inp.m_tensor[i];
    
    Tensor gen = model.generate(test_enc_input, start_token, end_token);
    
    tokenizer.tok_to_eng(test_enc_input.m_tensor, test_enc_input.m_size);
    tokenizer.tok_to_spa(gen.m_tensor, gen.m_size);
    wef::print(gen);

    
  

    return 0;
}
#endif
#if 0
#include "include/layers.h"
#include "include/tensor.h"
#include "include/model.h"
#include "include/mnist.h"

// TODO: mutithread tensor ops and maxpool
// TODO: add gpu support for linux and windows
// TODO: tile dw and dx shaders for conv2d backward pass
// TODO: make tensor class tempelatable
// TODO: add shaders for reduce sum / layer normalization
// TODO: use function pointers in shaders
// TODO: GPU version of transpose (should be simple flat gx=256, gy=1, gz=1)

int main() {

    int n_test = 100;
    int n_train = 1000;

    Tensor train_im = load_mnist_images("mnist/train-images-idx3-ubyte", n_train);
    Tensor train_l = load_mnist_labels("mnist/train-labels-idx1-ubyte", n_train);

    Tensor test_im = load_mnist_images("mnist/t10k-images-idx3-ubyte", n_test);
    Tensor test_l = load_mnist_labels("mnist/t10k-labels-idx1-ubyte", n_test);

    std::cout << "train image shape is: "; train_im.print_shape();
    std::cout << "train label shape is: "; train_l.print_shape();

    float lr = 0.01f;

    int m_units1 = 16;
    int m_units2 = 16;
    int m_units5 = 10;
    
    Conv2D_GPU cov1(3, 3, m_units1, true, 3), cov2(3, 3, m_units2, true, 4), cov3(3, 3, m_units2, true, 5);
    Linear_GPU out(m_units5, true, 7), ffn1(16, true, 8), ffn2(512, true, 8), ffn3(512, true, 8);
    ReLU relu1, relu2, relu3;
    ReduceSum r1(1), r2(1);
    Flatten flat;
    LayerNorm norm(1);
    MaxPool2D_GPU mp(2, 2), mp2(2, 2);

    std::vector<Layer*> network = {&cov1, &relu1, &mp, &cov2, &relu3, &mp2, &cov3, &relu2, &flat, &ffn1, &out};
    Model model(network, true);
    
    /*
    //we can also build a model this way
    Model model;
    model.add(&cov1); model.add(&relu1); model.add(&cov2); model.add(&r1); model.add(&r2); model.add(&layer);
    */
    
    model.fit(train_l, train_im, test_l, test_im, 10, lr);

    Tensor pred = model.predict(test_im);

    std::cout << "\npred: { ";
    for (int i = 0; i < 10; i++) std::cout << wef::argmax(wef::softmax(pred)).m_tensor[i] << " ";
    std::cout << "}\nreal: { ";
    for (int i = 0; i < 10; i++) std::cout << test_l.m_tensor[i] << " ";
    std::cout << "} \n\n";

    model.summary();
    std::cout << "\n";

    return 0;
}

#endif