#include "example_models.h"
#include "tokenizer.h"
#include "transformer_model.h"
#include <iostream>

#if !defined(LEARNN_HAS_VULKAN) && !defined(LEARNN_HAS_CUDA)
void transformer()
{
    std::cerr << "[learnn] transformer() requires GPU support (Vulkan or CUDA). "
              << "Rebuild with Vulkan SDK or CUDA toolkit to use this example.\n";
}
#else

void transformer()
{
    Tokenizer tokenizer;
    tokenizer.process("../english_spanish_tab.txt", 4000);
    std::cout << "English Sentences: " << tokenizer.english_sen.size() << std::endl;
    std::cout << "Spanish Sentences: " << tokenizer.spanish_sen.size() << std::endl;
    
    size_t batch = (size_t)tokenizer.english_sen.size();
    size_t de_batch = (size_t)tokenizer.spanish_sen.size();
    size_t val_share = 50;

    if (batch != de_batch)
        throw std::invalid_argument("enc and dec batchs must match");

    float start_token = std::max(tokenizer.english_vsize, tokenizer.spanish_vsize) + 1;
    float end_token = start_token + 1;
    size_t start_tok = (size_t)start_token;
    size_t end_tok = (size_t)end_token;

    size_t newshape[2] = {batch - val_share, tokenizer.maxlen + 1};

    Tensor inp = Tensor::create(newshape, 2);
    memset(inp.m_tensor, 0, sizeof(float) * inp.m_size);

    Tensor dec = Tensor::create(newshape, 2);
    memset(dec.m_tensor, 0, sizeof(float) * dec.m_size);

    Tensor tar = Tensor::create(newshape, 2);
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
            tar_temp[j] = (float)(tokenizer.spanish_sen[i][j]);
        }
    }

    size_t val_newshape[2] = {val_share, tokenizer.maxlen + 1};

    Tensor val_inp = Tensor::create(val_newshape, 2);
    memset(val_inp.m_tensor, 0, sizeof(float) * val_inp.m_size);

    Tensor val_dec = Tensor::create(val_newshape, 2);
    memset(val_dec.m_tensor, 0, sizeof(float) * val_dec.m_size);

    Tensor val_tar = Tensor::create(val_newshape, 2);
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

    TransformerModel model(std::max(tokenizer.english_vsize, tokenizer.spanish_vsize),
                           /*d_model=*/128, /*num_heads=*/4, /*batch_size=*/50, /*use_gpu=*/true);
    model.train(inp, dec, tar, val_inp, val_dec, val_tar, 10, 0.05f);

    size_t tein_newshape[2] = {1, (tokenizer.maxlen + 1)};

    Tensor test_enc_input = Tensor::create(tein_newshape, 2);
    memset(test_enc_input.m_tensor, 0, sizeof(float) * test_enc_input.m_size);
    for (size_t i = 0; i < test_enc_input.m_size; i++)
        test_enc_input.m_tensor[i] = val_inp.m_tensor[i];
    
    Tensor gen = model.generate(test_enc_input, start_tok, end_tok);
    
    tokenizer.tok_to_eng(test_enc_input.m_tensor, test_enc_input.m_size);
    tokenizer.tok_to_spa(gen.m_tensor, gen.m_size);
    wef::print(gen);

}

#endif // LEARNN_HAS_VULKAN || LEARNN_HAS_CUDA