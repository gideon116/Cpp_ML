#pragma once

#include <fstream>
#include <iostream>
#include <unordered_map>
#include <random>
#include <vector>

struct Tokenizer
{
    std::vector<std::vector<size_t>> english_sen, spanish_sen;
    std::unordered_map<std::string, size_t> english, spanish;
    std::unordered_map<size_t, std::string> tok_to_english, tok_to_spanish;

    size_t english_vsize = 1, spanish_vsize = 1, maxlen = 1;

    std::string line, prev;
    std::vector<size_t> temp_sen;
    bool switch_vocab = true;

    std::unordered_map<std::string, size_t>* vocab;
    std::vector<std::vector<size_t>>* sen;
    size_t* vsize;
    
    void process(const std::string& filepath="english_spanish_tab.txt")
    {
        std::ifstream file(filepath);
        while (std::getline(file, line, '\t'))
        {        
            if (switch_vocab)
            {
                vocab = &english;
                sen = &english_sen;
                vsize = &english_vsize;
            }
            else
            {
                vocab = &spanish;
                sen = &spanish_sen;
                vsize = &spanish_vsize;
            }

            for (size_t ch = 0; ch < line.size() + 1; ch++)
            {
                if (line[ch] == '\n')
                {
                    maxlen = temp_sen.size() > maxlen ? temp_sen.size() : maxlen;
                    sen->push_back(temp_sen);
                    temp_sen.clear();

                    vocab = &english;
                    sen = &english_sen;
                    vsize = &english_vsize;
                    switch_vocab = !switch_vocab;
                }

                if (
                    ((int)line[ch] >= 0 && (int)line[ch] < 65) ||
                    ((int)line[ch] >= 91 && (int)line[ch] < 97) ||
                    ((int)line[ch] >= 123 && (int)line[ch] < 128)
                )
                {
                    if (prev.size())
                    {
                        if (!(*vocab)[prev])
                            (*vocab)[prev] = (*vsize)++;

                        temp_sen.push_back((*vocab)[prev]);
                        prev = "";
                    }
                }
                else
                    prev += line[ch];
            }

            maxlen = temp_sen.size() > maxlen ? temp_sen.size() : maxlen;
            sen->push_back(temp_sen);
            temp_sen.clear();

            switch_vocab = !switch_vocab;
        }

        file.close(); 

        for (const auto& kv : english)
            tok_to_english[kv.second] = kv.first;
        for (const auto& kv : spanish)
            tok_to_spanish[kv.second] = kv.first;


        shuffle(english_sen, spanish_sen);

    }

    void shuffle(std::vector<std::vector<size_t>>& m1, std::vector<std::vector<size_t>>& m2)
    {
        const size_t num = m1.size();
        if (m2.size() != num || num < 2)
            throw std::runtime_error("shape mismatch [A4]");

        std::mt19937 gen(std::random_device{}());
        for (size_t i = num - 1; i > 0; i--)
        {
            std::uniform_int_distribution<size_t> dist(0, i);
            size_t ran_in = dist(gen);

            std::vector<size_t> temp = std::move(m1[i]);
            m1[i] = std::move(m1[ran_in]);
            m1[ran_in] = std::move(temp);

            temp = std::move(m2[i]);
            m2[i] = std::move(m2[ran_in]);
            m2[ran_in] = std::move(temp);
        }
    }


    void tok_to_eng(const float* buffer, const size_t& len)
    {
        for (size_t i = 0; i < len; i++)
            std::cout << tok_to_english[(size_t)buffer[i]] << " ";
        std::cout << std::endl;
    }

    void tok_to_spa(const float* buffer, const size_t& len)
    {
        for (size_t i = 0; i < len; i++)
            std::cout << tok_to_spanish[(size_t)buffer[i]] << " ";
        std::cout << std::endl;
    }

    void vocab_sample()
    {
        int index = 0;
        for (const auto& sent : english_sen)
        {
            index++;
            if (index > english_sen.size() - 10)
            {
                for (size_t token : sent)
                std::cout << tok_to_english[token] << " ";
                std::cout << std::endl;
            }
        }

        std::cout << "__________" << std::endl;

        index = 0;
        for (const auto& sent : spanish_sen)
        {
            index++;
            if (index > spanish_sen.size() - 10)
            {
                for (size_t token : sent)
                    std::cout << tok_to_spanish[token] << " ";
                std::cout << std::endl;
            }
        }
    }
};