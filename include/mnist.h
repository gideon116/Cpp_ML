#pragma once

#include <fstream>
#include <cstdint>
#include <stdexcept>
#include "tensor.h"

static inline uint32_t read_u32_be(std::ifstream& f)
{
    uint8_t b[4];
    f.read(reinterpret_cast<char*>(b), 4);
    return (uint32_t(b[0]) << 24) | (uint32_t(b[1]) << 16) | (uint32_t(b[2]) <<  8) |  uint32_t(b[3]);
}

// images (b x h x c x 1) normalized 0-1 from 0-255
Tensor load_mnist_images(const std::string& path, size_t max_items=1e19)
{
    std::ifstream f(path, std::ios::binary);
    if (!f)
        throw std::runtime_error("cannot open " + path);

    uint32_t magic = read_u32_be(f); // 2051 = 0x00000803
    uint32_t count = read_u32_be(f);
    uint32_t rows = read_u32_be(f);
    uint32_t cols = read_u32_be(f);

    count = std::min<size_t>(count, max_items);

    if (magic != 2051)
        throw std::runtime_error("bad magic in " + path);

    Tensor img = Tensor::create({count, rows, cols, 1});

    size_t need = size_t(count) * rows * cols;
    for (size_t i = 0; i < need; i++)
    {
        uint8_t byte;
        f.read(reinterpret_cast<char*>(&byte), 1);

        if (!f)
            throw std::runtime_error("file truncated: " + path);

        img.m_tensor[i] = float(byte) / 255.0f;
    }

    return img;
}

// labels: 0-9
Tensor load_mnist_labels(const std::string& path, size_t max_items=1e19)
{
    std::ifstream f(path, std::ios::binary);
    if (!f) 
        throw std::runtime_error("cannot open " + path);

    uint32_t magic = read_u32_be(f); // 2049 = 0x00000801
    uint32_t count = read_u32_be(f);

    if (magic != 2049)
        throw std::runtime_error("bad magic in " + path);

    count = std::min<size_t>(count, max_items);

    Tensor lab = Tensor::create({count, 1});
    size_t need = count; 
    for (size_t i = 0; i < need; i++)
    {
        uint8_t byte;
        f.read(reinterpret_cast<char*>(&byte), 1);
        if (!f)
            throw std::runtime_error("file truncated: " + path);
        lab.m_tensor[i] = float(byte);
    }

    return lab;
}
