#include <iostream>
#include "tensor.h"

Tensor::Tensor(std::initializer_list<double> ds)
{
    rank = 1;
    tot_size = static_cast<int>(ds.size());
    index_helper = nullptr;
    shape = std::make_unique<int[]>(rank);
    batch = tot_size;
    shape[0] = tot_size;
    tensor = std::make_unique<double[]>(tot_size);

    int indexer = 0;
    for (double i : ds)
    {
        tensor[indexer] = i;
        indexer++;
    }
}

Tensor::Tensor(const std::initializer_list<Tensor>& vs)
{   
    if (static_cast<int>(vs.size()) == 0) throw std::invalid_argument("empty tensor");

    size_t indexer = 0;
    rank = vs.begin()->rank + 1;
    shape = std::make_unique<int[]>(rank);
    shape[0] = static_cast<size_t>(vs.size());
    for (int i = 0; i < vs.begin()->rank; i++) shape[i+1] = vs.begin()->shape[i];

    for (const Tensor& v : vs)
    { 
        tot_size += v.tot_size;
        if (rank != v.rank + 1) throw std::invalid_argument("ragged tensor");
        for (int i = 0; i < v.rank; i++) 
        {
            if (shape[i+1] != v.shape[i]) throw std::invalid_argument("ragged tensor");
        }
    }

    tensor = std::make_unique<double[]>(tot_size);
    for (const Tensor& v : vs)
    {
        for (int j = 0; j < v.tot_size; j++)
        {
            tensor[indexer + j] = v.tensor[j];
        }
        indexer += v.tot_size;
    }

    index_helper = std::make_unique<int[]>(rank - 1);
    index_helper[rank - 2] = shape[rank - 1];
    for (int i = rank - 3; i > -1; i--) index_helper[i] = shape[i + 1] * index_helper[i + 1];
    for (int i = rank - 1; i > -1; i--)
    {
        if (i == rank - 1) col = shape[i];
        else if (i == rank - 2) row = shape[i];
        else batch *= shape[i];
    }
}

double& Tensor::index(const std::vector<size_t>& params)
{
    if (static_cast<int>(params.size()) != rank) throw std::invalid_argument("requested shape does not match tensor");
    
    int val = params[rank - 1];
    for (int i = 0; i < (rank - 1); i++) val += params[i] * index_helper[i];
    return tensor[val];
}

double Tensor::index(const std::vector<size_t>& params) const
{
    if (static_cast<int>(params.size()) != rank) throw std::invalid_argument("requested shape does not match tensor");
    
    int val = params[rank - 1];
    for (int i = 0; i < (rank - 1); i++) val += params[i] * index_helper[i];
    return tensor[val];
}

Tensor Tensor::ops(const Tensor& other, const char op) const
{   
    if (rank != other.rank) throw std::invalid_argument("matrix size mismatch");
    for (int i = 0; i < rank; i++) 
    {
        if (shape[i] != other.shape[i]) throw std::invalid_argument("matrix size mismatch");
    }

    Tensor t = Tensor(*this);
    double* a = (this->tensor).get();
    double* b = (other.tensor).get();
    double* c = (t.tensor).get();

    for (size_t i = 0; i < batch * row * col; i++) 
    {   
        switch (op)
        {
        case 'A':
            c[i] = a[i] + b[i];
            break;
        case 'S':
            c[i] = a[i] - b[i];
            break;
        case 'M':
            c[i] = a[i] * b[i];
            break;
        case 'D':
            c[i] = a[i] / b[i];
            break;
        default:
            std::cout << "ERROR, SOMETHING WENT WRONG; THATS ALL I KNOW" << std::endl;
            break;
        }
    }
    return t;
}

Tensor Tensor::ops(const double scalar, const char op) const
{   
    
    Tensor t = Tensor(*this);
    double* a = (this->tensor).get();
    double* c = (t.tensor).get();

    for (size_t i = 0; i < batch * row * col; i++) 
    {   
        switch (op)
        {
        case 'A':
            c[i] = a[i] + scalar;
            break;
        case 'S':
            c[i] = a[i] - scalar;
            break;
        case 'M':
            c[i] = a[i] * scalar;
            break;
        case 'D':
            c[i] = a[i] / scalar;
            break;
        case 'I':
            c[i] = scalar / a[i];
            break;
        default:
            std::cout << "ERROR, SOMETHING WENT WRONG; THATS ALL I KNOW" << std::endl;
            break;
        }
    }
    return t;
}

Tensor& Tensor::operator=(const Tensor& other)
{
    if (this != &other) 
    {
        batch = other.batch; row = other.row; col = other.col; rank = other.rank; tot_size = other.tot_size;

        shape = std::make_unique<int[]>(rank);
        index_helper = std::make_unique<int[]>(rank-1);
        tensor = std::make_unique<double[]>(batch*row*col);

        std::memcpy(shape.get(), other.shape.get(), sizeof(int)*rank);
        std::memcpy(index_helper.get(), other.index_helper.get(), sizeof(int)*(rank-1));
        std::memcpy(tensor.get(), other.tensor.get(), sizeof(double)*batch*row*col);
    }
    return *this;
}

Tensor::Tensor(const Tensor& other) // copy constructor
    :
        batch(other.batch),
        row(other.row),
        col(other.col),
        rank(other.rank),
        tot_size(other.tot_size),
        shape(std::make_unique<int[]>(rank)),
        index_helper(std::make_unique<int[]>(rank-1)),
        tensor(std::make_unique<double[]>(batch*row*col))
{
    std::memcpy(shape.get(), other.shape.get(), sizeof(int)*rank);
    std::memcpy(index_helper.get(), other.index_helper.get(), sizeof(int)*(rank-1));
    std::memcpy(tensor.get(), other.tensor.get(), sizeof(double)*batch*row*col);
}

Tensor& Tensor::operator+=(const double scalar) 
{   
    double* a = (this->tensor).get();
    for (size_t i = 0; i < batch * row * col; i++) a[i] += scalar;
    return *this;
}
