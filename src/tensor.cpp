#include <iostream>
#include "../include/tensor.h"

Tensor::Tensor(const std::initializer_list<Tensor>& vs)
{   
    if (static_cast<size_t>(vs.size()) == 0) throw std::invalid_argument("empty tensor");

    size_t indexer = 0;
    rank = vs.begin()->rank + 1;
    shape = std::make_unique<size_t[]>(rank);
    shape[0] = static_cast<size_t>(vs.size());
    for (size_t i = 0; i < vs.begin()->rank; i++) shape[i+1] = vs.begin()->shape[i];

    for (const Tensor& v : vs)
    { 
        tot_size += v.tot_size;
        if (rank != v.rank + 1) throw std::invalid_argument("ragged tensor");
        for (size_t i = 0; i < v.rank; i++) 
        {
            if (shape[i+1] != v.shape[i]) throw std::invalid_argument("ragged tensor");
        }
    }

    tensor = std::make_unique<float[]>(tot_size);
    for (const Tensor& v : vs)
    {
        for (size_t j = 0; j < v.tot_size; j++)
        {
            tensor[indexer + j] = v.tensor[j];
        }
        indexer += v.tot_size;
    }
}

Tensor::Tensor(const std::initializer_list<float>& input)
{   
    getRank(input);
    if (rank == 0) throw std::invalid_argument("need at least one dim");
    shape = std::make_unique<size_t[]>(rank);

    size_t level = 0;
    getShape(input, level);

    level = 0;
    tot_size = 1;
    for (size_t i = 0; i < rank; i++) tot_size *= shape[i];
    tensor = std::make_unique<float[]>(tot_size);
    getArr(input, level);

};

Tensor::Tensor(const std::initializer_list<size_t>& in_shape, const char&)
{   
    rank = static_cast<size_t>(in_shape.size());
    if (rank <= 0) throw std::invalid_argument("need at least one dimension");

    shape = std::make_unique<size_t[]>(rank);
    {
        size_t i = 0; for (int s : in_shape) shape[i++] = s;
    }

    tot_size = 1;
    for (size_t i = 0; i < rank; i++) tot_size *= shape[i];
    tensor = std::make_unique<float[]>(tot_size);

}

Tensor::Tensor(const size_t in_shape[], const char&, const size_t& carray_len)
{   
    // this is added in cases where c-array is provided
    rank = carray_len;

    if (rank <= 0) throw std::invalid_argument("need at least one dimension");

    shape = std::make_unique<size_t[]>(rank);
    {
        // b/c init list does not allow indexing
        for (size_t s = 0; s < rank; s++) shape[s] = in_shape[s];
    }

    tot_size = 1;
    for (size_t i = 0; i < rank; i++) tot_size *= shape[i];
    tensor = std::make_unique<float[]>(tot_size);

}

float& Tensor::index(const size_t params[])
{
    // TODO: ADD CHECKS!!!!!
    // if (sizeof(params)/sizeof(params[0]) != rank) throw std::invalid_argument("requested shape does not match tensor");
    size_t val = 0;
    for (size_t i = 0; i < rank; i++)
        val = val * shape[i] + params[i];
    return tensor[val];
}

float Tensor::index(const size_t params[]) const
{
    // if (sizeof(params)/sizeof(params[0]) != rank) throw std::invalid_argument("requested shape does not match tensor");
    
    size_t val = 0;
    for (size_t i = 0; i < rank; i++)
        val = val * shape[i] + params[i];
    return tensor[val];
}

void Tensor::print_shape()
{
    std::cout << "[ ";
    for (size_t i = 0; i < rank; i++) std::cout << shape[i] << " ";
    std::cout << "]";
    std::cout << "\n";

}

Tensor Tensor::ops(const Tensor& other, float (*f)(float, float)) const
{   
    // if we should broadcast one of the tensors because its like [a, b, c] and [1, 1, c] then run ops_bcast
    if (rank != other.rank) return ops_bcast(other, f);
    for (size_t i = 0; i < rank; i++)
        if (shape[i] != other.shape[i]) return ops_bcast(other, f);

    // no need to broadcast...
    Tensor t = Tensor(*this);
    float* a = (this->tensor).get();
    float* b = (other.tensor).get();
    float* c = (t.tensor).get();

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < tot_size; i++) 
    {   
        c[i] = f(a[i], b[i]);
    }
    return t;
}

Tensor Tensor::ops_bcast(const Tensor& other, float (*f)(float, float)) const
{
    size_t out_rank = std::max(rank, other.rank);
    std::unique_ptr<size_t[]> out = std::make_unique<size_t[]>(out_rank);

    // making new a and b shapes because we might need to pad the lower sized tensor
    std::unique_ptr<size_t[]> shape_a = std::make_unique<size_t[]>(out_rank);
    std::unique_ptr<size_t[]> shape_b = std::make_unique<size_t[]>(out_rank);
    std::unique_ptr<size_t[]> stride_a = std::make_unique<size_t[]>(out_rank);
    std::unique_ptr<size_t[]> stride_b = std::make_unique<size_t[]>(out_rank);
    std::unique_ptr<size_t[]> pitch = std::make_unique<size_t[]>(out_rank);

    // 1 fill
    for (size_t i = 0; i < out_rank; i++) 
    {
        shape_a[i] = 1; shape_b[i] = 1; stride_a[i] = 1; stride_b[i] = 1; pitch[i] = 1;
    }

    // if we need to pad other e.g. [2, 3, 4] and [3, 4] -> [2, 3, 4] and [1, 3, 4]
    if (rank > other.rank)
        for (size_t i = 0; i < out_rank; i++)
        {
            shape_a[i] = shape[i];
            if (i >= (rank - other.rank)) shape_b[i] = other.shape[i - (rank - other.rank)];
        }
    else if (rank < other.rank)
        for (size_t i = 0; i < out_rank; i++)
        {
            shape_b[i] = other.shape[i];
            if (i >= (other.rank - rank)) shape_a[i] = shape[i - (other.rank - rank)];
        }

    // if no need to pad
    else
        for (size_t i = 0; i < out_rank; i++)
        {
            shape_a[i] = shape[i];
            shape_b[i] = other.shape[i];
        }

    // ensure shapes match
    for (size_t i = 0; i < out_rank; i++) {
        if (!(shape_a[i] == shape_b[i] || shape_a[i] == 1 || shape_b[i] == 1))
            throw std::invalid_argument("matrix size mismatch [T2]");
        out[i] = std::max(shape_a[i], shape_b[i]);
    }

    // our out tensor
    Tensor c = Tensor::create(out.get(), out_rank);

    for (int i = out_rank-2; i >= 0; i--)
    {
        stride_a[i] = stride_a[i + 1] * shape_a[i + 1];
        stride_b[i] = stride_b[i + 1] * shape_b[i + 1];
        pitch[i] = pitch[i + 1] * out[i + 1];
    }

    // broadcasting: if size==1 on an axis that stride must be 0
    for (size_t i = 0; i < out_rank; i++)
        {
            if (shape_a[i] == 1) stride_a[i] = 0;
            if (shape_b[i] == 1) stride_b[i] = 0;
        }

    float* p_a = tensor.get();
    float* p_b = other.tensor.get();
    float* p_c = c.tensor.get();

    size_t total = 1;
    for (size_t i = 0; i < out_rank; ++i) total *= out[i];

    #pragma omp parallel for schedule(static)
    for (size_t lin = 0; lin < total; lin++)
    {
        size_t offA = 0, offB = 0, rem = lin;
        for (size_t ax = 0; ax < out_rank; ax++) {
            const size_t idx = rem / pitch[ax];
            rem %= pitch[ax];
            offA += idx * stride_a[ax];
            offB += idx * stride_b[ax];
        }
        p_c[lin] = f(p_a[offA], p_b[offB]);
    }
    return c;
}

Tensor Tensor::ops(const float scalar, float (*f)(float, float)) const
{   
    
    Tensor t = Tensor(*this);
    float* a = (this->tensor).get();
    float* c = (t.tensor).get();

    for (size_t i = 0; i < tot_size; i++)  c[i] = f(a[i], scalar);
    
    return t;
}

Tensor& Tensor::operator=(const Tensor& other)
{
    if (this != &other) 
    {
        rank = other.rank; tot_size = other.tot_size;

        shape = std::make_unique<size_t[]>(rank);
        tensor = std::make_unique<float[]>(tot_size);

        std::memcpy(shape.get(), other.shape.get(), sizeof(size_t)*rank);
        std::memcpy(tensor.get(), other.tensor.get(), sizeof(float)*tot_size);
    }
    return *this;
}

Tensor::Tensor(Tensor&& other) noexcept
{
    rank = other.rank; tot_size = other.tot_size;
    shape = std::move(other.shape);
    tensor = std::move(other.tensor);
}

Tensor& Tensor::operator=(Tensor&& other)
{
    rank = other.rank; tot_size = other.tot_size;
    shape = std::move(other.shape);
    tensor = std::move(other.tensor);
    
    return *this;
}

Tensor::Tensor(const Tensor& other) // copy constructor
    :
        rank(other.rank),
        tot_size(other.tot_size),
        shape(std::make_unique<size_t[]>(other.rank)),
        tensor(std::make_unique<float[]>(other.tot_size))
{
    std::memcpy(shape.get(), other.shape.get(), sizeof(size_t)*rank);
    std::memcpy(tensor.get(), other.tensor.get(), sizeof(float)*tot_size);
}

Tensor& Tensor::operator+=(const float scalar) 
{   
    float* a = (this->tensor).get();
    for (size_t i = 0; i < tot_size; i++) a[i] += scalar;
    return *this;
}

// when the scalar is in front
Tensor operator+(float s, const Tensor& t) { return t + s; }
Tensor operator-(float s, const Tensor& t) { return (t * - 1) + s; }
Tensor operator*(float s, const Tensor& t) { return t * s; }
Tensor operator/(float s, const Tensor& t) { return t.ops(s, [](float a, float b){ return b / a; }); } // here makes sure to put tensor in denom
