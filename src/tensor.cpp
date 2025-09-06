#include <iostream>
#include "tensor.h"

Tensor::Tensor(const std::initializer_list<Tensor>& init_list) // CONST??
{   
    if ((size_t)(init_list.size()) == 0)
        throw std::invalid_argument("empty tensor");

    m_rank = init_list.begin()->m_rank + 1;
    m_shape = new size_t[m_rank];
    m_shape[0] = (size_t)(init_list.size());
    for (size_t i = 0; i < init_list.begin()->m_rank; i++)
        m_shape[i + 1] = init_list.begin()->m_shape[i];

    for (const Tensor& tens : init_list)
    {
        m_size += tens.m_size;
        if (m_rank != tens.m_rank + 1)
            throw std::invalid_argument("ragged tensor");
        for (size_t i = 0; i < tens.m_rank; i++) 
            if (m_shape[i + 1] != tens.m_shape[i])
                throw std::invalid_argument("ragged tensor");
    }

    size_t stride = 0;
    m_tensor = new float[m_size];
    for (const Tensor& tens : init_list)
    {
        for (size_t i = 0; i < tens.m_size; i++)
            m_tensor[stride + i] = tens.m_tensor[i];
        stride += tens.m_size;
    }
}

Tensor::Tensor(const std::initializer_list<float>& input)
{   
    getRank(input);
    if (m_rank == 0)
        throw std::invalid_argument("need at least one dim");
    m_shape = new size_t[m_rank];

    size_t level = 0;
    getShape(input, level);

    level = 0;
    m_size = 1;
    for (size_t i = 0; i < m_rank; i++) 
        m_size *= m_shape[i];
    m_tensor = new float[m_size];
    getArr(input, level);
};

Tensor::Tensor(const std::initializer_list<size_t>& in_shape, const char&)
{   
    m_rank = (size_t)(in_shape.size());
    if (m_rank <= 0)
        throw std::invalid_argument("need at least one dimension");

    m_shape = new size_t[m_rank];
    
    size_t ix = 0;
    for (size_t s : in_shape)
        m_shape[ix++] = s;
    
    m_size = 1;
    for (size_t i = 0; i < m_rank; i++) // b/c init list does not allow indexing
        m_size *= m_shape[i];

    m_tensor = new float[m_size];

}

Tensor::Tensor(const size_t in_shape[], const char&, const size_t& carray_len)
{   
    // this is added in cases where c-array is provided
    m_rank = carray_len;

    if (m_rank <= 0)
        throw std::invalid_argument("need at least one dimension");

    m_shape = new size_t[m_rank];

    memcpy(m_shape, in_shape, m_rank * sizeof(size_t));
    
    m_size = 1;
    for (size_t i = 0; i < m_rank; i++) 
        m_size *= m_shape[i];
    m_tensor = new float[m_size];

}

void Tensor::print_shape()
{
    std::cout << "[ ";
    for (size_t i = 0; i < m_rank; i++) std::cout << m_shape[i] << " ";
    std::cout << "]";
    std::cout << "\n";
}

void Tensor::reshape(const size_t shape[], const size_t& rank)
{
    size_t temp = 1;
    for (size_t i = 0; i < rank; i++)
        temp *= shape[i];
    if (temp != m_size)
        throw std::invalid_argument("[ERROR ] requested shape does not match tensor dimensions");

    m_rank = rank;
    delete[] m_shape;
    m_shape = new size_t[rank];
    std::memcpy(m_shape, shape, sizeof(size_t) * rank);
}

float& Tensor::operator[](const std::initializer_list<size_t>& params)
{
    if ((size_t)params.size() != m_rank)
        throw std::invalid_argument("len(shape) must be == rank");

    size_t val = 0;
    size_t i = 0;
    for (const auto& index : params)
    {
        val = val * m_shape[i] + index;
        i++;
    }
    return m_tensor[val];
}

float Tensor::operator[](const std::initializer_list<size_t>& params) const
{
    if ((size_t)params.size() != m_rank)
        throw std::invalid_argument("len(shape) must be == rank");

    size_t val = 0;
    size_t i = 0;
    for (const auto& index : params)
    {
        val = val * m_shape[i] + index;
        i++;
    }
    return m_tensor[val];
}

float& Tensor::operator[](const size_t params[])
{
    // TODO: ADD CHECKS!!!!!
    // if (sizeof(params)/sizeof(params[0]) != rank) throw std::invalid_argument("requested shape does not match tensor");
    size_t val = 0;
    for (size_t i = 0; i < m_rank; i++)
        val = val * m_shape[i] + params[i];
    return m_tensor[val];
}

float Tensor::operator[](const size_t params[]) const
{
    // if (sizeof(params)/sizeof(params[0]) != rank) throw std::invalid_argument("requested shape does not match tensor");
    size_t val = 0;
    for (size_t i = 0; i < m_rank; i++)
        val = val * m_shape[i] + params[i];
    return m_tensor[val];
}

Tensor Tensor::operator[](const size_t& index)
{
    if (index >= m_shape[0])
        throw std::invalid_argument("index out of bounds");

    std::unique_ptr<size_t[]> new_shape = std::make_unique<size_t[]>(m_rank - 1);
    memcpy(new_shape.get(), m_shape + 1, (m_rank - 1) * sizeof(size_t));
    Tensor out = Tensor::create(new_shape.get(), m_rank - 1); // TODO : make new with loc same as og tensor
    memcpy(out.m_tensor, m_tensor + index * m_size/m_shape[0], (m_size/m_shape[0]) * sizeof(float));
    return out;
}

Tensor Tensor::ops(const Tensor& other, float (*f)(float&, float&)) const
{   
    // if we should broadcast one of the tensors because its like [a, b, c] and [1, 1, c] then run ops_bcast
    if (m_rank != other.m_rank)
        return ops_bcast(other, f);
    if (memcmp(m_shape, other.m_shape, m_rank * sizeof(size_t)))
        return ops_bcast(other, f);

    // no need to broadcast...
    Tensor t = Tensor(*this);
    float* a = this->m_tensor;
    float* b = other.m_tensor;
    float* c = t.m_tensor;

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < m_size; i++) 
        c[i] = f(a[i], b[i]);
    
    return t;
}

Tensor& Tensor::ops_eq(const Tensor& other, float (*f)(float&, float&))
{   

    if (m_rank != other.m_rank) // TODO : add ability to += with broadcast
        throw std::invalid_argument("[ERROR ] tensor shapes must match");
    if (memcmp(m_shape, other.m_shape, m_rank * sizeof(size_t)))
        throw std::invalid_argument("[ERROR ] tensor shapes must match");

    // no need to broadcast...
    float* a = this->m_tensor;
    float* b = other.m_tensor;

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < m_size; i++) 
        a[i] = f(a[i], b[i]);
    return *this;
}

Tensor Tensor::ops_bcast(const Tensor& other, float (*f)(float&, float&)) const
{
    size_t out_rank = std::max(m_rank, other.m_rank);
    size_t* out = new size_t[out_rank];

    // making new a and b shapes because we might need to pad the lower sized tensor
    size_t* shape_a     = new size_t[out_rank];
    size_t* shape_b     = new size_t[out_rank];
    size_t* stride_a    = new size_t[out_rank];
    size_t* stride_b    = new size_t[out_rank];
    size_t* pitch       = new size_t[out_rank];

    // 1 fill
    std::fill(shape_a,  shape_a     + out_rank, 1);
    std::fill(shape_b,  shape_b     + out_rank, 1);
    std::fill(stride_a, stride_a    + out_rank, 1);
    std::fill(stride_b, stride_b    + out_rank, 1);
    std::fill(pitch,    pitch       + out_rank, 1);

    // if we need to pad other e.g. [2, 3, 4] and [3, 4] -> [2, 3, 4] and [1, 3, 4]
    if (m_rank > other.m_rank)
        for (size_t i = 0; i < out_rank; i++)
        {
            shape_a[i] = m_shape[i];
            if (i >= (m_rank - other.m_rank))
                shape_b[i] = other.m_shape[i - (m_rank - other.m_rank)];
        }
    else if (m_rank < other.m_rank)
        for (size_t i = 0; i < out_rank; i++)
        {
            shape_b[i] = other.m_shape[i];
            if (i >= (other.m_rank - m_rank))
                shape_a[i] = m_shape[i - (other.m_rank - m_rank)];
        }

    // if no need to pad
    else
        for (size_t i = 0; i < out_rank; i++)
        {
            shape_a[i] = m_shape[i];
            shape_b[i] = other.m_shape[i];
        }

    // ensure shapes match
    for (size_t i = 0; i < out_rank; i++)
    {
        if (!(shape_a[i] == shape_b[i] || shape_a[i] == 1 || shape_b[i] == 1))
            throw std::invalid_argument("matrix size mismatch [T2]");
        out[i] = std::max(shape_a[i], shape_b[i]);
    }

    // our out tensor
    Tensor c = Tensor::create(out, out_rank);

    for (int i = out_rank-2; i >= 0; i--)
    {
        stride_a[i] = stride_a[i + 1]   * shape_a[i + 1];
        stride_b[i] = stride_b[i + 1]   * shape_b[i + 1];
        pitch[i]    = pitch[i + 1]      * out[i + 1];
    }

    // broadcasting: if size==1 on an axis that stride must be 0
    for (size_t i = 0; i < out_rank; i++)
    {
        if (shape_a[i] == 1) stride_a[i] = 0;
        if (shape_b[i] == 1) stride_b[i] = 0;
    }

    float* p_a = m_tensor;
    float* p_b = other.m_tensor;
    float* p_c = c.m_tensor;

    size_t total = 1;
    for (size_t i = 0; i < out_rank; ++i)
        total *= out[i];

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < total; i++)
    {
        size_t offset_a = 0, offset_b = 0, rem = i;
        for (size_t ax = 0; ax < out_rank; ax++)
        {
            const size_t idx = rem / pitch[ax];
            rem %= pitch[ax];
            offset_a += idx * stride_a[ax];
            offset_b += idx * stride_b[ax];
        }
        p_c[i] = f(p_a[offset_a], p_b[offset_b]);
    }

    delete[] out;
    delete[] shape_a;
    delete[] shape_b;
    delete[] stride_a;
    delete[] stride_b;
    delete[] pitch;
    return c;
}

Tensor Tensor::ops(const float& scalar, float (*f)(float&, const float&)) const
{   
    
    Tensor t = Tensor(*this);
    float* a = this->m_tensor;
    float* c = t.m_tensor;

    for (size_t i = 0; i < m_size; i++)  c[i] = f(a[i], scalar);
    
    return t;
}

Tensor& Tensor::ops_eq(const float& scalar, float (*f)(float&, const float&))
{   
    float* a = this->m_tensor;
    for (size_t i = 0; i < m_size; i++)
        a[i] = f(a[i], scalar);
    return *this;
}

Tensor& Tensor::operator=(const Tensor& other)
{
    if (this != &other) 
    {
        m_rank = other.m_rank; 
        m_size = other.m_size;

        delete[] m_shape;
        delete[] m_tensor;
        m_shape = new size_t[m_rank];
        m_tensor = new float[m_size];

        std::memcpy(m_shape, other.m_shape, sizeof(size_t)*m_rank);
        std::memcpy(m_tensor, other.m_tensor, sizeof(float)*m_size);
    }
    return *this;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept
{
    m_rank = other.m_rank;
    m_size = other.m_size;

    delete[] m_shape;
    delete[] m_tensor;

    m_shape = other.m_shape;
    m_tensor = other.m_tensor;

    other.m_shape = nullptr;
    other.m_tensor = nullptr;
    
    return *this;
}

Tensor::Tensor(Tensor&& other) noexcept
{
    m_rank = other.m_rank;
    m_size = other.m_size;
    m_shape = other.m_shape;
    m_tensor = other.m_tensor;

    other.m_shape = nullptr;
    other.m_tensor = nullptr;
}

Tensor::Tensor(const Tensor& other) // copy constructor
    :
        m_rank(other.m_rank),
        m_size(other.m_size),
        m_shape(new size_t[other.m_rank]),
        m_tensor(new float[other.m_size])
{
    std::memcpy(m_shape, other.m_shape, sizeof(size_t)*m_rank);
    std::memcpy(m_tensor, other.m_tensor, sizeof(float)*m_size);
}

// when the scalar is in front
Tensor operator+(float s, const Tensor& t) { return t + s; }
Tensor operator-(float s, const Tensor& t) { return (t * - 1) + s; }
Tensor operator*(float s, const Tensor& t) { return t * s; }
Tensor operator/(float s, const Tensor& t) { return t.ops(s, [](float& a, const float& b){ return b / a; }); } // here makes sure to put tensor in denom
