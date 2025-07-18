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

double& Tensor::index(const size_t params[])
{
    // TO DO: ADD CHECKS!!!!!
    // if (sizeof(params)/sizeof(params[0]) != rank) throw std::invalid_argument("requested shape does not match tensor");
    
    size_t val = params[rank-1];
    for (size_t i = 0; i < (rank - 1); i++) val += params[i] * index_helper[i];
    return tensor[val];
}

double Tensor::index(const size_t params[]) const
{
    // if (sizeof(params)/sizeof(params[0]) != rank) throw std::invalid_argument("requested shape does not match tensor");
    
    size_t val = params[rank-1];
    for (size_t i = 0; i < (rank - 1); i++) val += params[i] * index_helper[i];
    return tensor[val];
}

void Tensor::printShape()
{
    std::cout << "[ ";
    for (int i = 0; i < rank; i++) std::cout << shape[i] << " ";
    std::cout << "]";
    std::cout << "\n";

}

Tensor Tensor::ops(const Tensor& other, double (*f)(double, double)) const
{   
    // if we should broadcast one of the tensors because its like [a, b, c] and [1, 1, c] then run ops_bcast
    if (rank != other.rank) return ops_bcast(other, f);
    for (int i = 0; i < rank; i++)
        if (shape[i] != other.shape[i]) return ops_bcast(other, f);

    // no need to broadcast...
    Tensor t = Tensor(*this);
    double* a = (this->tensor).get();
    double* b = (other.tensor).get();
    double* c = (t.tensor).get();

    for (size_t i = 0; i < batch * row * col; i++) 
    {   
        c[i] = f(a[i], b[i]);
    }
    return t;
}

Tensor Tensor::ops_bcast(const Tensor& other, double (*f)(double, double)) const
{
    int out_rank = std::max(rank, other.rank);
    std::unique_ptr<int[]> out = std::make_unique<int[]>(out_rank);

    // making new a and b shapes because we might need to pad the lower sized tensor
    std::unique_ptr<int[]> shape_a = std::make_unique<int[]>(out_rank);
    std::unique_ptr<int[]> shape_b = std::make_unique<int[]>(out_rank);
    std::unique_ptr<int[]> stride_a = std::make_unique<int[]>(out_rank);
    std::unique_ptr<int[]> stride_b = std::make_unique<int[]>(out_rank);
    std::unique_ptr<int[]> pitch = std::make_unique<int[]>(out_rank);

    // 1 fill
    for (int i = 0; i < out_rank; i++) 
    {
        shape_a[i] = 1; shape_b[i] = 1; stride_a[i] = 1; stride_b[i] = 1; pitch[i] = 1;
    }

    // if we need to pad other e.g. [2, 3, 4] and [3, 4] -> [2, 3, 4] and [1, 3, 4]
    if (rank > other.rank)
        for (int i = 0; i < out_rank; i++)
        {
            shape_a[i] = shape[i];
            if (i >= (rank - other.rank)) shape_b[i] = other.shape[i - (rank - other.rank)];
        }
    else if (rank < other.rank)
        for (int i = 0; i < out_rank; i++)
        {
            shape_b[i] = other.shape[i];
            if (i >= (other.rank - rank)) shape_a[i] = shape[i - (other.rank - rank)];
        }

    // if no need to pad
    else
        for (int i = 0; i < out_rank; i++)
        {
            shape_a[i] = shape[i];
            shape_b[i] = other.shape[i];
        }

    // ensure shapes match
    for (int i = 0; i < out_rank; i++) {
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
    for (int i = 0; i < out_rank; i++)
        {
            if (shape_a[i] == 1) stride_a[i] = 0;
            if (shape_b[i] == 1) stride_b[i] = 0;
        }

    double* p_a = tensor.get();
    double* p_b = other.tensor.get();
    double* p_c = c.tensor.get();

    size_t total = 1;
    for (int i = 0; i < out_rank; ++i) total *= out[i];

    for (size_t lin = 0; lin < total; lin++)
    {
        size_t offA = 0, offB = 0, rem = lin;
        for (int ax = 0; ax < out_rank; ax++) {
            const size_t idx = rem / pitch[ax];
            rem %= pitch[ax];
            offA += idx * stride_a[ax];
            offB += idx * stride_b[ax];
        }
        p_c[lin] = f(p_a[offA], p_b[offB]);
    }
    return c;
}

Tensor Tensor::ops(const double scalar, double (*f)(double, double)) const
{   
    
    Tensor t = Tensor(*this);
    double* a = (this->tensor).get();
    double* c = (t.tensor).get();

    for (size_t i = 0; i < batch * row * col; i++)  c[i] = f(a[i], scalar);
    
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

// when the scalar is in front
Tensor operator+(double s, const Tensor& t) { return t + s; }
Tensor operator-(double s, const Tensor& t) { return (t * - 1) + s; }
Tensor operator*(double s, const Tensor& t) { return t * s; }
Tensor operator/(double s, const Tensor& t) { return t.ops(s, [](double a, double b){ return b / a; }); } // here makes sure to put tensor in denom
