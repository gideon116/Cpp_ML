#ifndef TENSOR_H
#define TENSOR_H

#include <iostream>

using matrix2D = std::vector<std::vector<double>>;
using matrix3D = std::vector<matrix2D>;
using matrix4D = std::vector<matrix3D>;

class Tensor 
{
    public:

        int batch = 0; int row = 0; int col = 0; int rank = 0;
        std::unique_ptr<int[]> shape;
        std::unique_ptr<double[]> tensor;

        Tensor() : batch(0), row(0), col(0), rank(0), shape(nullptr), tensor(nullptr) {}

        // create from nested vector
        Tensor(const matrix4D& input) 
            :   batch(static_cast<int>((input).size()) * static_cast<int>((input)[0].size())), // all non col and row dims -> batch
                row(static_cast<int>((input)[0][0].size())), 
                col(static_cast<int>((input)[0][0][0].size())), rank(4), shape(std::make_unique<int[]>(4)),
                tensor(std::make_unique<double[]>(batch*row*col))
        {
            if (batch == 0) throw std::invalid_argument("empty tensor");

            shape[0] = static_cast<int>((input).size());
            shape[1] = static_cast<int>((input)[0].size());
            shape[2] = row;
            shape[3] = col;

            // TO DO: use multithreading here
            for (int bi = 0; bi < batch; bi++) {
                for (int ri = 0; ri < row; ri++) {
                    for (int ci = 0; ci < col; ci++) {
                        index(bi, ri, ci) = input[bi/shape[0]][bi%shape[0]][ri][ci];
                    }
                }
            }

        };

        Tensor(const matrix3D& input) 
            :   batch(static_cast<int>((input).size())), row(static_cast<int>((input)[0].size())), 
                col(static_cast<int>((input)[0][0].size())), rank(3), shape(std::make_unique<int[]>(3)),
                tensor(std::make_unique<double[]>(batch*row*col))
        {
            if (batch == 0) throw std::invalid_argument("empty tensor");

            shape[0] = batch;
            shape[1] = row;
            shape[2] = col;

            // TO DO: use multithreading here
            for (int bi = 0; bi < batch; bi++) {
                for (int ri = 0; ri < row; ri++) {
                    for (int ci = 0; ci < col; ci++) {
                        index(bi, ri, ci) = input[bi][ri][ci];
                    }
                }
            }

        };

        Tensor(const matrix2D& input) 
            :   batch(1), row(static_cast<int>((input).size())), 
                col(static_cast<int>((input)[0].size())), rank(2), shape(std::make_unique<int[]>(2)),
                tensor(std::make_unique<double[]>(row*col))
        {
            if (row == 0) throw std::invalid_argument("empty tensor");

            shape[0] = row;
            shape[1] = col;

            // TO DO: use multithreading here
            for (int ri = 0; ri < row; ri++) {
                for (int ci = 0; ci < col; ci++) {
                    index(0, ri, ci) = input[ri][ci];
                    
                }
            }

        };

        // create from scratch
        Tensor(std::vector<int> in_shape)
        {   
            rank = static_cast<int>(in_shape.size());
            batch = 1;
            for (int i = 0; i < rank - 2; i ++) batch *= in_shape[i];
            row = in_shape[rank - 2];
            col = in_shape[rank - 1];
            shape = std::make_unique<int[]>(rank);
            for (int i = 0; i < rank; i ++) shape[i] = in_shape[i];
            tensor = std::make_unique<double[]>(batch * row * col);
        }

        // change by index
        double& index(int b, int r, int c) { return tensor[(b*row + r)*col + c]; }

        // overload for read only access
        double index(int b, int r, int c) const { return tensor[(b*row + r)*col + c]; }

        // bc unique ptr forbids copying -> rule of 5
        Tensor(Tensor&&) = default; // move constructor
        Tensor& operator=(Tensor&&) = default; // move assignment
        // Tensor& operator=(const Tensor&) = delete; // copy assignment
        Tensor& operator=(const Tensor& other)  // copy assignment
        {
            if (this != &other) 
            {
                batch = other.batch; row = other.row; col = other.col; rank = other.rank;

                shape = std::make_unique<int[]>(rank);
                tensor = std::make_unique<double[]>(batch*row*col);

                std::memcpy(shape.get(), other.shape.get(), sizeof(int)*rank);
                std::memcpy(tensor.get(), other.tensor.get(), sizeof(double)*batch*row*col);
            }
            return *this;
        }
        // Tensor(const Tensor&) = delete; // copy constructor
        Tensor(const Tensor& other) // copy constructor
            :
                batch(other.batch),
                row(other.row),
                col(other.col),
                rank(other.rank),
                shape(std::make_unique<int[]>(rank)),
                tensor(std::make_unique<double[]>(batch*row*col))
        {
            std::memcpy(shape.get(), other.shape.get(), sizeof(int)*rank);
            std::memcpy(tensor.get(), other.tensor.get(), sizeof(double)*batch*row*col);
        }

        // operator overloads
        
        // Tensor operator overload helper
        Tensor ops(const Tensor& other, const char op) const
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

        // Tensor overload operators
        Tensor operator+(const Tensor& other) const { return ops(other, 'A'); }
        Tensor operator-(const Tensor& other) const { return ops(other, 'S'); }
        Tensor operator*(const Tensor& other) const { return ops(other, 'M'); }
        Tensor operator/(const Tensor& other) const { return ops(other, 'D'); }

        // scalar operator overload helper
        Tensor ops(const double scalar, const char op) const
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

        // overload operators
        Tensor operator+(const double scalar) const { return ops(scalar, 'A'); }
        Tensor operator-(const double scalar) const { return ops(scalar, 'S'); }
        Tensor operator*(const double scalar) const { return ops(scalar, 'M'); }
        Tensor operator/(const double scalar) const { return ops(scalar, 'D'); }

        // += is special
        Tensor& operator+=(const double scalar) 
        {   
            double* a = (this->tensor).get();
            for (size_t i = 0; i < batch * row * col; i++) a[i] += scalar;
            return *this;
        }

    };

// then the scalar is in front
inline Tensor operator+(double s, const Tensor& t) { return t + s; }
inline Tensor operator-(double s, const Tensor& t) { return (t * - 1) + s; }
inline Tensor operator*(double s, const Tensor& t) { return t * s; }
inline Tensor operator/(double s, const Tensor& t) { return t.ops(s, 'D'); }

#endif
