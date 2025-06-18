#ifndef TENSOR_H
#define TENSOR_H

#include <iostream>

using matrix2D = std::vector<std::vector<double>>;
using matrix3D = std::vector<matrix2D>;

class Tensor 
{
    public:

        int batch = 1;
        int row = 1;
        int col = 1;
        int rank = 0;
        std::array<int, 3> shape;
        std::unique_ptr<double[]> tensor;

        Tensor() : batch(0), row(0), col(0), rank(0), shape{0, 0, 0}, tensor(nullptr) {}
        
        Tensor(const matrix3D& input) 
        : batch(static_cast<int>((input).size())), row(static_cast<int>((input)[0].size())), 
          col(static_cast<int>((input)[0][0].size())), rank(3), shape({batch, row, col}),
          tensor(std::make_unique<double[]>(batch*row*col))
        {
            if (batch == 0) throw std::invalid_argument("empty tensor");
            
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
        : batch(1), row(static_cast<int>((input).size())), 
          col(static_cast<int>((input)[0].size())), rank(2), shape({1, row, col}),
          tensor(std::make_unique<double[]>(row*col))
        {
            if (row == 0) throw std::invalid_argument("empty tensor");

            // TO DO: use multithreading here
            for (int ri = 0; ri < row; ri++) {
                for (int ci = 0; ci < col; ci++) {
                    index(ri, ci) = input[ri][ci];
                    
                }
            }

        };

        Tensor(int b, int r, int c)
        : batch(b), row(r), col(c), rank(3), shape({batch, row, col}),
          tensor(std::make_unique<double[]>(b*r*c))
        {

        }

        Tensor(int r, int c) 
        : batch(1), row(r), col(c), rank(2), shape({1, row, col}),
          tensor(std::make_unique<double[]>(r*c))
        {
        
        }

        double& index(int b, int r, int c) 
        {
            return tensor[(b*row + r)*col + c];
        }

        // overload for read only access
        double index(int b,int r,int c) const 
        {
            return tensor[(b*row + r)*col + c];
        }

        double& index(int r, int c) 
        {
            return tensor[r*col + c];
        }

        // overload for read only access
        double index(int r, int c) const
        {
            return tensor[r*col + c];
        }

        // bc unique ptr forbids copying
        Tensor(Tensor&&) = default; // move constructor
        Tensor& operator=(Tensor&&) = default; // move assignment
        Tensor& operator=(const Tensor&) = delete; // copy assignment

        // Tensor(const Tensor&) = delete; // copy constructor
        Tensor(const Tensor& other) // copy constructor
            :
                batch(other.batch),
                row(other.row),
                col(other.col),
                rank(other.rank),
                shape(other.shape),
                tensor(std::make_unique<double[]>(batch*row*col))
        {
            std::memcpy(tensor.get(), other.tensor.get(), sizeof(double)*batch*row*col);
        }

};

#endif