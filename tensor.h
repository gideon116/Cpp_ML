#ifndef TENSOR_H
#define TENSOR_H

#include <iostream>

class Tensor 
{
    public:

        int batch = 1; int row = 0; int col = 0; int rank = 0; int tot_size = 0;
        std::unique_ptr<int[]> shape;
        std::unique_ptr<int[]> index_helper;
        std::unique_ptr<double[]> tensor;

        // default 
        Tensor() : batch(1), row(0), col(0), rank(0), tot_size(0), shape(nullptr), index_helper(nullptr), tensor(nullptr) {}

        // create from init list
        Tensor(std::initializer_list<double> ds);
        Tensor(const std::initializer_list<Tensor>& vs);
        
        // create from scratch based on shape (init list or vector)
        static Tensor create(std::initializer_list<int> shape) { return Tensor(shape, 'C'); }
        static Tensor create(std::vector<int> shape) { return Tensor(shape, 'C'); }

        // create from nested vector
        template<typename TENSOR>
        Tensor(const TENSOR& input) 
        {   
            int level = 0;
            getRank(input);
            if (rank == 0) throw std::invalid_argument("need at least one dim");

            shape = std::make_unique<int[]>(rank);
            getShape(input, level);

            level = 0;
            tot_size = 1;
            for (int i = 0; i < rank; i++) tot_size *= shape[i];
            tensor = std::make_unique<double[]>(tot_size);
            getArr(input, level);

            index_helper = std::make_unique<int[]>(rank - 1);
            index_helper[rank - 2] = shape[rank - 1];
            for (int i = rank - 3; i > -1; i--) index_helper[i] = shape[i + 1] * index_helper[i + 1];

            for (int i = rank - 1; i > -1; i--)
            {
                if (i == rank - 1) col = shape[i];
                else if (i == rank - 2) row = shape[i];
                else batch *= shape[i];  // all non col and row dims -> batch
            }
        };

        template<typename T>
        Tensor(T in_shape, char)
        {   
            rank = static_cast<int>(in_shape.size());
            if (rank <= 0) throw std::invalid_argument("need at least one dimension");

            shape = std::make_unique<int[]>(rank);
            {
                int i = 0;
                for (int s : in_shape) shape[i++] = s;
            }

            tot_size = 1;
            for (int i = 0; i < rank; i++) tot_size *= shape[i];
            tensor = std::make_unique<double[]>(tot_size);

            if (rank >= 2)
            {
                batch = 1;
                for (int i = 0; i < rank - 2; i++) batch *= shape[i];
                row = shape[rank - 2];
                col = shape[rank - 1];

                index_helper = std::make_unique<int[]>(rank - 1);
                index_helper[rank - 2] = shape[rank - 1];
                for (int i = rank - 3; i > -1; i--) index_helper[i] = shape[i + 1] * index_helper[i + 1];
            }
            else
            {
                batch = tot_size;
                row = col = 1;
                index_helper = nullptr;
            } 
        }
     
        // change by index
        double& index(const std::vector<size_t>& params);

        // overload for read only access
        double index(const std::vector<size_t>& params) const;

        void printShape();

        // bc unique ptr forbids copying and rule of 5
        Tensor(Tensor&&) = default; // move constructor
        Tensor& operator=(Tensor&&) = default; // move assignment
        Tensor& operator=(const Tensor& other); // copy assignment
        Tensor(const Tensor& other); // copy constructor

        // operator overloads
        // Tensor operator overload helper
        Tensor ops(const Tensor& other, const char op) const;

        // Tensor overload operators
        Tensor operator+(const Tensor& other) const { return ops(other, 'A'); }
        Tensor operator-(const Tensor& other) const { return ops(other, 'S'); }
        Tensor operator*(const Tensor& other) const { return ops(other, 'M'); }
        Tensor operator/(const Tensor& other) const { return ops(other, 'D'); }

        // scalar operator overload helper
        Tensor ops(const double scalar, const char op) const;

        // overload operators
        Tensor operator+(const double scalar) const { return ops(scalar, 'A'); }
        Tensor operator-(const double scalar) const { return ops(scalar, 'S'); }
        Tensor operator*(const double scalar) const { return ops(scalar, 'M'); }
        Tensor operator/(const double scalar) const { return ops(scalar, 'D'); }

        // += is special
        Tensor& operator+=(const double scalar);

        // init helpers
        void getArr(const double& d, int& l) { tensor[l] = d; l++; }
        template<typename T>
        void getArr(const T& vec, int& l)
        {   
            int vs = static_cast<int>(vec.size());
            for (int i = 0; i < vs; i++) getArr(vec[i], l);
        }

        void getShape(const double& d, int& level){}
        template<typename T>
        void getShape(const T& vec, int& level)
        {   
            shape[level] = static_cast<int>(vec.size());
            level++;
            getShape(vec[0], level);
        }

        void getRank(const double& d){}
        template<typename T>
        void getRank(const T& vec) { rank++; getRank(vec[0]); }
    };

// when the scalar is in front
inline Tensor operator+(double s, const Tensor& t) { return t + s; }
inline Tensor operator-(double s, const Tensor& t) { return (t * - 1) + s; }
inline Tensor operator*(double s, const Tensor& t) { return t * s; }
inline Tensor operator/(double s, const Tensor& t) { return t.ops(s, 'D'); }

#endif
