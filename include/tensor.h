#ifndef TENSOR_H
#define TENSOR_H

#include <iostream>
#include <cstring>
#include <memory>

class Tensor 
{
    public:

        size_t batch = 1; size_t row = 0; size_t col = 0; size_t rank = 0; size_t tot_size = 0;
        std::unique_ptr<size_t[]> shape;
        std::unique_ptr<size_t[]> index_helper;
        std::unique_ptr<float[]> tensor;

        // default 
        Tensor() : batch(1), row(0), col(0), rank(0), tot_size(0), shape(nullptr), index_helper(nullptr), tensor(nullptr) {}

        // create from init list
        Tensor(std::initializer_list<float> ds);
        Tensor(const std::initializer_list<Tensor>& vs);
        
        // create from scratch based on shape (init list or arr)
        static Tensor create(std::initializer_list<int> shape) { return Tensor(shape, 'C'); }
        static Tensor create(size_t shape[], size_t a_len) { return Tensor(shape, 'C', a_len); }

        // create from nested init list
        template<typename TENSOR>
        Tensor(const TENSOR& input) 
        {   
            int level = 0;
            getRank(input);
            if (rank == 0) throw std::invalid_argument("need at least one dim");

            shape = std::make_unique<size_t[]>(rank);
            getShape(input, level);

            level = 0;
            tot_size = 1;
            for (size_t i = 0; i < rank; i++) tot_size *= shape[i];
            tensor = std::make_unique<float[]>(tot_size);
            getArr(input, level);

            index_helper = std::make_unique<size_t[]>(rank - 1);
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
            rank = static_cast<size_t>(in_shape.size());
            if (rank <= 0) throw std::invalid_argument("need at least one dimension");

            shape = std::make_unique<size_t[]>(rank);
            {
                size_t i = 0; for (int s : in_shape) shape[i++] = s;
            }

            tot_size = 1;
            for (size_t i = 0; i < rank; i++) tot_size *= shape[i];
            tensor = std::make_unique<float[]>(tot_size);

            if (rank >= 2)
            {
                batch = 1;
                for (size_t i = 0; i < rank - 2; i++) batch *= shape[i];
                row = shape[rank - 2];
                col = shape[rank - 1];

                index_helper = std::make_unique<size_t[]>(rank - 1);
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

        // this is soooo repetetive, MAKE SURE to COMBINE it with the last method somehow
        template<typename T>
        Tensor(T in_shape, char, int carray_len)
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

            if (rank >= 2)
            {
                batch = 1;
                for (size_t i = 0; i < rank - 2; i++) batch *= shape[i];
                row = shape[rank - 2];
                col = shape[rank - 1];

                index_helper = std::make_unique<size_t[]>(rank - 1);
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
        float& index(const size_t params[]);
        // overload for read only access
        float index(const size_t params[]) const;

        void printShape();

        // bc unique ptr forbids copying and rule of 5
        Tensor(Tensor&& other) noexcept; // move constructor
        Tensor(const Tensor& other); // copy constructor
        Tensor& operator=(Tensor&& other); // move assignment
        Tensor& operator=(const Tensor& other); // copy assignment
        
        // operator overloads
        // Tensor operator overload helper
        Tensor ops(const Tensor& other, float (*f)(float, float)) const;
        Tensor ops_bcast(const Tensor& other, float (*f)(float, float)) const;

        // Tensor overload operators
        Tensor operator+(const Tensor& other) const { return ops(other, [](float a, float b){ return a + b; }); }
        Tensor operator-(const Tensor& other) const { return ops(other, [](float a, float b){ return a - b; }); }
        Tensor operator*(const Tensor& other) const { return ops(other, [](float a, float b){ return a * b; }); }
        Tensor operator/(const Tensor& other) const { return ops(other, [](float a, float b){ return a / b; }); }

        // scalar operator overload helper
        Tensor ops(const float scalar, float (*f)(float, float)) const;

        // overload operators
        Tensor operator+(const float scalar) const { return ops(scalar, [](float a, float b){ return a + b; }); }
        Tensor operator-(const float scalar) const { return ops(scalar, [](float a, float b){ return a - b; }); }
        Tensor operator*(const float scalar) const { return ops(scalar, [](float a, float b){ return a * b; }); }
        Tensor operator/(const float scalar) const { return ops(scalar, [](float a, float b){ return a / b; }); }

        // += is special
        Tensor& operator+=(const float scalar);

        // init helpers
        void getArr(const float& d, int& l) { tensor[l] = d; l++; }
        template<typename T>
        void getArr(const T& vec, int& l)
        {   
            int vs = static_cast<int>(vec.size());
            for (int i = 0; i < vs; i++) getArr(vec[i], l);
        }

        void getShape(const float& d, int& level){}
        template<typename T>
        void getShape(const T& vec, int& level)
        {   
            shape[level] = static_cast<int>(vec.size());
            level++;
            getShape(vec[0], level);
        }

        void getRank(const float& d){}
        template<typename T>
        void getRank(const T& vec) { rank++; getRank(vec[0]); }
    };

// when the scalar is in front
Tensor operator+(float s, const Tensor& t);
Tensor operator-(float s, const Tensor& t);
Tensor operator*(float s, const Tensor& t);
Tensor operator/(float s, const Tensor& t);

#endif