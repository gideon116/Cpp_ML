#ifndef TENSOR_H
#define TENSOR_H

#include <iostream>
#include <cstring>
#include <memory>

class Tensor 
{
    public:

        size_t rank = 0; size_t tot_size = 0;
        std::unique_ptr<size_t[]> shape;
        std::unique_ptr<float[]> tensor;

        // default 
        Tensor() : rank(0), tot_size(0), shape(nullptr), tensor(nullptr) {}

        // create from scratch based on shape (init list or arr)
        static Tensor create(const std::initializer_list<size_t>& shape) { return Tensor(shape, 'C'); }
        static Tensor create(const size_t shape[], const size_t& a_len) { return Tensor(shape, 'C', a_len); }
        
        Tensor(const std::initializer_list<size_t>& in_shape, const char&);
        // this is soooo repetetive, MAKE SURE to COMBINE it with the last method somehow
        Tensor(const size_t in_shape[], const char&, const size_t& carray_len);

        // create from nested init list
        Tensor(const std::initializer_list<Tensor>& vs);
        Tensor(const std::initializer_list<float>& input);
     
        // change by index
        float& index(const size_t params[]);
        // overload for read only access
        float index(const size_t params[]) const;

        void print_shape();

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
        void getArr(const float& d, size_t& l)
        { tensor[l++] = d; }
        template<typename T>
        void getArr(const std::initializer_list<T>& vec, size_t& l)
        { for (const T& i : vec) getArr(i, l); }

        void getShape(const float& d, size_t& level) {}
        template<typename T>
        void getShape(const std::initializer_list<T>& vec, size_t& level)
        {   
            shape[level++] = static_cast<size_t>(vec.size());
            getShape(*vec.begin(), level);
        }

        void getRank(const float& d) {}
        template<typename T>
        void getRank(const std::initializer_list<T>& vec)
        { rank++; getRank(*vec.begin()); }
    };

// when the scalar is in front
Tensor operator+(float s, const Tensor& t);
Tensor operator-(float s, const Tensor& t);
Tensor operator*(float s, const Tensor& t);
Tensor operator/(float s, const Tensor& t);

#endif