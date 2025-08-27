#pragma once

#include <iostream>
#include <cstring>
#include <memory>

class Tensor 
{
public: // TODO : make these private and have getters
    size_t m_rank = 0; 
    size_t m_size = 0;
    size_t* m_shape = nullptr;
    float* m_tensor = nullptr;

public:

    // default 
    Tensor() : m_rank(0), m_size(0), m_shape(nullptr), m_tensor(nullptr) {}
    ~Tensor()
    {
        if (m_shape) delete[] m_shape;
        if (m_tensor) delete[] m_tensor;
    }

    // create from scratch based on shape (init list or arr)
    static Tensor create(const std::initializer_list<size_t>& shape)
        { return Tensor(shape, 'C'); }
    static Tensor create(const size_t shape[], const size_t& a_len)
        { return Tensor(shape, 'C', a_len); }
    
    Tensor(const std::initializer_list<size_t>& in_shape, const char&);
    // this is soooo repetetive, MAKE SURE to COMBINE it with the last method somehow
    Tensor(const size_t in_shape[], const char&, const size_t& carray_len);

    // create from nested init list
    Tensor(const std::initializer_list<Tensor>& vs);
    Tensor(const std::initializer_list<float>& input);

    void print_shape();
    void reshape(const size_t shape[], const size_t& rank);
    
    // change by index
    float& operator[](const std::initializer_list<size_t>& params);
    float& operator[](const size_t params[]);
    
    // overload for read only access
    float operator[](const std::initializer_list<size_t>& params) const;
    float operator[](const size_t params[]) const;

    Tensor operator[](const size_t& index); // TODO : DONT EVEN THINK ABOUT SHIPPING THIS

    Tensor(Tensor&& other) noexcept; // move constructor
    Tensor(const Tensor& other); // copy constructor
    Tensor& operator=(Tensor&& other) noexcept; // move assignment
    Tensor& operator=(const Tensor& other); // copy assignment
    
    // operator overloads
    // Tensor operator overload helper
    Tensor ops(const Tensor& other, float (*f)(float&, float&)) const;
    Tensor& ops_eq(const Tensor& other, float (*f)(float&, float&));
    Tensor ops_bcast(const Tensor& other, float (*f)(float&, float&)) const;

    // Tensor overload operators
    Tensor operator+(const Tensor& other) const
        { return ops(other, [](float& a, float& b){ return a + b; }); }
    Tensor operator-(const Tensor& other) const
        { return ops(other, [](float& a, float& b){ return a - b; }); }
    Tensor operator*(const Tensor& other) const
        { return ops(other, [](float& a, float& b){ return a * b; }); }
    Tensor operator/(const Tensor& other) const
        { return ops(other, [](float& a, float& b){ return a / b; }); }

    // Tensor overload operators (+= etc are special)
    Tensor& operator+=(const Tensor& other)
        { return ops_eq(other, [](float& a, float& b){ return a + b; }); }
    Tensor& operator-=(const Tensor& other)
        { return ops_eq(other, [](float& a, float& b){ return a - b; }); }
    Tensor& operator*=(const Tensor& other)
        { return ops_eq(other, [](float& a, float& b){ return a * b; }); }
    Tensor& operator/=(const Tensor& other)
        { return ops_eq(other, [](float& a, float& b){ return a / b; }); }

    // scalar operator overload helper
    Tensor ops(const float& scalar, float (*f)(float&, const float&)) const;
    Tensor& ops_eq(const float& scalar, float (*f)(float&, const float&));

    // overload operators
    Tensor operator+(const float& scalar) const
        { return ops(scalar, [](float& a, const float& b){ return a + b; }); }
    Tensor operator-(const float& scalar) const
        { return ops(scalar, [](float& a, const float& b){ return a - b; }); }
    Tensor operator*(const float& scalar) const
        { return ops(scalar, [](float& a, const float& b){ return a * b; }); }
    Tensor operator/(const float& scalar) const
        { return ops(scalar, [](float& a, const float& b){ return a / b; }); }

    // += etc are special
    Tensor& operator+=(const float& scalar)
        { return ops_eq(scalar, [](float& a, const float& b){ return a + b; }); }
    Tensor& operator-=(const float& scalar)
        { return ops_eq(scalar, [](float& a, const float& b){ return a - b; }); }
    Tensor& operator*=(const float& scalar)
        { return ops_eq(scalar, [](float& a, const float& b){ return a * b; }); }
    Tensor& operator/=(const float& scalar)
        { return ops_eq(scalar, [](float& a, const float& b){ return a / b; }); }

private:

    // init helpers
    void getArr(const float& d, size_t& l)
    { m_tensor[l++] = d; }
    template<typename T>
    void getArr(const std::initializer_list<T>& vec, size_t& l)
    { for (const T& i : vec) getArr(i, l); }

    void getShape(const float& d, size_t& level) {}
    template<typename T>
    void getShape(const std::initializer_list<T>& vec, size_t& level)
    {   
        m_shape[level++] = static_cast<size_t>(vec.size());
        getShape(*vec.begin(), level);
    }

    void getRank(const float& d) {}
    template<typename T>
    void getRank(const std::initializer_list<T>& vec)
    { m_rank++; getRank(*vec.begin()); }
};

// when the scalar is in front
Tensor operator+(float s, const Tensor& t);
Tensor operator-(float s, const Tensor& t);
Tensor operator*(float s, const Tensor& t);
Tensor operator/(float s, const Tensor& t);

