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

        Tensor() : batch(1), row(0), col(0), rank(0), tot_size(0), shape(nullptr), index_helper(nullptr), tensor(nullptr) {}

        Tensor(std::initializer_list<double> ds)
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

        Tensor(const std::initializer_list<Tensor>& vs)
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

        // create from scratch
        static Tensor create(std::initializer_list<int> shape) { return Tensor(shape, create_tag{}); }
        static Tensor create(std::vector<int> shape) { return Tensor(shape, create_tag{}); }

        // change by index
        double& index(const std::vector<size_t>& params)
        {
            if (static_cast<int>(params.size()) != rank) throw std::invalid_argument("requested shape does not match tensor");
            
            int val = params[rank - 1];
            for (int i = 0; i < (rank - 1); i++) val += params[i] * index_helper[i];
            return tensor[val];
        }

        // overload for read only access
        double index(const std::vector<size_t>& params) const
        {
            if (static_cast<int>(params.size()) != rank) throw std::invalid_argument("requested shape does not match tensor");
            
            int val = params[rank - 1];
            for (int i = 0; i < (rank - 1); i++) val += params[i] * index_helper[i];
            return tensor[val];
        }

        // bc unique ptr forbids copying and rule of 5
        Tensor(Tensor&&) = default; // move constructor
        Tensor& operator=(Tensor&&) = default; // move assignment
        Tensor& operator=(const Tensor& other)  // copy assignment
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
        Tensor(const Tensor& other) // copy constructor
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

        // init helpers
        void getArr(const double& d, int& l) { tensor[l] = d; l += 1; }
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
            level += 1;
            getShape(vec[0], level);
        }

        void getRank(const double& d){}
        template<typename T>
        void getRank(const T& vec) { rank += 1; getRank(vec[0]); }
    
    private:
        struct create_tag {};
        template<typename T>
        Tensor(T in_shape, create_tag)
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

    };

// when the scalar is in front
inline Tensor operator+(double s, const Tensor& t) { return t + s; }
inline Tensor operator-(double s, const Tensor& t) { return (t * - 1) + s; }
inline Tensor operator*(double s, const Tensor& t) { return t * s; }
inline Tensor operator/(double s, const Tensor& t) { return t.ops(s, 'D'); }

#endif
