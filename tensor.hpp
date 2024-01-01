#pragma once

#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>
#include <numeric> 
#include <fstream>


template< class T >
concept Arithmetic = std::is_arithmetic_v< T >;

template< Arithmetic ComponentType >
class Tensor
{
public:

    Tensor() {
        data_ = new ComponentType[1];
        data_[0] = ComponentType{};  
    }

    Tensor(const std::vector<size_t>& matrix_shape) {
        rank_ = matrix_shape.size();
        this->matrix_shape = matrix_shape;
        elements_ = calculateNumElements();
        data_ = new ComponentType[elements_]; 
        std::fill(data_, data_ + elements_, ComponentType{});
    }

    explicit Tensor(const std::vector<size_t>& matrix_shape, const ComponentType& fillValue) {
        rank_ = matrix_shape.size();
        this->matrix_shape = matrix_shape;
        elements_ = calculateNumElements();
        data_ = new ComponentType[elements_];
        std::fill(data_, data_ + elements_, fillValue);
    }

    ~Tensor() {
        delete[] data_;
    }

    Tensor(const Tensor< ComponentType >& other);

    Tensor(Tensor< ComponentType >&& other) noexcept;

    Tensor& operator=(const Tensor< ComponentType >& other);

    Tensor& operator=(Tensor< ComponentType >&& other) noexcept;

    // Destructor
    // ~Tensor() = default;

    [[nodiscard]] size_t rank() const {
        return matrix_shape.size();
    };

    // [[nodiscard]] std::vector< size_t > matrix_shape() const {return matrix_shape;}


    [[nodiscard]] size_t numElements() const { 
        return elements_; 
    };

    const ComponentType&operator()(const std::vector< size_t >& idx) const{
        checkIndices(idx);
        size_t flatIndex = calculateFlatIndex(idx);
        return data_[flatIndex];
    };

    ComponentType& operator()(const std::vector< size_t >& idx){
       checkIndices(idx);
        size_t flatIndex = calculateFlatIndex(idx);
        return data_[flatIndex];
    };
    friend bool operator==<>(const Tensor<ComponentType>& a, const Tensor<ComponentType>& b);
    friend Tensor<ComponentType> readTensorFromFile<>(const std::string& filename);
    friend void writeTensorToFile<>(const Tensor<ComponentType>& tensor, const std::string& filename);





private:
    size_t rank_;
    std::vector<size_t> matrix_shape;
    size_t elements_;
    ComponentType* data_;

    size_t calculateNumElements() const {
        size_t numElements = 1;
        for (size_t dimensionSize : matrix_shape) {
            numElements *= dimensionSize;
        }
        return numElements;
    }
     void checkIndices(const std::vector<size_t>& idx) const {
        if (idx.size() != matrix_shape.size()) {
            throw std::out_of_range("Mismatched number of indices");
        }

        for (size_t i = 0; i < idx.size(); ++i) {
            if (idx[i] >= matrix_shape[i]) {
                throw std::out_of_range("Index out of bounds");
            }
        }
    }

    size_t calculateFlatIndex(const std::vector<size_t>& idx) const {
        size_t flatIndex = 0;
        size_t multiplier = 1;

        for (int i = matrix_shape.size() - 1; i >= 0; --i) {
            flatIndex += idx[i] * multiplier;
            multiplier *= matrix_shape[i];
        }

        return flatIndex;
    }
};

template< Arithmetic ComponentType >
Tensor<ComponentType>::Tensor(const Tensor< ComponentType >& other){
    rank_ = other.rank_;
    matrix_shape = other.matrix_shape;
    elements_ = other.elements_;
    data_ = new ComponentType[other.elements_];
    std::copy(other.data_,other.data_ + other.elements_,data_);
}

template <Arithmetic ComponentType>
Tensor<ComponentType>::Tensor(Tensor<ComponentType>&& other) noexcept {
    rank_ = other.rank_;
    matrix_shape = std::move(other.matrix_shape);
    elements_ = other.elements_;
    data_ = other.data_;
    other.rank_ = 0;
    other.elements_ = 0;
    other.data_ = nullptr;
}

template <Arithmetic ComponentType>
Tensor<ComponentType>& Tensor<ComponentType>::operator=(const Tensor<ComponentType>& other) {
    if (this != &other) {  
        delete[] data_;

        rank_ = other.rank_;
        matrix_shape = other.matrix_shape;
        elements_ = other.elements_;
        data_ = new ComponentType[elements_];
        std::copy(other.data_, other.data_ + elements_, data_);
    }
    return *this;
}

template <Arithmetic ComponentType>
Tensor<ComponentType>& Tensor<ComponentType>::operator=(Tensor<ComponentType>&& other) noexcept {
    if (this != &other) {  
        delete[] data_;

        rank_ = other.rank_;
        matrix_shape = std::move(other.matrix_shape);
        elements_ = other.elements_;
        data_ = other.data_;

        other.rank_ = 0;
        other.elements_ = 0;
        other.data_ = nullptr;
    }
    return *this;
}

template< Arithmetic ComponentType >
bool operator==(const Tensor< ComponentType >& a, const Tensor< ComponentType >& b)
{
    if (a.elements_ != b.elements_) {
        return false;
    }

    for (size_t i = 0; i < a.numElements(); ++i) {
        if (a.data_[i] != b.data_[i]) {
            return false;
        }
    }

    return true;

}

template< Arithmetic ComponentType >
std::ostream&operator<<(std::ostream& out, const Tensor< ComponentType >& tensor)
{
    out << "Tensor Shape: [";
    const auto shape = tensor.shape();
    for (size_t i = 0; i < shape.size(); ++i) {
        out << shape[i];
        if (i < shape.size() - 1) {
            out << ", ";
        }
    }
    out << "]" << std::endl;

    out << "Tensor Elements:" << std::endl;
    const auto& data = tensor.data_();
    const auto& rank = tensor.rank();
    printTensorElements(out, data, shape, rank, 0);

    return out;
}

template <typename ComponentType>
void printTensorElements(std::ostream& out, const ComponentType* data, const std::vector<size_t>& shape, size_t rank, size_t currentIndex) {
    if (rank == 0) {
        out << data[currentIndex] << std::endl;
        return;
    }

    out << "[";
    for (size_t i = 0; i < shape[currentIndex]; ++i) {
        printTensorElements(out, data, shape, rank - 1, currentIndex + 1);
        if (i < shape[currentIndex] - 1) {
            out << ", ";
        }
    }
    out << "]";
}

template< Arithmetic ComponentType >
Tensor<ComponentType> readTensorFromFile(const std::string& filename) {
    std::ifstream file(filename);
    
    if (!file.is_open()) {
        throw std::runtime_error("Error opening file: " + filename);
    }

    size_t rank;
    file >> rank;

    std::vector<size_t> shape(rank);
    for (size_t i = 0; i < rank; ++i) {
        file >> shape[i];
    }

    Tensor<ComponentType> tensor(shape);

    for (size_t i = 0; i < tensor.numElements(); ++i) {
        file >> tensor.data_[i];
    }

    return tensor;
}

template< Arithmetic ComponentType >
void writeTensorToFile(const Tensor<ComponentType>& tensor, const std::string& filename) {
    std::ofstream file(filename);
    
    if (!file.is_open()) {
        throw std::runtime_error("Error opening file: " + filename);
    }

    file << tensor.rank() << '\n';

    const auto& shape = tensor.matrix_shape;
    for (size_t i = 0; i < tensor.rank(); ++i) {
        file << shape[i] << '\n';
    }

    const auto& data = tensor.data_;
    for (size_t i = 0; i < tensor.numElements(); ++i) {
        file << data[i] << ' ';
    }

    std::cout << "Tensor written to file: " << filename << std::endl;
}
