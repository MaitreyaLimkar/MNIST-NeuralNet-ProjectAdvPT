#pragma once

#include "tensor.hpp"

template< typename ComponentType >
class Vector
{
public:
    // Default-constructor.
    Vector() = default;

    // Constructor for vector of certain size.
    explicit Vector(size_t size);

    // Constructor for vector of certain size with constant fill-value.
    Vector(size_t size, const ComponentType& fillValue);

    // Constructing vector from file.
    Vector(const std::string& filename);

    // Number of elements in this vector.
    [[nodiscard]] size_t size() const;

    // Element access function
    const ComponentType&
    operator()(size_t idx) const;

    // Element mutation function
    ComponentType&
    operator()(size_t idx);

    // Reference to internal tensor.
    Tensor< ComponentType >& tensor();

private:
    Tensor< ComponentType > tensor_;
};

template< typename ComponentType >
class Matrix
{
public:
    // Default-constructor.
    Matrix() = default;

    // Constructor for matrix of certain size.
    explicit Matrix(size_t rows, size_t cols);

    // Constructor for matrix of certain size with constant fill-value.
    Matrix(size_t rows, size_t cols, const ComponentType& fillValue);

    // Constructing matrix from file.
    Matrix(const std::string& filename);

    // Number of rows.
    [[nodiscard]] size_t rows() const;

    // Number of columns.
    [[nodiscard]] size_t cols() const;

    // Element access function
    const ComponentType&
    operator()(size_t row, size_t col) const;

    // Element mutation function
    ComponentType&
    operator()(size_t row, size_t col);

    // Reference to internal tensor.
    Tensor< ComponentType >& tensor();

private:
    Tensor< ComponentType > tensor_;
};

// TODO: Implement all methods.


// Performs a matrix-vector multiplication.
template< typename ComponentType >
Vector< ComponentType > matvec(const Matrix< ComponentType >& mat, const Vector< ComponentType >& vec)
{
    // TODO: Implement this.

}

