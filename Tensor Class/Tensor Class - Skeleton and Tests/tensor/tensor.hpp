
#pragma once

#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>



template< class T >
concept Arithmetic = std::is_arithmetic_v< T >;

template< Arithmetic ComponentType >
class Tensor
{
public:
    // Constructs a tensor with rank = 0 and zero-initializes the element.
    Tensor();

    // Constructs a tensor with arbitrary shape and zero-initializes all elements.
    Tensor(const std::vector< size_t >& shape);

    // Constructs a tensor with arbitrary shape and fills it with the specified value.
    explicit Tensor(const std::vector< size_t >& shape, const ComponentType& fillValue);

    // Copy-constructor.
    Tensor(const Tensor< ComponentType >& other);

    // Move-constructor.
    Tensor(Tensor< ComponentType >&& other) noexcept;

    // Copy-assignment
    Tensor&
    operator=(const Tensor< ComponentType >& other);

    // Move-assignment
    Tensor&
    operator=(Tensor< ComponentType >&& other) noexcept;

    // Destructor
    ~Tensor() = default;

    // Returns the rank of the tensor.
    [[nodiscard]] size_t rank() const;

    // Returns the shape of the tensor.
    [[nodiscard]] std::vector< size_t > shape() const;

    // Returns the number of elements of this tensor.
    [[nodiscard]] size_t numElements() const;

    // Element access function
    const ComponentType&
    operator()(const std::vector< size_t >& idx) const;

    // Element mutation function
    ComponentType&
    operator()(const std::vector< size_t >& idx);

private:
    // TODO: Probably you need some members here...

};

// TODO: Implement all methods of the Tensor class template.


// Returns true if the shapes and all elements of both tensors are equal.
template< Arithmetic ComponentType >
bool operator==(const Tensor< ComponentType >& a, const Tensor< ComponentType >& b)
{
    // TODO: Implement this comparison.

}

// Pretty-prints the tensor to stdout.
// This is not necessary (and not covered by the tests) but nice to have, also for debugging (and for exercise of course...).
template< Arithmetic ComponentType >
std::ostream&
operator<<(std::ostream& out, const Tensor< ComponentType >& tensor)
{
    // TODO (optional): Implement some nice stdout printer for debugging/exercise.

}

// Reads a tensor from file.
template< Arithmetic ComponentType >
Tensor< ComponentType > readTensorFromFile(const std::string& filename)
{
    // TODO: Implement this function to read in tensors from file.
    //       The format is defined in the instructions.

}

// Writes a tensor to file.
template< Arithmetic ComponentType >
void writeTensorToFile(const Tensor< ComponentType >& tensor, const std::string& filename)
{
    // TODO: Implement this function to write tensors to file.
    //       The format is defined in the instructions.

}
