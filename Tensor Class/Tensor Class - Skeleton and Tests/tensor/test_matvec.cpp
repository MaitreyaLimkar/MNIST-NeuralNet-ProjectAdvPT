
#include "matvec.hpp"
#include "tensor.hpp"

void check(bool condition, const std::string& msg)
{
    if (!condition)
    {
        std::cout << "FAILED: " << msg << "\n";
    }
    else
    {
        std::cout << "PASSED: " << msg << "\n";
    }
}

void test_matvec(std::vector< std::pair< bool, std::string > >& results)
{

    Matrix< int > A("data/matrix");
    Vector< int > x("data/vector_in");
    Vector< int > y_read("data/vector_out");

    std::cout << A.tensor() << std::endl;
    std::cout << x.tensor() << std::endl;
    std::cout << y_read.tensor() << std::endl;

    auto y_comp = matvec(A, x);

    results.push_back({y_comp.tensor() == y_read.tensor(), "test_matvec: result equal to file"});
}

int main()
{
    std::vector< std::pair< bool, std::string > > results;

    test_matvec(results);

    size_t passed = 0;
    for (auto [condition, msg] : results)
    {
        check(condition, msg);
        if (condition)
        {
            passed++;
        }
    }

    std::cout << "--- " << passed << "/" << results.size() << " checks passed ---" << std::endl;

    return passed != results.size();
}