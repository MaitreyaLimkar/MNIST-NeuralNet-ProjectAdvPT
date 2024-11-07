
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


void test_constructor(std::vector< std::pair< bool, std::string > >& results)
{
    Tensor< int > a;
    Tensor< int > b({3, 5, 8});
    Tensor< int > c({3, 5, 8}, 42);
    Tensor< int > d({3, 5, 8, 6, 2, 8, 2, 7, 7, 8}, 42);

    results.push_back({c.numElements() == 3 * 5 * 8, "test_constructor: correct number of elements"});
    results.push_back({a({}) == 0, "test_constructor: correct initialization"});
    results.push_back({b({2, 1, 1}) == 0, "test_constructor: correct initialization"});
    results.push_back({c({2, 1, 1}) == 42, "test_constructor: correct initialization"});
    results.push_back({d.rank() == 10, "test_constructor: correct rank"});

    Tensor< float > e;
    Tensor< double > f({4, 5, 6, 7});
    Tensor< bool > g({3, 3}, true);
}

void test_move(std::vector< std::pair< bool, std::string > >& results)
{
    {
        Tensor< int > a({2, 2, 2}, 2);
        auto b = a;

        results.push_back({a == b, "test_move: copy-assignment"});

        auto c = std::move(a);
        Tensor< int > d;

        results.push_back({a == d, "test_move: correct state after move"});
        results.push_back({c == b, "test_move: move succeeded"});

        a = b;
        Tensor< int > e(std::move(a));
        Tensor< int > f;

        results.push_back({a == f, "test_move: correct state after move"});
        results.push_back({e == b, "test_move: move succeeded"});
    }
}

void test_access(std::vector< std::pair< bool, std::string > >& results)
{
    Tensor< int > a;
    a({}) = 444;
    results.push_back({a({}) == 444, "test_constructor: correct access rank 0"});
}

void test_fileio(std::vector< std::pair< bool, std::string > >& results)
{
    auto a = readTensorFromFile< int >("data/tensor_01");
    results.push_back({a.rank() == 2, "test_io: tensor 01 correct rank"});
    results.push_back({a({0, 0}) == 1, "test_io: tensor 01 correct entry"});
    results.push_back({a({0, 1}) == 0, "test_io: tensor 01 correct entry"});
    results.push_back({a({1, 0}) == 0, "test_io: tensor 01 correct entry"});
    results.push_back({a({1, 1}) == -1, "test_io: tensor 01 correct entry"});

    Tensor< int > b({2, 2, 2});
    b({0, 0, 0}) = 1;
    b({0, 0, 1}) = 2;
    b({0, 1, 0}) = 3;
    b({0, 1, 1}) = 4;
    b({1, 0, 0}) = 5;
    b({1, 0, 1}) = 6;
    b({1, 1, 0}) = 7;
    b({1, 1, 1}) = 8;

    writeTensorToFile< int >(b, "data/tensor_out");
    auto c = readTensorFromFile< int >("data/tensor_out");

    results.push_back({c({0, 0, 0}) == 1, "test_io: tensor write correct entry"});
    results.push_back({c({0, 0, 1}) == 2, "test_io: tensor write correct entry"});
    results.push_back({c({0, 1, 0}) == 3, "test_io: tensor write correct entry"});
    results.push_back({c({0, 1, 1}) == 4, "test_io: tensor write correct entry"});
    results.push_back({c({1, 0, 0}) == 5, "test_io: tensor write correct entry"});
    results.push_back({c({1, 0, 1}) == 6, "test_io: tensor write correct entry"});
    results.push_back({c({1, 1, 0}) == 7, "test_io: tensor write correct entry"});
    results.push_back({c({1, 1, 1}) == 8, "test_io: tensor write correct entry"});

    auto d = readTensorFromFile< int >("data/tensor_02");

    results.push_back({c == d, "test_io: tensor read/write correct"});
}

int main()
{
    std::vector< std::pair< bool, std::string > > results;

    test_constructor(results);
    test_move(results);
    test_access(results);
    test_fileio(results);

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