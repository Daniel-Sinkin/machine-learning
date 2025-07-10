#include <iostream>
#include <print>

#include <Eigen/Dense>

#include "common.hpp"

int main() {
    using namespace Eigen;
    MatrixXd M(2, 2);
    M << 3, -1,
        2.5, M(1, 0) + M(0, 1);

    VectorXd v(3);
    v << 1, 2, 3;

    VectorXd result = M * v.head(2);

    std::cout << "M:\n"
              << M << "\nresult:\n"
              << result << '\n';
    return 0;
}