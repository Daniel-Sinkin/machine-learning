#include <cmath>
#include <iostream>
#include <print>

#include <Eigen/Dense>

#include "common.hpp"

using namespace Eigen;

double spd_determinant(const MatrixXd &A) {
    LLT<MatrixXd> llt(A);
    if (llt.info() != Success) {
        throw std::domain_error("Matrix is not positive definite.");
    }

    const auto &L = llt.matrixL();
    double log_det = 0.0;
    for (Index i = 0; i < L.rows(); ++i) {
        double diag = L(i, i);
        log_det += std::log(diag);
    }
    return std::exp(2.0 * log_det);
}

double N(const VectorXd &x, const VectorXd &mu, const MatrixXd &sigma) {
    const int D = x.size();

    if (mu.size() != D || sigma.rows() != D || sigma.cols() != D) {
        throw std::invalid_argument("Dimension mismatch: x, mu, and sigma must align.");
    }

    if (!sigma.isApprox(sigma.transpose())) {
        throw std::domain_error("Covariance matrix Sigma is not symmetric.");
    }

    LLT<MatrixXd> llt(sigma);
    if (llt.info() != Success) {
        throw std::domain_error("Covariance matrix Sigma is not positive definite.");
    }

    VectorXd diff = x - mu;
    double exponent = diff.transpose() * llt.solve(diff);
    double det_sigma = spd_determinant(sigma);

    double norm_const = std::pow(2.0 * M_PI, -0.5 * D) * std::pow(det_sigma, -0.5);
    return norm_const * std::exp(-0.5 * exponent);
}

int main() {
    VectorXd mu(2);
    mu << 0.0, 0.0;

    VectorXd x(2);
    x << 0.1, -0.1;

    MatrixXd sigma(2, 2);
    sigma << 2.0, 1.0,
        1.0, 1.0;

    std::cout << "mu = \n"
              << mu << "\n\n";
    std::cout << "x = \n"
              << x << "\n\n";
    std::cout << "sigma = \n"
              << sigma << "\n\n";

    try {
        double px = N(x, mu, sigma);
        std::cout << "N(x | mu, sigma) = " << px << "\n";
    } catch (const std::exception &e) {
        std::cerr << "Error evaluating N(x | mu, sigma): " << e.what() << "\n";
    }

    return 0;
}