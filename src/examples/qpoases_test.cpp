#include <iostream>
#include <optimization_lib/QpoasesOptimizer.hpp>

using namespace optimization_lib;

int main(int argc, char const* argv[])
{
    int nV = 2, nC = 1;

    Eigen::MatrixXd H(nV, nV), A(nC, nV);
    Eigen::VectorXd g(nV), lbA(nC), ubA(nC), lb(nV), ub(nV);

    H << 1.0, 0.0,
        0.0, 0.5;
    A << 1.0, 1.0;
    g << 1.5, 1.0;
    lb << 0.5, -2.0;
    ub << 5.0, 2.0;
    lbA << -1.0;
    ubA << 2.0;

    QpoasesOptimizer myqp;
    myqp
        .setHessianMatrix(H)
        .setGradientVector(g)
        .setLinearConstraints(A)
        .setLinearConstraintsBoundaries(lbA, ubA)
        .setVariablesBoundaries(lb, ub)
        .init();

    std::cout << "init: " << myqp.solution().transpose() << std::endl;

    g = Eigen::Vector2d(1.0, 1.5);
    lb = Eigen::Vector2d(0.0, -1.0);
    ub = Eigen::Vector2d(5.0, -0.5);
    lbA[0] = -2.0;
    ubA[0] = 1.0;

    myqp
        .setGradientVector(g)
        .setLinearConstraintsBoundaries(lbA, ubA)
        .setVariablesBoundaries(lb, ub)
        .setRecalculation(10)
        .optimize();

    std::cout << "hotstart: " << myqp.solution().transpose() << std::endl;

    return 0;
}
