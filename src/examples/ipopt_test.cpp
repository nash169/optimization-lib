#include <optimization_lib/IpoptOptimizer.hpp>

#include <iostream>
#include <memory>

using namespace Ipopt;
using namespace optimization_lib;

double objective(const Eigen::VectorXd& x)
{
    return x(0) * x(3) * (x(0) + x(1) + x(2)) + x(2);
}

double constraint1(const Eigen::VectorXd& x)
{
    return x(0) * x(1) * x(2) * x(3);
}

double constraint2(const Eigen::VectorXd& x)
{
    return x(0) * x(0) + x(1) * x(1) + x(2) * x(2) + x(3) * x(3);
}

Eigen::VectorXd gradientObjective(const Eigen::VectorXd& x)
{
    Eigen::VectorXd grad(4);
    grad << x[0] * x[3] + x[3] * (x[0] + x[1] + x[2]),
        x[0] * x[3],
        x[0] * x[3] + 1,
        x[0] * (x[0] + x[1] + x[2]);

    return grad;
}

Eigen::SparseMatrix<double> jacobianConstraints(const Eigen::VectorXd& x)
{
    typedef Eigen::Triplet<double> T;
    std::vector<T> tripletList;
    tripletList.reserve(8);

    tripletList.push_back(T(0, 0, x[1] * x[2] * x[3]));
    tripletList.push_back(T(0, 1, x[0] * x[2] * x[3]));
    tripletList.push_back(T(0, 2, x[0] * x[1] * x[3]));
    tripletList.push_back(T(0, 3, x[0] * x[1] * x[2]));
    tripletList.push_back(T(1, 0, 2 * x[0]));
    tripletList.push_back(T(1, 1, 2 * x[1]));
    tripletList.push_back(T(1, 2, 2 * x[2]));
    tripletList.push_back(T(1, 3, 2 * x[3]));

    Eigen::SparseMatrix<double> mat(2, 4);
    mat.setFromTriplets(tripletList.begin(), tripletList.end());

    return mat;
}

Eigen::SparseMatrix<double> hessianObjective(const Eigen::VectorXd& x)
{
    typedef Eigen::Triplet<double> T;
    std::vector<T> tripletList;
    tripletList.reserve(6);

    tripletList.push_back(T(0, 0, 2 * x[3]));

    tripletList.push_back(T(1, 0, x[3]));
    tripletList.push_back(T(2, 0, x[3]));
    tripletList.push_back(T(3, 0, 2 * x[0] + x[1] + x[2]));
    tripletList.push_back(T(3, 1, x[0]));
    tripletList.push_back(T(3, 2, x[0]));

    Eigen::SparseMatrix<double> mat(4, 4);
    mat.setFromTriplets(tripletList.begin(), tripletList.end());

    return mat;
}

Eigen::SparseMatrix<double> hessianConstraint1(const Eigen::VectorXd& x)
{
    typedef Eigen::Triplet<double> T;
    std::vector<T> tripletList;
    tripletList.reserve(6);

    tripletList.push_back(T(1, 0, x[2] * x[3]));
    tripletList.push_back(T(2, 0, x[1] * x[3]));
    tripletList.push_back(T(2, 1, x[0] * x[3]));
    tripletList.push_back(T(3, 0, x[1] * x[2]));
    tripletList.push_back(T(3, 1, x[0] * x[2]));
    tripletList.push_back(T(3, 2, x[0] * x[1]));

    Eigen::SparseMatrix<double> mat(4, 4);
    mat.setFromTriplets(tripletList.begin(), tripletList.end());

    return mat;
}

Eigen::SparseMatrix<double> hessianConstraint2(const Eigen::VectorXd& x)
{
    typedef Eigen::Triplet<double> T;
    std::vector<T> tripletList;
    tripletList.reserve(4);

    tripletList.push_back(T(0, 0, 2));
    tripletList.push_back(T(1, 1, 2));
    tripletList.push_back(T(2, 2, 2));
    tripletList.push_back(T(3, 3, 2));

    Eigen::SparseMatrix<double> mat(4, 4);
    mat.setFromTriplets(tripletList.begin(), tripletList.end());

    return mat;
}

using namespace Ipopt;

int main(int, char**)
{
    // Starting point
    Eigen::Vector4d x0(1, 5, 5, 1);

    // Create an instance of your nlp...
    SmartPtr<IpoptOptimizer> mynlp = new IpoptOptimizer();
    // SmartPtr<IpoptOptimizer> mynlp(new IpoptOptimizer());

    // Set optimization problem
    (*mynlp)
        .setDimension(4)
        .setStartingPoint(x0)
        .addVariablesBounds(0, 1, 5, 1, 1, 5, 2, 1, 5, 3, 1, 5)
        .setObjective(objective)
        .setObjectiveGradient(gradientObjective)
        .setObjectiveHessian(hessianObjective)
        .addConstraints(constraint1, 25, 2e19, constraint2, 40, 40)
        .setConstraintsJacobian(jacobianConstraints)
        .addConstraintsHessians(hessianConstraint1, hessianConstraint2);

    return (int)mynlp->optimize();
}
