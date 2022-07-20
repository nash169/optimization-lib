#ifndef OPTIMIZATIONLIB_ABSTRACTOPTIMIZER
#define OPTIMIZATIONLIB_ABSTRACTOPTIMIZER

#include <Eigen/Sparse>
#include <functional>
#include <tuple>
#include <utility>
#include <vector>

namespace optimization_lib {
    class AbstractOptimizer {
    public:
        AbstractOptimizer() : _n(1), _m(0), _x0(Eigen::VectorXd::Random(1)), _xSol(1)
        {
        }

        AbstractOptimizer& setDimension(const size_t& n)
        {
            // Set problem dimension
            _n = n;

            // Resize
            _x0.conservativeResize(n);
            _xSol.conservativeResize(n);

            // should be withing the constraits (maybe put this in the first ipopt function)
            _x0 = Eigen::VectorXd::Random(n);

            return *this;
        }

        AbstractOptimizer& setStartingPoint(const Eigen::VectorXd& x)
        {
            _x0 = x;

            return *this;
        }

        template <typename... Args>
        AbstractOptimizer& addVariablesBounds(const size_t i, const double& x_l, const double& x_u, Args... args)
        {
            _xC.push_back(std::make_tuple(i, x_l, x_u));

            if constexpr (sizeof...(args) > 0)
                addVariablesBounds(args...);

            return *this;
        }

        AbstractOptimizer& setObjective(const std::function<double(const Eigen::VectorXd&)>& f)
        {
            _f = f;

            return *this;
        }

        AbstractOptimizer& setObjectiveGradient(const std::function<Eigen::VectorXd(const Eigen::VectorXd&)>& grad)
        {
            _grad = grad;

            return *this;
        }

        AbstractOptimizer& setObjectiveHessian(const std::function<Eigen::SparseMatrix<double>(const Eigen::VectorXd&)>& hess)
        {
            _hess = hess;

            return *this;
        }

        template <typename... Args>
        AbstractOptimizer& addConstraints(const std::function<double(Eigen::VectorXd)>& g,
            const double& g_l = -2.0e19, const double& g_u = +2.0e19, Args... args)
        {
            // Constraint function
            _g.push_back(g);

            // Bounds (upper and lower bound the same for equality constraints)
            _gC.push_back(std::make_pair(g_l, g_u));

            // Increase number of constraints
            _m += 1;

            if constexpr (sizeof...(args) > 0)
                addConstraints(args...);

            return *this;
        }

        AbstractOptimizer& setConstraintsJacobian(const std::function<Eigen::SparseMatrix<double>(const Eigen::VectorXd&)>& jac)
        {
            _jac = jac;

            return *this;
        }

        template <typename... Args>
        AbstractOptimizer& addConstraintsHessians(const std::function<Eigen::SparseMatrix<double>(const Eigen::VectorXd&)>& hessConstraint, Args... args)
        {
            // Constraint function
            _hessConstraints.push_back(hessConstraint);

            if constexpr (sizeof...(args) > 0)
                addConstraintsHessians(args...);

            return *this;
        }

        virtual bool optimize() { return false; }

        virtual Eigen::VectorXd solution() { return _xSol; }

    protected:
        size_t _n, // Problem dimension
            _m; // Number of constraints

        // Starting point and solution
        Eigen::VectorXd _x0, _xSol;

        // Objective (maybe allow for multiple objectives)
        std::function<double(const Eigen::VectorXd&)> _f;

        // Objective gradient
        std::function<Eigen::VectorXd(const Eigen::VectorXd&)> _grad;

        // Objective hessian
        std::function<Eigen::SparseMatrix<double>(const Eigen::VectorXd&)> _hess;

        // Constraints
        std::vector<std::function<double(const Eigen::VectorXd&)>> _g;

        // Constraints jacobian
        std::function<Eigen::SparseMatrix<double>(const Eigen::VectorXd&)> _jac;

        // Constraints hessian
        std::vector<std::function<Eigen::SparseMatrix<double>(const Eigen::VectorXd&)>> _hessConstraints;

        // Variable bounds
        std::vector<std::tuple<size_t, double, double>> _xC;

        // Constraints bounds
        std::vector<std::pair<double, double>> _gC;

        // Extract rows indices
        int* rowMajorIndices(const Eigen::SparseMatrix<double, Eigen::RowMajor>& mat)
        {
            const int* outerindices = mat.outerIndexPtr();

            int* rowsindices = new int[mat.nonZeros()];

            int index = 0;

            for (size_t i = 0; i < mat.rows(); i++)
                for (size_t j = outerindices[i]; j < outerindices[i + 1]; j++) {
                    rowsindices[index] = i;
                    index++;
                }

            return rowsindices;
        }

        // Extract cols indices
        int* colMajorIndices(const Eigen::SparseMatrix<double, Eigen::ColMajor>& mat)
        {
            const int* outerindices = mat.outerIndexPtr();

            int* colsindices = new int[mat.nonZeros()];

            int index = 0;

            for (size_t i = 0; i < mat.cols(); i++)
                for (size_t j = outerindices[i]; j < outerindices[i + 1]; j++) {
                    colsindices[index] = i;
                    index++;
                }

            return colsindices;
        }
    };
} // namespace optimization_lib

#endif // OPTIMIZATIONLIB_ABSTRACTOPTIMIZER