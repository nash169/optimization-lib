#ifndef OPTIMIZATIONLIB_QPOASESOPTIMIZER
#define OPTIMIZATIONLIB_QPOASESOPTIMIZER

#include "optimization_lib/AbstractOptimizer.hpp"
#include <memory>
#include <qpOASES.hpp>
#include <type_traits>

using namespace qpOASES;

namespace optimization_lib {
    template <typename Solver = QProblem>
    class QpoasesOptimizer : public AbstractOptimizer {
    public:
        QpoasesOptimizer()
        {
            _nWSR = 10;
        }

        QpoasesOptimizer& setHessianMatrix(const Eigen::MatrixXd& H)
        {
            _H = H;
            return *this;
        }

        QpoasesOptimizer& setGradientVector(const Eigen::VectorXd& g)
        {
            _g = g;
            _n = _g.size();
            return *this;
        }

        QpoasesOptimizer& setLinearConstraints(const Eigen::MatrixXd& A)
        {
            _A = A;
            _m = _A.rows();
            return *this;
        }

        QpoasesOptimizer& setLinearConstraintsBoundaries(const Eigen::VectorXd& lbA, const Eigen::VectorXd& ubA)
        {
            _lbA = lbA;
            _ubA = ubA;
            return *this;
        }

        QpoasesOptimizer& setVariablesBoundaries(const Eigen::VectorXd& lb, const Eigen::VectorXd& ub)
        {
            _lb = lb;
            _ub = ub;
            return *this;
        }

        QpoasesOptimizer& setRecalculation(const int& nwsr)
        {
            _nWSR = nwsr;
            return *this;
        }

        bool init()
        {
            // Setting QP problem
            _qp = std::make_unique<Solver>(_n, _m);
            Options options;
            _qp->setOptions(options);

            return _qp->init(_H.data(), _g.data(), _A.data(), _lb.data(), _ub.data(), _lbA.data(), _ubA.data(), _nWSR);
        }

        bool optimize() override
        {
            if constexpr (std::is_same_v<Solver, SQProblem>)
                return _qp->hotstart(_H.data(), _g.data(), _A.data(), _lb.data(), _ub.data(), _lbA.data(), _ubA.data(), _nWSR);
            else
                return _qp->hotstart(_g.data(), _lb.data(), _ub.data(), _lbA.data(), _ubA.data(), _nWSR);
        }

        Eigen::VectorXd solution() override
        {
            real_t xOpt[_n];
            _qp->getPrimalSolution(xOpt);
            _xSol = Eigen::Map<Eigen::VectorXd>(xOpt, _n);
            return AbstractOptimizer::solution();
        }

    protected:
        // QP problem
        std::unique_ptr<Solver> _qp;

        // Hessian and linear constraints matrices
        Eigen::MatrixXd _H, _A;

        // Gradient vector, linear constraints and variable upper/lower bounds
        Eigen::VectorXd _g, _lbA, _ubA, _lb, _ub;

        // Maximum number of working set recalculations
        int_t _nWSR;

        // Cover not needed functions
        using AbstractOptimizer::addConstraints;
        using AbstractOptimizer::addConstraintsHessians;
        using AbstractOptimizer::setConstraintsJacobian;
        using AbstractOptimizer::setObjective;
        using AbstractOptimizer::setObjectiveGradient;
        using AbstractOptimizer::setObjectiveHessian;
    };
} // namespace optimization_lib

#endif // OPTIMIZATIONLIB_QPOASESOPTIMIZER