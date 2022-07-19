#ifndef OPTIMIZATIONLIB_IPOPTOPTIMIZER
#define OPTIMIZATIONLIB_IPOPTOPTIMIZER

#include "optimization_lib/AbstractOptimizer.hpp"

#define HAVE_CSTDDEF
#include <IpTNLP.hpp>
#undef HAVE_CSTDDEF

#include <IpIpoptApplication.hpp>
#include <IpSolveStatistics.hpp>

#include <cassert>

#ifdef __GNUC__
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif

using namespace Ipopt;

namespace optimization_lib {
    class IpoptOptimizer : public AbstractOptimizer, public TNLP {
    public:
        IpoptOptimizer() : _app(IpoptApplicationFactory())
        {
            // Default options
            _app->Options()->SetNumericValue("tol", 1e-6);
            // _app->Options()->SetNumericValue("max_iter", 500);
            // _app->Options()->SetNumericValue("print_level", 5); // 0 <= print_level <= 2
            _app->Options()->SetStringValue("mu_strategy", "adaptive");
            // _app->Options()->SetStringValue("linear_solver", "ma57");
            // _app->Options()->SetStringValue("output_file", "ipopt.out");

            // The following overwrites the default name (ipopt.opt) of the options file
            // _app->Options()->SetStringValue("option_file_name", "hs071.opt");
        }

        ~IpoptOptimizer()
        {
        }

        bool optimize() override
        {
            // Set options depending on the available functions
            if (!_hess)
                _app->Options()->SetStringValue("hessian_approximation", "limited-memory");

            ApplicationReturnStatus status;
            status = _app->Initialize();

            if (status != Solve_Succeeded) {
                std::cout << std::endl
                          << std::endl
                          << "*** Error during initialization!" << std::endl;
                return (bool)status;
            }

            status = _app->OptimizeTNLP(this);

            if (status != Solve_Succeeded) {
                std::cout << std::endl
                          << std::endl
                          << "*** The problem FAILED!" << std::endl;
                return (int)status ? false : true;
            }

            return (int)status ? false : true;
        }

        /** Method to return some info about the nlp */
        bool get_nlp_info(Index& n, Index& m, Index& nnz_jac_g, Index& nnz_h_lag, IndexStyleEnum& index_style) override
        {
            // The problem described in MyNLP.hpp has 2 variables, x1, & x2,
            n = _n;

            // one equality constraint,
            m = _m;

            // nonzeros in the jacobian of the constraints
            if (_jac)
                nnz_jac_g = _jac(Eigen::VectorXd::Random(n)).nonZeros(); // the vector should be generated within the constraints
            else
                nnz_jac_g = 0;

            // nonzeros in the hessian of the lagrangian
            if (_hess) {
                Eigen::SparseMatrix<double> hess = _hess(Eigen::VectorXd::Random(n));
                for (auto& hessConstraint : _hessConstraints)
                    hess += hessConstraint(Eigen::VectorXd::Random(n));

                nnz_h_lag = hess.nonZeros();
            }
            else
                nnz_h_lag = 0;

            // standard fortran index style for row/col entries
            index_style = TNLP::C_STYLE;

            return true;
        }

        /** Method to return the bounds for my problem */
        bool get_bounds_info(Index n, Number* x_l, Number* x_u, Index m, Number* g_l, Number* g_u) override
        {
            assert(n == _n);
            assert(m == _m);

            // Preset variables bounds
            for (size_t i = 0; i < n; i++) {
                x_l[i] = -2e19;
                x_u[i] = +2e19;
            }

            // Set variables bounds
            for (size_t i = 0; i < _xC.size(); i++) {
                x_l[std::get<0>(_xC[i])] = std::get<1>(_xC[i]);
                x_u[std::get<0>(_xC[i])] = std::get<2>(_xC[i]);
            }

            // Set constraints bounds
            for (size_t i = 0; i < _gC.size(); i++) {
                g_l[i] = std::get<0>(_gC[i]);
                g_u[i] = std::get<1>(_gC[i]);
            }

            return true;
        }

        /** Method to return the starting point for the algorithm */
        bool get_starting_point(Index n, bool init_x, Number* x, bool init_z, Number* z_L, Number* z_U, Index m, bool init_lambda, Number* lambda) override
        {
            // Here, we assume we only have starting values for x, if you code
            // your own NLP, you can provide starting values for the others if
            // you wish.
            assert(init_x == true);
            assert(init_z == false);
            assert(init_lambda == false);

            for (size_t i = 0; i < n; i++)
                x[i] = _x0(i);

            return true;
        }

        /** Method to return the objective value */
        bool eval_f(Index n, const Number* x, bool new_x, Number& obj_value) override
        {
            assert(n == _n);

            // Fill Eigen vector
            Eigen::VectorXd xE(n);
            for (size_t i = 0; i < n; i++)
                xE(i) = x[i];

            // Evaluate objective function
            obj_value = _f(xE);

            return true;
        }

        /** Method to return the gradient of the objective */
        bool eval_grad_f(Index n, const Number* x, bool new_x, Number* grad_f) override
        {
            if (_grad) {
                assert(n == _n);

                // // Fill Eigen vector
                // Eigen::VectorXd xE(_n);
                // for (size_t i = 0; i < _n; i++)
                //     xE(i) = x[i];

                // Compute gradient
                Eigen::VectorXd grad = _grad(Eigen::Map<Eigen::VectorXd>((double*)x, n));

                // return the gradient of the objective function grad_{x} f(x)
                for (size_t i = 0; i < n; i++)
                    grad_f[i] = grad(i);

                return true;
            }
            else
                return false;
        }

        /** Method to return the constraint residuals */
        bool eval_g(Index n, const Number* x, bool new_x, Index m, Number* g) override
        {
            if (m) {
                assert(n == _n);
                assert(m == _m);

                // Fill Eigen vector
                Eigen::VectorXd xE(n);
                for (size_t i = 0; i < n; i++)
                    xE(i) = x[i];

                for (size_t i = 0; i < m; i++)
                    g[i] = _g[i](xE);

                return true;
            }
            else
                return false;
        }

        /** Method to return:
         *   1) The structure of the Jacobian (if "values" is NULL)
         *   2) The values of the Jacobian (if "values" is not NULL)
         */
        bool eval_jac_g(Index n, const Number* x, bool new_x, Index m, Index nele_jac, Index* iRow, Index* jCol, Number* values) override
        {
            if (nele_jac) {
                assert(n == _n);
                assert(m == _m);

                if (values == NULL) {
                    Eigen::SparseMatrix<double> jac = _jac(Eigen::VectorXd::Random(n));

                    // int *rowsindices = jac.innerIndexPtr(),
                    //     *colsindices = colMajorIndices(jac);

                    // for (size_t i = 0; i < nele_jac; i++) {
                    //     iRow[i] = rowsindices[i];
                    //     jCol[i] = colsindices[i];
                    // }

                    // Here is not very clear yet. innerIndexPtr() pointer returned by the Eigen function is pointing towards
                    // memory in the (static) stack. iRow is likely a pointer towards memory in the heap; thus I copy the memory
                    // pointed by innerIndexPtr() into the memory pointed by iRow. Is there a way to directly allocate innerIndexPtr()
                    // in the heap and then move the memory address into iRow?
                    memcpy(iRow, jac.innerIndexPtr(), nele_jac * sizeof(int)); // This should be equivalent to a for loop assignment

                    // colMajorIndices(jac) returns a pointer pointing towards heap memory. jCol = colMajorIndices(jac) only should
                    // although causing memory leak but it doesn't. So first I release the memory owned by jCol and then I assign
                    // to jCol the address returned by colMajorIndices(jac).

                    // delete[] jCol; // using delete[] because jCol is supposed to be an array pointer
                    // jCol = colMajorIndices(jac);

                    int* colsindices = colMajorIndices(jac);
                    memcpy(jCol, colsindices, nele_jac * sizeof(int));
                    delete[] colsindices;
                }
                else {
                    Eigen::SparseMatrix<double> jac = _jac(Eigen::Map<Eigen::VectorXd>((double*)x, n));

                    // The same seen above for iRow
                    memcpy(values, jac.valuePtr(), nele_jac * sizeof(double));

                    // double* val = jac.valuePtr();

                    // for (size_t i = 0; i < nele_jac; i++)
                    //     values[i] = val[i];
                }

                return true;
            }
            else
                return false;
        }

        /** Method to return:
         *   1) The structure of the Hessian of the Lagrangian (if "values" is NULL)
         *   2) The values of the Hessian of the Lagrangian (if "values" is not NULL)
         */
        bool eval_h(Index n, const Number* x, bool new_x, Number obj_factor, Index m, const Number* lambda, bool new_lambda, Index nele_hess, Index* iRow, Index* jCol, Number* values) override
        {
            if (nele_hess) {
                assert(n == _n);
                assert(m == _m);

                if (values == NULL) {
                    Eigen::SparseMatrix<double> hess = _hess(Eigen::VectorXd::Random(n));

                    for (auto& hessConstraint : _hessConstraints)
                        hess += hessConstraint(Eigen::VectorXd::Random(n));

                    memcpy(iRow, hess.innerIndexPtr(), nele_hess * sizeof(int));

                    // delete[] jCol;
                    // jCol = colMajorIndices(hess);

                    int* colsindices = colMajorIndices(hess);
                    memcpy(jCol, colsindices, nele_hess * sizeof(int));
                    delete[] colsindices;

                    // int *rowsindices = hess.innerIndexPtr(),
                    //     *colsindices = colMajorIndices(hess);

                    // for (size_t i = 0; i < nele_hess; i++) {
                    //     // iRow[i] = rowsindices[i];
                    //     jCol[i] = colsindices[i];
                    // }

                    // delete[] colsindices;
                }
                else {

                    Eigen::SparseMatrix<double> hess = obj_factor * _hess(Eigen::Map<Eigen::VectorXd>((double*)x, n));

                    for (size_t i = 0; i < m; i++)
                        hess += lambda[i] * _hessConstraints[i](Eigen::Map<Eigen::VectorXd>((double*)x, n));

                    memcpy(values, hess.valuePtr(), nele_hess * sizeof(double));

                    // double* val = hess.valuePtr();

                    // for (size_t i = 0; i < nele_hess; i++)
                    //     values[i] = val[i];
                }

                return true;
            }
            else
                return false;
        }

        /** This method is called when the algorithm is complete so the TNLP can store/write the solution */
        void finalize_solution(SolverReturn status, Index n, const Number* x, const Number* z_L, const Number* z_U, Index m, const Number* g, const Number* lambda, Number obj_value, const IpoptData* ip_data, IpoptCalculatedQuantities* ip_cq) override
        {
            // here is where we would store the solution to variables, or write to a file, etc
            // so we could use the solution.
            _xSol = Eigen::Map<Eigen::VectorXd>((double*)x, n);

            // For this example, we write the solution to the console
            std::cout << std::endl
                      << std::endl
                      << "Solution of the primal variables, x" << std::endl;
            for (Index i = 0; i < n; i++) {
                std::cout << "x[" << i << "] = " << x[i] << std::endl;
            }

            std::cout << std::endl
                      << std::endl
                      << "Solution of the bound multipliers, z_L and z_U" << std::endl;
            for (Index i = 0; i < n; i++) {
                std::cout << "z_L[" << i << "] = " << z_L[i] << std::endl;
            }
            for (Index i = 0; i < n; i++) {
                std::cout << "z_U[" << i << "] = " << z_U[i] << std::endl;
            }

            std::cout << std::endl
                      << std::endl
                      << "Objective value" << std::endl;
            std::cout << "f(x*) = " << obj_value << std::endl;

            std::cout << std::endl
                      << "Final value of the constraints:" << std::endl;
            for (Index i = 0; i < m; i++) {
                std::cout << "g(" << i << ") = " << g[i] << std::endl;
            }
        }

    private:
        SmartPtr<IpoptApplication> _app;
        /**@name Methods to block default compiler methods.
         *
         * The compiler automatically generates the following three methods.
         *  Since the default compiler implementation is generally not what
         *  you want (for all but the most simple classes), we usually
         *  put the declarations of these methods in the private section
         *  and never implement them. This prevents the compiler from
         *  implementing an incorrect "default" behavior without us
         *  knowing. (See Scott Meyers book, "Effective C++")
         */
        //@{
        IpoptOptimizer(
            const IpoptOptimizer&);

        IpoptOptimizer& operator=(
            const IpoptOptimizer&);
        //@}
    };
} // namespace optimization_lib

#endif // OPTIMIZATIONLIB_IPOPTOPTIMIZER