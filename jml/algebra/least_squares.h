/* least_squares.h                                                 -*- C++ -*-
   Jeremy Barnes, 15 June 2003
   Copyright (c) 2003 Jeremy Barnes.  All rights reserved.
   $Source$

   Least squares solution.
*/

#ifndef __algebra__least_squares_h__
#define __algebra__least_squares_h__

#include "jml/stats/distribution.h"
#include "jml/stats/distribution_simd.h"
#include <boost/multi_array.hpp>
#include "jml/arch/exception.h"
#include "jml/algebra/matrix_ops.h"
#include "svd.h"
#include <boost/timer.hpp>
#include "jml/arch/timers.h"
#include "lapack.h"
#include <cmath>
#include "jml/utils/string_functions.h"
#include "jml/arch/simd_vector.h"
#include "jml/utils/worker_task.h"


namespace ML {


/*****************************************************************************/
/* LEAST_SQUARES                                                             */
/*****************************************************************************/

/** Solves an equality constrained least squares problem.  This calculates
    the vector x that minimises ||c - Ax||2 such that Bx = d holds.

    \param A                     a (m x n) matrix
    \param c                     a n element vector
    \param B                     a (p x n) matrix
    \param d                     a p element vector

    Note that this routine requires p <n n <= m + p.

    It uses the LAPACK routine xGGLSE to perform the dirty work.
*/
template<class Float>
distribution<Float>
least_squares(const boost::multi_array<Float, 2> & A,
              const distribution<Float> & c,
              const boost::multi_array<Float, 2> & B,
              const distribution<Float> & d)
{
    using namespace LAPack;

    size_t m = A.shape()[0];
    size_t n = A.shape()[1];
    size_t p = B.shape()[0];

    //cerr << "A: (mxn) " << A.shape()[0] << " x " << A.shape()[1] << endl;
    //cerr << "B: (pxn) " << B.shape()[0] << " x " << B.shape()[1] << endl;
    //cerr << "c: m     " << c.size() << endl;
    //cerr << "d: p     " << d.size() << endl;

    if (c.size() != m || B.shape()[1] != n || d.size() != p)
        throw Exception("least_squares: sizes didn't match");

    if (p > n || n > (m + p))
        throw Exception("least_squares: overconstrained system");

    // todo: check that B has full row rank p and that the matrix (A B)' has
    // full column rank n.

    distribution<Float> result(n);

    /* We need to transpose them for Fortran, but since they are destroyed
       anyway it's no big deal since they would have been copied. */
    boost::multi_array<Float, 2> AF = fortran(A);
    boost::multi_array<Float, 2> BF = fortran(B);
    distribution<Float> c2 = c;
    distribution<Float> d2 = d;

    int res = gglse(m, n, p,
                    AF.data_begin(), AF.shape()[0],
                    BF.data_begin(), BF.shape()[1],
                    &c2[0], &d2[0], &result[0]);

    if (res != 0)
        throw Exception(format("least_squares(): gglse returned error in arg "
                               "%d", res));

    return result;
}

//extern __thread std::ostream * debug_irls;

/** Solve a least squares linear problem.
    This solves the linear least squares problem
    \f[
        A\mathbf{x} = \mathbf{b}
    \f]
 
    for the parameter <bf>x</bf>.

    \param    A the coefficient matrix
    \param    b the required output vector
    \returns  x

    \pre      A.shape()[0] == b.size()

    This <em>should</em> work for any shape of A, but has only been verified
    for A square.  It uses a SVD internally to do its work; this is not the
    best way for a square system but is general enough to handle the other
    cases.
 */
template<class Float>
distribution<Float>
least_squares(const boost::multi_array<Float, 2> & A, const distribution<Float> & b)
{
    using namespace std;

    //boost::timer t;

    if (A.shape()[0] != b.size()) {
        cerr << "A.shape()[0] = " << A.shape()[0] << endl;
        cerr << "A.shape()[1] = " << A.shape()[1] << endl;
        cerr << "b.size() = " << b.size() << endl;
        throw Exception("incompatible dimensions for least_squares");
    }

    using namespace LAPack;
    
    int m = A.shape()[0];
    int n = A.shape()[1];

    distribution<Float> x = b;
    x.resize(std::max<size_t>(m, n));

    boost::multi_array<Float, 2> A2 = A;

#if 0
    using namespace std;
    cerr << "m = " << m << " n = " << n << " A2.shape()[0] = " << A2.shape()[0]
         << " A2.shape()[1] = " << A2.shape()[1] << endl;
    cerr << "A2 = " << endl << A2 << endl;
    cerr << "b = " << b << endl;
#endif
    int res = gels('T', n, m, 1, A2.data(), n, &x[0],
                   x.size());

    if (res < 0)
        throw Exception(format("least_squares(): gels returned error in arg "
                               "%d", -res));

    //if (debug_irls) {
    //    (*debug_irls) << "gels returned " << res << endl;
    //    (*debug_irls) << "x = " << x << endl;
    //}

    if (res > 0) {
        //if (debug_irls)
        //      (*debug_irls) << "retrying; " << res << " are too small" << endl;
    
        /* Rank-deficient matrix.  Use the more efficient routine. */
        int rank;
        Float rcond = -1.0;
        Float sv[std::min(m, n)];
        std::fill(sv, sv + std::min(m, n), 0.0);

        // Rebuild A2, transposed this time
        A2.resize(boost::extents[n][m]);
        A2 = transpose(A);

        // Rebuild x as it was previously overwritten
        x = b;
        x.resize(std::max<size_t>(m, n));

        res = gelsd(m, n, 1, A2.data(), m, &x[0], x.size(), sv, rcond, rank);

        //if (debug_irls)
        //    (*debug_irls) << "rcond: " << rcond << " rank: "
        //                  << rank << endl;
    }

    if (res < 0) {
        throw Exception(format("least_squares(): gelsy returned error in arg "
                               "%d", -res));
    }

    x.resize(n);
 
    //using namespace std;
    //cerr << "least_squares: took " << t.elapsed() << "s" << endl;
    
    return x;
    //cerr << "least squares: gels returned " << x2 << endl;
    //cerr << "least squares: A2 = " << endl << A2 << endl;

    //cerr << "least_squares: " << t.elapsed() << "s" << endl;
    //distribution<Float> x3
    //    = least_squares(A, b, boost::multi_array<Float, 2>(0, n), distribution<Float>());
    
    //cerr << "least squares: gglse returned " << x3 << endl;

}

/* Solve a rank deficient least squares problem. */
template<class Float>
distribution<Float>
least_squares_rd(const boost::multi_array<Float, 2> & A, const distribution<Float> & b)
{
    using namespace LAPack;
    
    int m = A.shape()[0];
    int n = A.shape()[1];

    distribution<Float> x = b;
    x.resize(std::max<size_t>(m, n) + 1);

    boost::multi_array<Float, 2> A2 = A;

    distribution<Float> sv(std::min(m, n));

    int rank;

    int res = gelsd(m, n, 1, A2.data(), A2.shape()[0], &x[0], x.size(),
                    &sv[0], -1.0, rank);

    if (res != 0)
        throw Exception(format("least_squares(): gglse returned error in arg "
                               "%d", res));
    
    x.resize(n);
    return x;
}

/* Returns U * diag(d) * V */
template<class Float>
boost::multi_array<Float, 2>
diag_mult(const boost::multi_array<Float, 2> & U,
          const distribution<Float> & d,
          const boost::multi_array<Float, 2> & V,
          bool parallel)
{
    size_t m = U.shape()[0], n = V.shape()[1], x = d.size();

    boost::multi_array<Float, 2> result(boost::extents[m][n]);

    if (U.shape()[1] != x || V.shape()[0] != x)
        throw Exception("diag_mult(): wrong shape");

    auto doColumn = [&] (int j)
        {
            Float Vj_values[x];
            for (unsigned k = 0;  k < x;  ++k)
                Vj_values[k] = V[k][j];
            for (unsigned i = 0;  i < m;  ++i) {
                result[i][j] = SIMD::vec_accum_prod3(&U[i][0], &d[0], Vj_values, x);
            }
        };

    if (parallel)
        run_in_parallel_blocked(0, n, doColumn);
    else for (unsigned j = 0;  j < n;  ++j) doColumn(j);
    
    return result;
}


/** Solve a least squares linear problem using ridge regression.

    This solves the linear least squares problem
    \f[
        A\mathbf{x} = \mathbf{b}
    \f]
 
    for the parameter <bf>x</bf>.

    \param    A the coefficient matrix
    \param    b the required output vector
    \returns  x

    \pre      A.shape()[0] == b.size()
 */
template<class Float>
distribution<Float>
ridge_regression(const boost::multi_array<Float, 2> & A,
                 const distribution<Float> & b,
                 float lambda)
{
    using namespace std;
    //cerr << "ridge_regression: A = " << A.shape()[0] << "x" << A.shape()[1]
    //     << " b = " << b.size() << endl;

    //cerr << "b = " << b << endl;
    //cerr << "A = " << A << endl;

    bool debug = false;
    //debug = true;

    Timer t(debug);

    auto doneStep = [&] (const std::string & where)
        {
            if (!debug)
                return;
            cerr << where << ": " << t.elapsed() << endl;
            t.restart();
        };

    // Step 1: SVD

    if (A.shape()[0] != b.size())
        throw Exception("incompatible dimensions for least_squares");

    using namespace LAPack;
    
    int m = A.shape()[0];
    int n = A.shape()[1];

    int minmn = std::min(m, n);

    // See http://www.clopinet.com/isabelle/Projects/ETH/KernelRidge.pdf

    // The matrix to decompose is square
    boost::multi_array<Float, 2> GK(boost::extents[minmn][minmn]);

    
    //cerr << "m = " << m << " n = " << n << endl;

    
    // Take either A * transpose(A) or (A transpose) * A, whichever is smaller
    if (m < n) {
        for (unsigned i1 = 0;  i1 < m;  ++i1)
            for (unsigned i2 = 0;  i2 < m;  ++i2)
                GK[i1][i2] = SIMD::vec_dotprod_dp(&A[i1][0], &A[i2][0], n);

        //for (unsigned i1 = 0;  i1 < m;  ++i1)
        //    for (unsigned i2 = 0;  i2 < m;  ++i2)
        //        for (unsigned j = 0;  j < n;  ++j)
        //            GK[i1][i2] += A[i1][j] * A[i2][j];
    } else {
        // TODO: vectorize and look at loop order
        for (unsigned i = 0;  i < m;  ++i)
            for (unsigned j1 = 0;  j1 < n;  ++j1)
                for (unsigned j2 = 0;  j2 < n;  ++j2)
                    GK[j1][j2] += A[i][j1] * A[i][j2];
    }

    doneStep("    square");

    if (debug)
        cerr << "GK = " << endl << GK << endl;

    //cerr << "GK.shape()[0] = " << GK.shape()[0] << endl;
    //cerr << "GK.shape()[1] = " << GK.shape()[1] << endl;

    // Add in the ridge
    for (unsigned i = 0;  i < minmn;  ++i)
        GK[i][i] += lambda;

    if (debug)
        cerr << "GK with ridge = " << endl << GK << endl;

    // Decompose to get the pseudoinverse
    distribution<Float> svalues(minmn);
    boost::multi_array<Float, 2> VT(boost::extents[minmn][minmn]);
    boost::multi_array<Float, 2> U(boost::extents[minmn][minmn]);
    
    // SVD
    int result = LAPack::gesdd("S", minmn, minmn,
                               GK.data(), minmn,
                               &svalues[0],
                               &VT[0][0], minmn,
                               &U[0][0], minmn);

    doneStep("    gesdd");

    if (result != 0)
        throw Exception("gesdd returned non-zero");

    distribution<Float> singular_values
        (svalues.begin(), svalues.begin() + minmn);

    if (debug)
        cerr << "singular values = " << singular_values << endl;

    if (debug) {
        // Multiply decomposition back to make sure that we get the original
        // matrix
        boost::multi_array<Float, 2> D = diag(singular_values);

        boost::multi_array<Float, 2> GK_test
            = U * D * VT;

        cerr << "GK_test = " << endl << GK_test << endl;
        //cerr << "errors = " << endl << (GK_test - GK) << endl;
    }

    // Figure out the optimal value of lambda based upon leave-one-out cross
    // validation

    double current_lambda = 10.0;

    struct Iteration {
        double lambda;
        double total_mse_unbiased;
        distribution<Float> x;
    };

    vector<Iteration> iterations;

    for (unsigned i = 0; current_lambda >= 1e-14;  ++i, current_lambda /= 10.0) {
        Iteration iter;
        iter.lambda = current_lambda;
        iterations.push_back(iter);
    };

    auto doIter = [&] (int i)
        {
            double current_lambda = iterations[i].lambda;
            
            Timer t(debug);
        
            auto doneStep = [&] (const std::string & where)
            {
                if (!debug)
                    return;
                cerr << "      " << where << ": " << t.elapsed() << endl;
                t.restart();
            };

            //cerr << "i = " << i << " current_lambda = " << current_lambda << endl;
            // Adjust the singular values for the new lambda
            distribution<Float> my_singular = singular_values;
            my_singular += (current_lambda - lambda);

            //boost::multi_array<Float, 2> GK_pinv
            //    = U * diag((Float)1.0 / my_singular) * VT;

            boost::multi_array<Float, 2> GK_pinv
            = diag_mult(U, (Float)1.0 / my_singular, VT, false /* parallel */);

            doneStep("diag_mult");

            // TODO: reduce GK by removing those basis vectors where the singular
            // values are too close to lambda
        
            if (debug && false) {
                cerr << "GK_pinv = " << endl << GK_pinv
                     << endl;
                cerr << "prod = " << endl << (GK * GK_pinv * GK) << endl;
                cerr << "prod2 = " << endl << (GK_pinv * GK * GK) << endl;
            }

            distribution<Float> & x = iterations[i].x;

            boost::multi_array<Float, 2> A_pinv
            = (m < n ? GK_pinv * A : A * GK_pinv);

            doneStep("A_pinv");

            if (debug && false)
                cerr << "A_pinv = " << endl << A_pinv << endl;

            x = b * A_pinv;
        
            if (debug)
                cerr << "x = " << x << endl;

            distribution<Float> predictions = A * x;

            //cerr << "A: " << A.shape()[0] << "x" << A.shape()[1] << endl;
            //cerr << "A_pinv: " << A_pinv.shape()[0] << "x" << A_pinv.shape()[1]
            //     << endl;

            //boost::multi_array<Float, 2> A_A_pinv
            //    = A * transpose(A_pinv);

            doneStep("predictions");

#if 0
            boost::multi_array<Float, 2> A_A_pinv
            = multiply_transposed(A, A_pinv);

            cerr << "A_A_pinv: " << A_A_pinv.shape()[0] << "x"
            << A_A_pinv.shape()[1] << " m = " << m << endl;

            if (debug && false)
                cerr << "A_A_pinv = " << endl << A_A_pinv << endl;
#else
            // We only need the diagonal of A * A_pinv

            distribution<Float> A_A_pinv_diag(m);
            for (unsigned j = 0;  j < m;  ++j)
                A_A_pinv_diag[j] = SIMD::vec_dotprod_dp(&A[j][0], &A_pinv[j][0], n);
#endif

            doneStep("A_A_pinv");

            // Now figure out the performance
            double total_mse_biased = 0.0, total_mse_unbiased = 0.0;
            for (unsigned j = 0;  j < m;  ++j) {

                if (j < 10 && false)
                    cerr << "j = " << j << " b[j] = " << b[j]
                         << " predictions[j] = " << predictions[j]
                         << endl;

                double resid = b[j] - predictions[j];

                // Adjust for the bias cause by training on this example.  This is
                // A * pinv(A), which is A * 

                double factor = 1.0 - A_A_pinv_diag[j];

                double resid_unbiased = resid / factor;

                total_mse_biased += (1.0 / m) * resid * resid;
                total_mse_unbiased += (1.0 / m) * resid_unbiased * resid_unbiased;
            }

            doneStep("mse");

            //cerr << "lambda " << current_lambda
            //     << " rmse_biased = " << sqrt(total_mse_biased)
            //     << " rmse_unbiased = " << sqrt(total_mse_unbiased)
            //     << endl;

            //if (sqrt(total_mse_biased) > 1.0) {
            //    cerr << "rmse_biased: x = " << x << endl;
            //}
        
#if 0
            cerr << "m = " << m << endl;
            cerr << "total_mse_biased   = " << total_mse_biased << endl;
            cerr << "total_mse_unbiased = " << total_mse_unbiased << endl;
            cerr << "best_error = " << best_error << endl;
            cerr << "x = " << x << endl;
#endif
            iterations[i].total_mse_unbiased = total_mse_unbiased;

        };

    run_in_parallel(0, iterations.size(), doIter);

    //double best_lambda = -1000;
    double best_error = 1000000;
    distribution<Float> x_best;


    for (unsigned i = 0;  i < iterations.size();  ++i) {

        if (iterations[i].total_mse_unbiased < best_error || i == 0) {
            x_best = iterations[i].x;
            //best_lambda = current_lambda;
            best_error = iterations[i].total_mse_unbiased;
        }
    }




    doneStep("    lambda");

    //cerr << "total: " << t.elapsed() << endl;

    return x_best;
}


/*****************************************************************************/
/* IRLS                                                                      */
/*****************************************************************************/

/** Multiply two matrices with a diagonal matrix.
    \f[
        X W X^T
    \f]

    \f[
        W = \left[
            \begin{array}{cccc}
              d_0  &    0   & \cdots &    0   \\
               0   &   d_1  & \cdots &    0   \\
            \vdots & \vdots & \ddots & \vdots \\
               0   &    0   &    0   &   d_n 
            \end{array}
            \right]
    \f]
 */

template<class Float>
boost::multi_array<Float, 2>
diag_mult(const boost::multi_array<Float, 2> & XT,
          const distribution<Float> & d)
{
    if (XT.shape()[1] != d.size())
        throw Exception("Incompatible matrix sizes");

    size_t nx = XT.shape()[1];
    size_t nv = XT.shape()[0];

    //cerr << "nx = " << nx << " nv = " << nv << endl;

    boost::multi_array<Float, 2> result(boost::extents[nv][nv]);

    if (false) {
        int chunk_size = 2048;  // ensure we fit in the cache
        distribution<Float> Xid(chunk_size);

        int x = 0;
        while (x < nx) {
            int nxc = std::min<size_t>(chunk_size, nx - x);

            for (unsigned i = 0;  i < nv;  ++i) {
                SIMD::vec_prod(&XT[i][x], &d[x], &Xid[0], nxc);
            
                for (unsigned j = 0;  j < nv;  ++j) {
                    result[i][j] += SIMD::vec_dotprod_dp(&XT[j][x], &Xid[0], nxc);
                    //result[i][j] += SIMD::vec_accum_prod3(&XT[i][x], &XT[j][x],
                    //                                      &d[x], nxc);
                    //for (unsigned x = 0;  x < nx;  ++x)
                    //result[i][j] += XT[i][x] * XT[j][x] * d[x];
                }
            }

            x += nxc;
        }
    } else {
        auto doRow = [&] (int i)
            {
                int chunk_size = 2048;  // ensure we fit in the cache

                int x = 0;
                while (x < nx) {
                    int nxc = std::min<size_t>(chunk_size, nx - x);
                    distribution<Float> Xid(chunk_size);
                    SIMD::vec_prod(&XT[i][x], &d[x], &Xid[0], nxc);
            
                    for (unsigned j = 0;  j < nv;  ++j) {
                        result[i][j] += SIMD::vec_dotprod_dp(&XT[j][x], &Xid[0], nxc);
                    }

                    x += nxc;
                }
            };
        
        run_in_parallel_blocked(0, nv, doRow);
    }

    return result;
}

/** Multiply a vector and matrix by a diagonal matrix.
    \f[
        X W \mathbf{y}
    \f]

    where

    \f[
        W = \left[
            \begin{array}{cccc}
              d_0  &    0   & \cdots &    0   \\
               0   &   d_1  & \cdots &    0   \\
            \vdots & \vdots & \ddots & \vdots \\
               0   &    0   &    0   &   d_n 
            \end{array}
            \right]
    \f]
*/
template<class Float>
distribution<Float>
diag_mult(const boost::multi_array<Float, 2> & X,
          const distribution<Float> & d,
          const distribution<Float> & y)
{
    size_t nx = X.shape()[1];
    if (nx != d.size() || nx != y.size())
        throw Exception("Incompatible matrix sizes");
    size_t nv = X.shape()[0];

    distribution<Float> result(nv, 0.0);
    for (unsigned v = 0;  v < nv;  ++v)
        for (unsigned x = 0;  x < nx;  ++x)
            result[v] += X[v][x] * d[x] * y[x];

    return result;
}

/** Iteratively reweighted least squares.  Allows a non-linear transformation
    (given by the link parameter) of a linear combination of features to be
    fitted in a least-squares fashion.  The dist parameter gives the
    distribution of the errors.
    
    \param y      the values to fit (target values)
    \param x      the matrix of values to fit with.  It should be nv x nx,
                  where nv is the number of variables to fit (and will be
                  the length of the output \b), and nx is the number of
                  examples (and is also the length of y).
    \param w      the relative weight (importance) of each example.  Is
                  normalized before use.  If unknown, pass a uniform
                  distribution.
    \param m      the number of observations for the binomial distribution.  If
                  the binomial distribution is not used, or the y values are
                  already proportions, then set all of the values to 1.
    \param link   the link function (see those above)
    \param dist   the error distribution function (see those above)

    \returns      the fitted parameters \p b, one for each column in x

    \pre          y.size() == w.size() == x.shape()[1]
    \post         b.size() == x.shape()[0]
*/

template<class Link, class Dist, class Float, class Regressor>
distribution<Float>
irls(const distribution<Float> & y, const boost::multi_array<Float, 2> & x,
     const distribution<Float> & w, const Link & link, const Dist & dist,
     const Regressor & regressor)
{
    using namespace std;

    bool debug = false;

    typedef distribution<Float> Vector;
    typedef boost::multi_array<Float, 2> Matrix;

    static const int max_iter = 20;           // from GLMlab
    static const float tolerence = 5e-5;      // from GLMlab
    
    size_t nv = x.shape()[0];                     // number of variables
    size_t nx = x.shape()[1];                     // number of examples

    if (y.size() != nx || w.size() != nx)
        throw Exception("incompatible data sizes");

    int iter = 0;
    Float rdev = std::sqrt((y * y).total());  // residual deviance
    Float rdev2 = 0;                          // last residual deviance
    Vector mu = (y + 0.5) / 2;                // link input
    Vector b(nv, 0.0);                        // ls fit parameters
    Vector b2;                                // last ls fit parameters
    Vector eta = link.forward(mu);            // link output
    Vector offset(nx, 0.0);                   // known values (??)
    distribution<Float> weights = w;          // sample weights

    for (unsigned i = 0;  i < mu.size();  ++i)
        if (!std::isfinite(mu[i]))
            throw Exception(format("mu[%d] = %f", i, mu[i]));

    Timer t(debug);
    
    auto doneStep = [&] (const std::string & step)
        {
            if (!debug)
                return;
            cerr << "  " << step << ": " << t.elapsed();
            t.restart();
        };

    /* Note: look in the irls.m function of GLMlab to see what we are trying
       to do here.  This is essentially a C++ reimplementation of that
       function.  I don't really know what it is doing. */
    while (abs(rdev - rdev2) > tolerence && iter < max_iter) {
        Timer t(debug);

        /* Find the new weights for this iteration. */
        Vector deta_dmu    = link.diff(mu);
        for (unsigned i = 0;  i < deta_dmu.size();  ++i)
            if (!std::isfinite(deta_dmu[i]))
                throw Exception(format("deta_dmu[%d] = %f", i, deta_dmu[i]));

        //if (debug_irls)
        //    (*debug_irls) << "deta_demu: " << deta_dmu << endl;

        doneStep("diff");

        Vector var         = dist.variance(mu);
        for (unsigned i = 0;  i < var.size();  ++i)
            if (!std::isfinite(var[i]))
                throw Exception(format("var[%d] = %f", i, var[i]));

        //if (debug_irls)
        //    (*debug_irls) << "var: " << deta_dmu << endl;

        doneStep("variance");

        Vector fit_weights = weights / (deta_dmu * deta_dmu * var);
        for (unsigned i = 0;  i < fit_weights.size();  ++i) {
            if (!std::isfinite(fit_weights[i])) {
                cerr << "weigths = " << weights[i]
                     << "  deta_dmu = " << deta_dmu[i]
                     << "  var = " << var[i] << endl;
                throw Exception(format("fit_weights[%d] = %f", i,
                                       fit_weights[i]));
            }
        }

        //if (debug_irls)
        //    (*debug_irls) << "fit_weights: " << fit_weights << endl;
        
        doneStep("fit_weights");
        
        //cerr << "fit_weights = " << fit_weights << endl;

        /* Set up the reweighted least squares problem. */
        Vector z           = eta - offset + (y - mu) * deta_dmu;
        Matrix xTwx        = diag_mult(x, fit_weights);
        doneStep("xTwx");
        Vector xTwz        = diag_mult(x, fit_weights, z);
        doneStep("xTwz");

        //if (debug_irls)
        //    (*debug_irls) << "z: " << z << endl
        //                  << "xTwx: " << xTwx << endl
        //                  << "xTwz: " << xTwz << endl;

        /* Solve the reweighted problem using a linear least squares. */
        b2                 = b;
        b                  = regressor.calc(xTwx, xTwz);

        //if (debug_irls)
        //    (*debug_irls) << "b: " << b << endl;

        doneStep("least squares");

        /* Re-estimate eta and mu based on refined estimate. */
        //cerr << "b.size() = " << b.size() << endl;
        //cerr << "x.shape()[0] = " << x.shape()[0]
        //     << " x.shape()[1] = " << x.shape()[1]
        //     << endl;

        eta                = (b * x) + offset;
        for (unsigned i = 0;  i < eta.size();  ++i)
            if (!std::isfinite(eta[i]))
                throw Exception(format("eta[%d] = %f", i, eta[i]));

        //if (debug_irls)
        //    (*debug_irls) << "eta: " << eta << endl;

        doneStep("eta");

        mu                 = link.inverse(eta);
        for (unsigned i = 0;  i < mu.size();  ++i)
            if (!std::isfinite(mu[i]))
                throw Exception(format("mu[%d] = %f", i, mu[i]));

        //if (debug_irls)
        //    (*debug_irls) << "me: " << mu << endl;

        doneStep("mu");

        /* Recalculate the residual deviance, and save the last one to check
           for convergence. */
        rdev2              = rdev;
        rdev               = dist.deviance(y, mu, weights);

        doneStep("deviance");

        ++iter;

        if (debug)
            cerr << "iter " << iter << ": " << t.elapsed() << endl;

        //if (debug_irls) {
        //    *debug_irls << "iter " << iter << " rdev " << rdev
        //                << " rdev2 " << rdev2 << " diff " << abs(rdev - rdev2)
        //                << " tolerance " << tolerence << endl;
        //}
    }

    return b;
}


} // namespace ML

#endif /* __algebra__least_squares_h__ */


