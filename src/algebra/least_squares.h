/* least_squares.h                                                 -*- C++ -*-
   Jeremy Barnes, 15 June 2003
   Copyright (c) 2003 Jeremy Barnes.  All rights reserved.
   $Source$

   Least squares solution.
*/

#ifndef __algebra__least_squares_h__
#define __algebra__least_squares_h__

#include "stats/distribution.h"
#include "stats/distribution_simd.h"
#include <boost/multi_array.hpp>
#include "arch/exception.h"
#include "algebra/matrix_ops.h"
#include "svd.h"
#include <boost/timer.hpp>
#include "lapack.h"
#include <cmath>
#include "utils/string_functions.h"
#include "arch/simd_vector.h"
#include "arch/threads.h"


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

    if (A.shape()[0] != b.size())
        throw Exception("incompatible dimensions for least_squares");

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
    // Step 1: SVD

    using namespace std;
    using namespace Stats;

    //boost::timer t;

    if (A.shape()[0] != b.size())
        throw Exception("incompatible dimensions for least_squares");

    using namespace LAPack;
    
    int m = A.shape()[0];
    int n = A.shape()[1];

    int minmn = std::min(m, n);

    // See http://www.clopinet.com/isabelle/Projects/ETH/KernelRidge.pdf

    // The matrix to decompose is square
    boost::multi_array<Float, 2> GK(boost::extents[minmn][minmn]);

    
    // Take either A * transpose(A) or (A transpose) * A, whichever is smaller
    if (m < n) {
        for (unsigned i1 = 0;  i1 < m;  ++i1)
            for (unsigned i2 = 0;  i2 < m;  ++i2)
                for (unsigned j = 0;  j < n;  ++j)
                    GK[i1][i2] += A[i1][j] * A[i2][j];
    } else {
        for (unsigned i = 0;  i < m;  ++i)
            for (unsigned j1 = 0;  j1 < n;  ++j1)
                for (unsigned j2 = 0;  j2 < n;  ++j2)
                    GK[j1][j2] += A[i][j1] * A[i][j2];
    }

    cerr << "GK = " << endl << GK << endl;

    // Add in the ridge
    for (unsigned i = 0;  i < minmn;  ++i)
        GK[i][i] += lambda;

    cerr << "GK with ridge = " << endl << GK << endl;

    // Decompose to get the pseudoinverse
    distribution<Float> svalues(minmn);
    boost::multi_array<Float, 2> VT(boost::extents[minmn][minmn]);
    boost::multi_array<Float, 2> U(boost::extents[minmn][minmn]);
    
    int result = LAPack::gesdd("S", minmn, minmn,
                               GK.data(), minmn,
                               &svalues[0],
                               &VT[0][0], minmn,
                               &U[0][0], minmn);

    if (result != 0)
        throw Exception("gesdd returned non-zero");

    // Transpose lvectors
    //boost::multi_array<float, 2> lvectors(boost::extents[minmn][minmn]);
    //for (unsigned i = 0;  i < minmn;  ++i)
    //    for (unsigned j = 0;  j < minmn;  ++j)
    //        lvectors[i][j] = lvectorsT[j][i];

    distribution<Float> singular_values
        (svalues.begin(), svalues.begin() + minmn);

    cerr << "singular values = " << singular_values << endl;

    bool debug = true;

    if (debug) {
        // Multiply decomposition back to make sure that we get the original
        // matrix
        boost::multi_array<Float, 2> D = diag(singular_values);

        boost::multi_array<Float, 2> GK_test
            = U * D * VT;

        cerr << "GK_test = " << endl << GK_test << endl;
        //cerr << "errors = " << endl << (GK_test - GK) << endl;
    }

    boost::multi_array<Float, 2> GK_pinv
        = transpose(U * diag(1.0 / singular_values) * VT);

    if (debug) {
        cerr << "GK_pinv = " << endl << GK_pinv
             << endl;
        cerr << "prod = " << endl << (GK * GK_pinv * GK) << endl;
        cerr << "prod2 = " << endl << (GK_pinv * GK * GK) << endl;
    }

    distribution<Float> x;
    if (m < n)
        x = GK_pinv * A * b;
    else x = A * GK_pinv * b;

    cerr << "x = " << x << endl;

    return x;



#if 0
    singular_models.resize(models.size());
    
    for (unsigned i = 0;  i < models.size();  ++i)
        singular_models[i]
            = distribution<float>(&lvectors[i][0],
                                  &lvectors[i][nwanted - 1] + 1);

    //cerr << "singular_models[0] = " << singular_models[0] << endl;
    //cerr << "singular_models[1] = " << singular_models[1] << endl;

    singular_targets.resize(targets.size());

    for (unsigned i = 0;  i < targets.size();  ++i)
        singular_targets[i]
            = distribution<float>(&rvectors[i][0],
                                  &rvectors[i][nwanted - 1] + 1);
#endif

    



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
    static Lock lock;
    Guard guard(lock);

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

template<class Link, class Dist, class Float>
distribution<Float>
irls(const distribution<Float> & y, const boost::multi_array<Float, 2> & x,
     const distribution<Float> & w, const Link & link, const Dist & dist)
{
    using namespace std;

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

    //boost::timer t;
    
    /* Note: look in the irls.m function of GLMlab to see what we are trying
       to do here.  This is essentially a C++ reimplementation of that
       function.  I don't really know what it is doing. */
    while (abs(rdev - rdev2) > tolerence && iter < max_iter) {
        //cerr << "iter " << iter << ": " << t.elapsed() << endl;

        /* Find the new weights for this iteration. */
        Vector deta_dmu    = link.diff(mu);
        for (unsigned i = 0;  i < deta_dmu.size();  ++i)
            if (!std::isfinite(deta_dmu[i]))
                throw Exception(format("deta_dmu[%d] = %f", i, deta_dmu[i]));

        //if (debug_irls)
        //    (*debug_irls) << "deta_demu: " << deta_dmu << endl;

        //cerr << "diff: " << t.elapsed() << endl;

        Vector var         = dist.variance(mu);
        for (unsigned i = 0;  i < var.size();  ++i)
            if (!std::isfinite(var[i]))
                throw Exception(format("var[%d] = %f", i, var[i]));

        //if (debug_irls)
        //    (*debug_irls) << "var: " << deta_dmu << endl;

        //cerr << "variance: " << t.elapsed() << endl;

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

        //cerr << "fit_weights: " << t.elapsed() << endl;

        /* Set up the reweighted least squares problem. */
        Vector z           = eta - offset + (y - mu) * deta_dmu;
        //cerr << "z: " << t.elapsed() << endl;
        Matrix xTwx        = diag_mult(x, fit_weights);
        //cerr << "xTwx: " << t.elapsed() << endl;
        Vector xTwz        = diag_mult(x, fit_weights, z);
        //cerr << "xTwz: " << t.elapsed() << endl;

        //if (debug_irls)
        //    (*debug_irls) << "z: " << z << endl
        //                  << "xTwx: " << xTwx << endl
        //                  << "xTwz: " << xTwz << endl;

        /* Solve the reweighted problem using a linear least squares. */
        b2                 = b;
        b                  = least_squares_rd(xTwx, xTwz);

        //if (debug_irls)
        //    (*debug_irls) << "b: " << b << endl;

        //cerr << "least squares: " << t.elapsed() << endl;

        /* Re-estimate eta and mu based on refined estimate. */
        eta                = (x * b) + offset;
        for (unsigned i = 0;  i < eta.size();  ++i)
            if (!std::isfinite(eta[i]))
                throw Exception(format("eta[%d] = %f", i, eta[i]));

        //if (debug_irls)
        //    (*debug_irls) << "eta: " << eta << endl;

        //cerr << "eta: " << t.elapsed() << endl;

        mu                 = link.inverse(eta);
        for (unsigned i = 0;  i < mu.size();  ++i)
            if (!std::isfinite(mu[i]))
                throw Exception(format("mu[%d] = %f", i, mu[i]));

        //if (debug_irls)
        //    (*debug_irls) << "me: " << mu << endl;

        //cerr << "mu: " << t.elapsed() << endl;

        /* Recalculate the residual deviance, and save the last one to check
           for convergence. */
        rdev2              = rdev;
        rdev               = dist.deviance(y, mu, weights);

        //cerr << "deviance: " << t.elapsed() << endl;

        ++iter;

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


