/* tsne.cc
   Jeremy Barnes, 15 January 2010
   Copyright (c) 2010 Jeremy Barnes.  All rights reserved.

   Implementation of the t-SNE algorithm.
*/

#include "tsne.h"
#include "stats/distribution.h"
#include "stats/distribution_ops.h"
#include "stats/distribution_simd.h"
#include "algebra/matrix_ops.h"
#include "arch/simd_vector.h"
#include <boost/tuple/tuple.hpp>
#include "algebra/lapack.h"
#include <cmath>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/variate_generator.hpp>
#include "boosting/worker_task.h"
#include <boost/timer.hpp>
#include "arch/timers.h"

using namespace std;

namespace ML {

/** Compute the perplexity and the P for a given value of beta.
    TODO: pre-compute the exp(D) values to remove exp from this loop*/
template<typename Float>
std::pair<double, distribution<Float> >
perplexity_and_prob(const distribution<Float> & D, double beta = 1.0,
                    int i = -1)
{
    distribution<double> P(D.size());
    SIMD::vec_exp(&D[0], -beta, &P[0], D.size());
    if (i != -1) P[i] = 0;
    double tot = P.total();

    if (!isfinite(tot) || tot == 0) {
        cerr << "beta = " << beta << endl;
        cerr << "D = " << D << endl;
        throw Exception("non-finite total for perplexity");
    }

    double H = log(tot) + beta * D.dotprod(P) / tot;
    P *= 1.0 / tot;

    if (!isfinite(P.total())) {
        cerr << "beta = " << beta << endl;
        cerr << "D = " << D << endl;
        cerr << "tot = " << tot << endl;
        throw Exception("non-finite total for perplexity");
    }


    return make_pair(H, P);
}

std::pair<double, distribution<float> >
perplexity_and_prob(const distribution<float> & D, double beta,
                    int i)
{
    return perplexity_and_prob<float>(D, beta, i);
}

std::pair<double, distribution<double> >
perplexity_and_prob(const distribution<double> & D, double beta,
                    int i)
{
    return perplexity_and_prob<double>(D, beta, i);
}

/** Given a matrix that gives the a number of points in a vector space of
    dimension d (ie, a number of points with coordinates of d dimensions),
    convert to a matrix that gives the square of the distance between
    each of the points.

    \params:
    X    a (n x d) matrix, where n is the number of points and d is the
         number of coordinates that each point has

    \returns:
    A (n x n) matrix giving the distance between each two points
*/
boost::multi_array<float, 2>
vectors_to_distances(boost::multi_array<float, 2> & X)
{
    // Note that (xi - yi)^2 = xi^2 - 2 xi yi + yi^2
    // sum_i (xi - yi)^2 = sum_i(xi^2 - 2xi yi + yi^2)
    //                   = sum_i(xi^2) + sum_i(yi^2) - 2sum_i(xiyi)
    // where i goes over the d dimensions

    int n = X.shape()[0];
    int d = X.shape()[1];

    distribution<float> sum_X(n);
    for (unsigned i = 0;  i < n;  ++i)
        sum_X[i] = SIMD::vec_dotprod_dp(&X[i][0], &X[i][0], d);
    
    // TODO: don't use this temporary; calculate as needed
    boost::multi_array<float, 2> XXT = multiply_transposed(X, X);

    boost::multi_array<float, 2> D(boost::extents[n][n]);
    for (unsigned i = 0;  i < n;  ++i) {
        for (unsigned j = i;  j < n;  ++j) {
            D[i][j] = D[j][i] = sum_X[i] + sum_X[j] - 2 * XXT[i][j];
        }
    }
            
    return D;
}

/** Given a matrix of distances, normalize them */

/** Calculate the beta for a single point.
    
    \param Di     The i-th row of the D matrix, for which we want to calculate
                  the probabilities.
    \param i      Which row number it is.

    \returns      The i-th row of the P matrix, which has the distances in D
                  converted to probabilities with the given perplexity.
 */
std::pair<distribution<float>, double>
binary_search_perplexity(const distribution<float> & Di,
                         double required_perplexity,
                         int i,
                         double tolerance = 1e-5)
{
    double betamin = -INFINITY, betamax = INFINITY;
    double beta = 1.0;

    distribution<float> P;
    double log_perplexity;
    double log_required_perplexity = log(required_perplexity);

    boost::tie(log_perplexity, P) = perplexity_and_prob(Di, beta, i);

    bool verbose = false;

    if (verbose)
        cerr << "iter currperp targperp     diff toleranc   betamin     beta  betamax" << endl;
    
    for (unsigned iter = 0;  iter != 50;  ++iter) {
        if (verbose) 
            cerr << format("%4d %8.4f %8.4f %8.4f %8.4f  %8.4f %8.4f %8.4f\n",
                           iter,
                           log_perplexity, log_required_perplexity,
                           fabs(log_perplexity - log_required_perplexity),
                           tolerance,
                           betamin, beta, betamax);
        
        if (fabs(log_perplexity - log_required_perplexity) < tolerance)
            break;

        if (log_perplexity > log_required_perplexity) {
            betamin = beta;
            if (!isfinite(betamax))
                beta *= 2;
            else beta = (beta + betamax) * 0.5;
        }
        else {
            betamax = beta;
            if (!isfinite(betamin))
                beta /= 2;
            else beta = (beta + betamin) * 0.5;
        }
        
        boost::tie(log_perplexity, P) = perplexity_and_prob(Di, beta, i);
    }

    return make_pair(P, beta);
}

/* Given a matrix of distances, convert to probabilities */
boost::multi_array<float, 2>
distances_to_probabilities(boost::multi_array<float, 2> & D,
                           double tolerance,
                           double perplexity)
{
    int n = D.shape()[0];
    if (D.shape()[1] != n)
        throw Exception("D is not square");

    boost::multi_array<float, 2> P(boost::extents[n][n]);

    distribution<float> beta(n, 1.0);

    for (unsigned i = 0;  i < n;  ++i) {
        //cerr << "i = " << i << endl;
        if (i % 250 == 0)
            cerr << "P-values for point " << i << " of " << n << endl;
        
        distribution<float> D_row(&D[i][0], &D[i][0] + n);
        distribution<float> P_row;
        boost::tie(P_row, beta[i])
            = binary_search_perplexity(D_row, perplexity, i, tolerance);

        if (P_row.size() != n)
            throw Exception("P_row has the wrong size");
        if (P_row[i] != 0.0) {
            cerr << "i = " << i << endl;
            //cerr << "D_row = " << D_row << endl;
            //cerr << "P_row = " << P_row << endl;
            cerr << "P_row.total() = " << P_row.total() << endl;
            cerr << "P_row[i] = " << P_row[i] << endl;
            throw Exception("P_row diagonal entry was not zero");
        }
        
        std::copy(P_row.begin(), P_row.end(), &P[i][0]);
    }

    cerr << "mean sigma is " << sqrt(1.0 / beta).mean() << endl;

    return P;
}

boost::multi_array<float, 2>
pca(boost::multi_array<float, 2> & coords, int num_dims)
{
    // TODO: normalize the input coordinates (especially if it seems to be
    // ill conditioned)

    int nx = coords.shape()[0];
    int nd = coords.shape()[1];

    int nvalues = std::min(nd, nx);

    int ndr = std::min(nvalues, num_dims);

    if (ndr < num_dims)
        throw Exception("svd_reduction: num_dims not low enough");
        
    distribution<float> svalues(nvalues);
    boost::multi_array<float, 2> lvectorsT(boost::extents[nvalues][nd]);
    boost::multi_array<float, 2> rvectors(boost::extents[nx][nvalues]);

    int res = LAPack::gesdd("S", nd, nx,
                            coords.data(), nd,
                            &svalues[0],
                            &lvectorsT[0][0], nd,
                            &rvectors[0][0], nvalues);
    
    // If some vectors are singular, ignore them
    // TODO: do...
        
    if (res != 0)
        throw Exception("gesdd returned non-zero");
        
    boost::multi_array<float, 2> result(boost::extents[nx][ndr]);
    for (unsigned i = 0;  i < nx;  ++i)
        std::copy(&rvectors[i][0], &rvectors[i][0] + ndr, &result[i][0]);

    return result;
}

inline int sign(float x)
{
    return -1 * (x < 0);
}

boost::multi_array<float, 2>
tsne(const boost::multi_array<float, 2> & probs,
     int num_dims)
{
    int n = probs.shape()[0];
    if (n != probs.shape()[1])
        throw Exception("probabilities were the wrong shape");

    int d = num_dims;

    boost::mt19937 rng;
    boost::normal_distribution<float> norm;

    boost::variate_generator<boost::mt19937,
                             boost::normal_distribution<float> >
        randn(rng, norm);
    
    boost::multi_array<float, 2> Y(boost::extents[n][d]);
    for (unsigned i = 0;  i < n;  ++i)
        for (unsigned j = 0;  j < d;  ++j)
            Y[i][j] = randn();

#if 1 // pseudo-random, for testing (matches the Python version)
    for (unsigned i = 0;  i < n;  ++i)
        for (unsigned j = 0;  j < d;  ++j)
            Y[i][j] = (((((i * 18446744073709551557ULL) + j) * 18446744073709551557ULL) % 4099) / 1050.0) - 2.0;
#endif // pseudo-random

    //for (unsigned i = 0;  i < 10;  ++i)
    //    cerr << "Y[" << i << "] = "
    //         << distribution<float>(&Y[i][0], &Y[i][0] + d)
    //         << endl;

    boost::multi_array<float, 2> dY(boost::extents[n][d]);
    boost::multi_array<float, 2> iY(boost::extents[n][d]);
    boost::multi_array<float, 2> gains(boost::extents[n][d]);
    std::fill(gains.data(), gains.data() + gains.num_elements(), 1.0f);

    // Symmetrize and probabilize P
    boost::multi_array<float, 2> P = probs + transpose(probs);

    // TODO: symmetric so only need to total the diagonal
    double sumP = 0.0;
    for (unsigned i = 0;  i < n;  ++i)
        sumP += SIMD::vec_sum_dp(&P[i][0], n);
    
    //cerr << "sumP = " << sumP << endl;

    // Factor that P should be multiplied by in all calculations
    // We boost it by 4 in early iterations to force the clusters to be
    // spread apart
    float pfactor = 4.0 / sumP;

    // TODO: do we need this?   P = Math.maximum(P, 1e-12);
    for (unsigned i = 0;  i < n;  ++i)
        for (unsigned j = 0;  j < n;  ++j)
            P[i][j] = std::max((i != j) * pfactor * P[i][j], 1e-12f);

    //for (unsigned i = 0;  i < 10;  ++i)
    //    cerr << "P[" << i << "] = "
    //         << distribution<float>(&P[i][0], &P[i][0] + n)
    //         << endl;


    int max_iter = 1000;
    float initial_momentum = 0.5;
    float final_momentum = 0.8;
    float eta = 500;
    float min_gain = 0.01;

    //cerr << "n = " << n << endl;

    for (int iter = 0;  iter < max_iter;  ++iter) {
        Timer timer;

        /*********************************************************************/
        // Pairwise affinities Qij
        // Implements formula 4 in (Van der Maaten and Hinton, 2008)
        // q_{ij} = d_{ij} / sum_{k,l, k != l} d_{kl}
        // where d_{ij} = 1 / (1 + ||y_i - y_j||^2)

        // again, ||y_i - y_j||^2 
        //     = sum_d ( y_id - y_jd )^2
        //     = sum_d ( y_id^2 + y_jd^2 - 2 y_id y_jd)
        //     = sum_d ( y_id^2) + sum_d(y_jd^2) - 2 sum_d(y_id y_jd)
        //     = ||y_i||^2 + ||y_j||^2 - 2 sum_d(y_id y_jd)

        // TODO: these will all be symmetric; we could save lots of work by
        // using diagonal matrices.

        // Y * Y^T: element (i, j) = sum_d(y_id y_jd)
        boost::multi_array<float, 2> YYT
            = multiply_transposed(Y, Y);

        // sum_Y: element i = ||y_i||^2
        distribution<float> sum_Y(n);
        for (unsigned i = 0;  i < n;  ++i) {
            // No vectorization as d is normally very small

            double total = 0.0;  // accum in double precision for accuracy
            for (unsigned j = 0;  j < d;  ++j)
                total += Y[i][j] * Y[i][j];
            sum_Y[i] = total;
        }
        
        //cerr << "sum_Y = " << sum_Y << endl;

        // D matrix: d_{ij} = 1 / (1 + ||y_i - y_j||^2)
        double d_total_offdiag = 0.0;
        boost::multi_array<float, 2> D(boost::extents[n][n]);
        for (unsigned i = 0;  i < n;  ++i) {
            for (unsigned j = 0;  j < n;  ++j) {
                D[i][j] = 1.0f / (1.0f + sum_Y[i] + sum_Y[j] -2.0f * YYT[i][j]);
                if (i == j) D[i][j] = 0.0;
                d_total_offdiag += D[i][j] * (i != j);
            }
        }

        //for (unsigned i = 0;  i < 3;  ++i)
        //    cerr << "D[" << i << "] = "
        //         << distribution<float>(&D[i][0], &D[i][0] + n)
        //         << endl;
        //cerr << endl;

        //cerr << "d_total_offdiag = " << d_total_offdiag << endl;

        // Q matrix: q_{i,j} = d_{ij} / sum_{k != l} d_{kl}
        boost::multi_array<float, 2> Q(boost::extents[n][n]);
        float qfactor = 1.0 / d_total_offdiag;
        for (unsigned i = 0;  i < n;  ++i)
            for (unsigned j = 0;  j < n;  ++j)
                Q[i][j] = std::max(1e-12f, D[i][j] * qfactor);

        //for (unsigned i = 0;  i < 3;  ++i)
        //    cerr << "Q[" << i << "] = "
        //         << distribution<float>(&Q[i][0], &Q[i][0] + n)
        //         << endl;
        //cerr << endl;

        
        
        /*********************************************************************/
        // Gradient
        // Implements formula 5 in (Van der Maaten and Hinton, 2008)
        // dC/dy_i = 4 * sum_j ( (p_ij - q_ij)(y_i - y_j)d_ij )

        boost::multi_array<float, 2> dY(boost::extents[n][d]);

        for (unsigned j = 0;  j < n;  ++j) {
            for (unsigned i = 0;  i < n;  ++i) {
                if (i == j) continue;
                float factor = 4.0f * (P[j][i] - Q[j][i]) * D[j][i];
                for (unsigned k = 0;  k < d;  ++k)
                    dY[i][k] += factor * (Y[i][k] - Y[j][k]);
            }
        }

        //for (unsigned i = 0;  i < 10;  ++i)
        //    cerr << "dY[" << i << "] = "
        //         << distribution<float>(&dY[i][0], &dY[i][0] + d)
        //         << endl;

        float momentum = (iter < 20 ? initial_momentum : final_momentum);

        // Implement scheme in Jacobs, 1988.  If we go in the same direction as
        // last time, we increase the learning speed of the parameter a bit.
        // If on the other hand the direction changes, we reduce exponentially
        // the rate.

        for (unsigned i = 0;  iter > 0 && i < n;  ++i) {
            // We use != here as we gradients in dY are the negatives of what
            // we want.
            for (unsigned j = 0;  j < d;  ++j) {
                //if (i < 10) 
                //    cerr << "i = " << i << " j = " << j << " dY = " << dY[i][j]
                //         << " iY " << iY[i][j] << " sign(dY) "
                //         << sign(dY[i][j]) << " sign(iY)" << sign(iY[i][j])
                //         << endl;
                if (dY[i][j] * iY[i][j] < 0.0f)
                    gains[i][j] = gains[i][j] + 0.2f;
                else gains[i][j] = gains[i][j] * 0.8f;
                gains[i][j] = std::max(min_gain, gains[i][j]);
            }
        }

        //for (unsigned i = 0;  i < 10;  ++i)
        //    cerr << "gains[" << i << "] = "
        //         << distribution<float>(&gains[i][0], &gains[i][0] + d)
        //         << endl;
        
        for (unsigned i = 0;  i < n;  ++i)
            for (unsigned j = 0;  j < d;  ++j)
                iY[i][j] = momentum * iY[i][j] - (eta * gains[i][j] * dY[i][j]);

        //for (unsigned i = 0;  i < 10;  ++i)
        //    cerr << "iY[" << i << "] = "
        //         << distribution<float>(&iY[i][0], &iY[i][0] + d)
        //         << endl;

        Y = Y + iY;

        //for (unsigned i = 0;  i < 10;  ++i)
        //    cerr << "Y[" << i << "] = "
        //         << distribution<float>(&Y[i][0], &Y[i][0] + d)
        //         << endl;


        // Recenter Y values about the origin
        double Y_means[d];
        std::fill(Y_means, Y_means + d, 0.0);
        for (unsigned i = 0;  i < n;  ++i)
            for (unsigned j = 0;  j < d;  ++j)
                Y_means[j] += Y[i][j];


        float n_recip = 1.0f / n;

        for (unsigned i = 0;  i < n;  ++i)
            for (unsigned j = 0;  j < d;  ++j)
                Y[i][j] -= Y_means[j] * n_recip;

        //for (unsigned i = 0;  i < 10;  ++i)
        //    cerr << "centered Y[" << i << "] = "
        //         << distribution<float>(&Y[i][0], &Y[i][0] + d)
        //         << endl;

        double cost = 0.0;
        for (unsigned i = 0;  i < n;  ++i) {
            for (unsigned j = 0;  j < n;  ++j) {
                double mycost = P[i][j] * logf(P[i][j] / Q[i][j]);
                cost += mycost;
                //if (i < 10)
                //    cerr << "i = " << i << " j = " << j << " mycost = "
                //         << mycost << endl;
            }
        }
        
        cerr << "iteration " << iter << " cost = " << cost << " elapsed "
             << timer.elapsed() << endl;

        // Stop lying about P values if we're finished
        if (iter == 100) {
            for (unsigned i = 0;  i < n;  ++i)
                for (unsigned j = 0;  j < n;  ++j)
                    P[i][j] *= 0.25f;
        }
    }

    return Y;
}

} // namespace ML
