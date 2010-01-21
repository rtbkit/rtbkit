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

template<typename Float>
double
vectors_to_distances(const boost::multi_array<Float, 2> & X,
                     boost::multi_array<Float, 2> & D,
                     bool fill_upper = true)
{
    // again, ||y_i - y_j||^2 
    //     = sum_d ( y_id - y_jd )^2
    //     = sum_d ( y_id^2 + y_jd^2 - 2 y_id y_jd)
    //     = sum_d ( y_id^2) + sum_d(y_jd^2) - 2 sum_d(y_id y_jd)
    //     = ||y_i||^2 + ||y_j||^2 - 2 sum_d(y_id y_jd)
    
    int n = X.shape()[0];

    if (D.shape()[0] != n || D.shape()[1] != n)
        throw Exception("D matrix should be square with (n x n) shape");
    
    int d = X.shape()[1];

    distribution<Float> sum_X(n);
    for (unsigned i = 0;  i < n;  ++i)
        sum_X[i] = SIMD::vec_dotprod_dp(&X[i][0], &X[i][0], d);
    
    // TODO: don't use this temporary; calculate as needed
    boost::multi_array<Float, 2> XXT
        = multiply_transposed<Float>(X);

    double total = 0.0;

    for (unsigned i = 0;  i < n;  ++i) {
        D[i][i] = 0.0f;
        for (unsigned j = 0;  j < i;  ++j) {
            Float val = sum_X[i] + sum_X[j] - 2.0f * XXT[i][j];
            D[i][j] = D[j][i] = val;
            total += val;
        }
    }
            
    return total;
}

double
vectors_to_distances(const boost::multi_array<float, 2> & X,
                     boost::multi_array<float, 2> & D,
                     bool fill_upper)
{
    return vectors_to_distances<float>(X, D, fill_upper);
}

double
vectors_to_distances(const boost::multi_array<double, 2> & X,
                     boost::multi_array<double, 2> & D,
                     bool fill_upper)
{
    return vectors_to_distances<double>(X, D, fill_upper);
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

double
kl(const float * p, const float * q, size_t n)
{
    double result = 0.0;
    for (unsigned i = 0;  i < n;  ++i)
        result += p[i] * logf(p[i] / q[i]);
    return result;
}

double
calc_D_row(float * Di, float sum_Yi, const float * sum_Y, const float * YYTi,
           int i)
{
    double result = 0.0;
    for (unsigned j = 0;  j < i;  ++j) {
        Di[j] = 1.0f / (1.0f + sum_Y[i] + sum_Y[j] -2.0f * YYTi[j]);
        result += Di[j];
    }

    return result;
}

namespace {

double t_YYT = 0.0, t_D = 0.0, t_Q = 0.0, t_dY = 0.0, t_update = 0.0;
double t_sumY = 0.0, t_recenter = 0.0, t_cost = 0.0;
struct AtEnd {
    ~AtEnd()
    {
        cerr << "tsne core profile:" << endl;
        cerr << "  YYT:        " << t_YYT << endl;
        cerr << "  sumY:       " << t_sumY << endl;
        cerr << "  D:          " << t_D << endl;
        cerr << "  Q:          " << t_Q << endl;
        cerr << "  dY:         " << t_dY << endl;
        cerr << "  update:     " << t_update << endl;
        cerr << "  recenter:   " << t_recenter << endl;
        cerr << "  cost:       " << t_cost << endl;
    }
} atend;

} // file scope

boost::multi_array<float, 2>
tsne(const boost::multi_array<float, 2> & probs,
     int num_dims,
     const TSNE_Params & params)
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
            Y[i][j] = 0.0001 * randn();

#if 0 // pseudo-random, for testing (matches the Python version)
    for (unsigned i = 0;  i < n;  ++i)
        for (unsigned j = 0;  j < d;  ++j)
            Y[i][j] = (((((i * 18446744073709551557ULL) + j) * 18446744073709551557ULL) % 4099) / 1050.0) - 2.0;
#endif // pseudo-random

    //for (unsigned i = 0;  i < 10;  ++i)
    //    cerr << "Y[" << i << "] = "
    //         << distribution<float>(&Y[i][0], &Y[i][0] + d)
    //         << endl;

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


    //cerr << "n = " << n << endl;

    Timer timer;

    for (int iter = 0;  iter < params.max_iter;  ++iter) {

        boost::timer t;

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
        // using upper/lower diagonal matrices.

        // Y * Y^T: element (i, j) = sum_d(y_id y_jd)
        boost::multi_array<float, 2> YYT = multiply_transposed(Y, Y);

        t_YYT += t.elapsed();  t.restart();

        // sum_Y: element i = ||y_i||^2
        distribution<float> sum_Y(n);
        for (unsigned i = 0;  i < n;  ++i) {
            // No vectorization as d is normally very small

            double total = 0.0;  // accum in double precision for accuracy
            for (unsigned j = 0;  j < d;  ++j)
                total += Y[i][j] * Y[i][j];
            sum_Y[i] = total;
        }

        t_sumY += t.elapsed();  t.restart();
        
        //cerr << "sum_Y = " << sum_Y << endl;

        // D matrix: d_{ij} = 1 / (1 + ||y_i - y_j||^2)
        double d_total_offdiag = 0.0;
        boost::multi_array<float, 2> D(boost::extents[n][n]);

        for (unsigned i = 0;  i < n;  ++i) {
            d_total_offdiag
                += 2.0f * calc_D_row(&D[i][0], sum_Y[i], &sum_Y[0], &YYT[i][0], i);
            D[i][i] = 0.0;
            for (unsigned j = 0;  j < i;  ++j)
                D[j][i] = D[i][j];
        }

        t_D += t.elapsed();  t.restart();

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

        t_Q += t.elapsed();  t.restart();

        //for (unsigned i = 0;  i < 3;  ++i)
        //    cerr << "Q[" << i << "] = "
        //         << distribution<float>(&Q[i][0], &Q[i][0] + n)
        //         << endl;
        //cerr << endl;

        
        
        /*********************************************************************/
        // Gradient
        // Implements formula 5 in (Van der Maaten and Hinton, 2008)
        // dC/dy_i = 4 * sum_j ( (p_ij - q_ij)(y_i - y_j)d_ij )

        boost::multi_array<float, 2> PmQ = P - Q;

        //for (unsigned i = 0;  i < 3;  ++i)
        //    cerr << "PmQ[" << i << "] = "
        //         << distribution<float>(&PmQ[i][0], &PmQ[i][0] + n)
        //         << endl;

        boost::multi_array<float, 2> dY(boost::extents[n][d]);

        if (d == 2) {
            for (unsigned j = 0;  j < n;  ++j) {
                for (unsigned i = 0;  i < n;  ++i) {
                    if (i == j) continue;
                    float factor = 4.0f * PmQ[j][i] * D[j][i];
                    dY[i][0] += factor * (Y[i][0] - Y[j][0]);
                    dY[i][1] += factor * (Y[i][1] - Y[j][1]);
                }
            }
        }
        else {
            for (unsigned j = 0;  j < n;  ++j) {
                for (unsigned i = 0;  i < n;  ++i) {
                    if (i == j) continue;
                    float factor = 4.0f * PmQ[j][i] * D[j][i];
                    for (unsigned k = 0;  k < d;  ++k)
                        dY[i][k] += factor * (Y[i][k] - Y[j][k]);
                }
            }
        }

        t_dY += t.elapsed();  t.restart();

        //for (unsigned i = 0;  i < 10;  ++i)
        //    cerr << "dY[" << i << "] = "
        //         << distribution<float>(&dY[i][0], &dY[i][0] + d)
        //         << endl;

        float momentum = (iter < 20
                          ? params.initial_momentum
                          : params.final_momentum);

        // Implement scheme in Jacobs, 1988.  If we go in the same direction as
        // last time, we increase the learning speed of the parameter a bit.
        // If on the other hand the direction changes, we reduce exponentially
        // the rate.

        for (unsigned i = 0;  iter > 0 && i < n;  ++i) {
            // We use != here as we gradients in dY are the negatives of what
            // we want.
            for (unsigned j = 0;  j < d;  ++j) {
                if (dY[i][j] * iY[i][j] < 0.0f)
                    gains[i][j] = gains[i][j] + 0.2f;
                else gains[i][j] = gains[i][j] * 0.8f;
                gains[i][j] = std::max(params.min_gain, gains[i][j]);
            }
        }

        //for (unsigned i = 0;  i < 10;  ++i)
        //    cerr << "gains[" << i << "] = "
        //         << distribution<float>(&gains[i][0], &gains[i][0] + d)
        //         << endl;
        
        for (unsigned i = 0;  i < n;  ++i)
            for (unsigned j = 0;  j < d;  ++j)
                iY[i][j] = momentum * iY[i][j] - (params.eta * gains[i][j] * dY[i][j]);

        //for (unsigned i = 0;  i < 10;  ++i)
        //    cerr << "iY[" << i << "] = "
        //         << distribution<float>(&iY[i][0], &iY[i][0] + d)
        //         << endl;

        Y = Y + iY;

        //for (unsigned i = 0;  i < 10;  ++i)
        //    cerr << "Y[" << i << "] = "
        //         << distribution<float>(&Y[i][0], &Y[i][0] + d)
        //         << endl;

        t_update += t.elapsed();  t.restart();


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

        t_recenter += t.elapsed();  t.restart();

        //for (unsigned i = 0;  i < 10;  ++i)
        //    cerr << "centered Y[" << i << "] = "
        //         << distribution<float>(&Y[i][0], &Y[i][0] + d)
        //         << endl;

        if ((iter + 1) % 20 == 0 || iter == params.max_iter - 1) {
            double cost = kl(P.data(), Q.data(), P.num_elements());
            cerr << "iteration " << (iter + 1)
                 << " cost = " << cost << " elapsed "
                 << timer.elapsed() << endl;
            timer.restart();
        }
        
        t_cost += t.elapsed();  t.restart();

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
