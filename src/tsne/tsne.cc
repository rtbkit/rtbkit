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
    P /= tot;

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

boost::multi_array<float, 2>
tsne(const boost::multi_array<float, 2> & probs,
     int num_dims = 2)
{
    int n = probs.shape()[0];
    if (n != probs.shape()[1])
        throw Exception("probabilities were the wrong shape");

    int d = num_dims;

    boost::multi_array<float, 2> Y(boost::extents[n][d]);
    return Y;
}

#if 0
def tsne(X = Math.array([]), no_dims = 2, initial_dims = 50, perplexity = 30.0, use_pca=True):
    //Runs t-SNE on the dataset in the NxD array X to reduce its dimensionality to no_dims dimensions.
    //The syntaxis of the function is Y = tsne.tsne(X, no_dims, perplexity), where X is an NxD NumPy array.
    # Check inputs

    max_iter = 1000;
    initial_momentum = 0.5;
    final_momentum = 0.8;
    eta = 500;
    min_gain = 0.01;
    Y = Math.random.randn(n, no_dims);
    dY = Math.zeros((n, no_dims));
    iY = Math.zeros((n, no_dims));
    gains = Math.ones((n, no_dims));
    
    # Compute P-values
    P = x2p(X, 1e-5, perplexity);
    P = P + Math.transpose(P);
    P = P / Math.sum(P);
    P = P * 4;                                    # early exaggeration
    P = Math.maximum(P, 1e-12);
    
    # Run iterations
    for iter in range(max_iter):
        
        # Compute pairwise affinities
        sum_Y = Math.sum(Math.square(Y), 1);        
        num = 1 / (1 + Math.add(Math.add(-2 * Math.dot(Y, Y.T), sum_Y).T, sum_Y));
        num[range(n), range(n)] = 0;
        Q = num / Math.sum(num);
        Q = Math.maximum(Q, 1e-12);
        
        # Compute gradient
        PQ = P - Q;
        for i in range(n):
            dY[i,:] = Math.sum(Math.tile(PQ[:,i] * num[:,i], (no_dims, 1)).T * (Y[i,:] - Y), 0);
            
        # Perform the update
        if iter < 20:
            momentum = initial_momentum
        else:
            momentum = final_momentum
        gains = (gains + 0.2) * ((dY > 0) != (iY > 0)) + (gains * 0.8) * ((dY > 0) == (iY > 0));
        gains[gains < min_gain] = min_gain;
        iY = momentum * iY - eta * (gains * dY);
        Y = Y + iY;
        Y = Y - Math.tile(Math.mean(Y, 0), (n, 1));
        
        # Compute current value of cost function
        if (iter + 1) % 10 == 0:
            C = Math.sum(P * Math.log(P / Q));
            print "Iteration ", (iter + 1), ": error is ", C
            
        # Stop lying about P-values
        if iter == 100:
            P = P / 4;
            
    # Return solution
    return Y;
#endif

} // namespace ML
