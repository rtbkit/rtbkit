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

using namespace std;

namespace ML {

/* Compute the perplexity and the P for a given value of beta */
template<typename Float>
std::pair<double, distribution<Float> >
Hbeta(const distribution<Float> & D, Float beta = 1.0)
{
    distribution<Float> P = exp(D * beta);
    double tot = P.total();
    double H = log(tot) + beta * D.dotprod(P) / tot;
    P /= tot;
    return make_pair(H, P);
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

/** Calculate the beta for a single point. */
void binary_search_perplexity()
{
#if 0
        # Compute the Gaussian kernel and entropy for the current precision
        betamin = -Math.inf; 
        betamax =  Math.inf;
        Di = D[i, Math.concatenate((Math.r_[0:i], Math.r_[i+1:n]))];
        (H, thisP) = Hbeta(Di, beta[i]);
            
        # Evaluate whether the perplexity is within tolerance
        Hdiff = H - logU;
        tries = 0;
        while Math.abs(Hdiff) > tol and tries < 50:
                
            # If not, increase or decrease precision
            if Hdiff > 0:
                betamin = beta[i];
                if betamax == Math.inf or betamax == -Math.inf:
                    beta[i] = beta[i] * 2;
                else:
                    beta[i] = (beta[i] + betamax) / 2;
            else:
                betamax = beta[i];
                if betamin == Math.inf or betamin == -Math.inf:
                    beta[i] = beta[i] / 2;
                else:
                    beta[i] = (beta[i] + betamin) / 2;
            
            # Recompute the values
            (H, thisP) = Hbeta(Di, beta[i]);
            Hdiff = H - logU;
            tries = tries + 1;
            
        # Set the final row of P
        P[i, Math.concatenate((Math.r_[0:i], Math.r_[i+1:n]))] = thisP;
#endif
}


/* Given a matrix of distances, convert to probabilities */
boost::multi_array<float, 2>
distances_to_probabilities(boost::multi_array<float, 2> & D,
                           double tolerance = 1e-5,
                           double perplexity = 30.0)
{
    int n = D.shape()[0];
    if (D.shape()[1] != n)
        throw Exception("D is not square");

    boost::multi_array<float, 2> P(boost::extents[n][n]);

    distribution<float> beta(n, 1.0);

    //double logU = log(perplexity);

    for (unsigned i = 0;  i < n;  ++i) {
        if (i % 500 == 0)
            cerr << "P-values for point " << i << " of " << n << endl;
        
        double betamin = -INFINITY, betamax = INFINITY;
        
        distribution<float> D_row;
        
        for (unsigned iter = 0;  iter != 50 && abs(H - logU) >= tolerance;
             ++iter) {
        }
    }
    
    
#if 0
    # Loop over all datapoints
    for i in range(n):
    
        # Compute the Gaussian kernel and entropy for the current precision
        betamin = -Math.inf; 
        betamax =  Math.inf;
        Di = D[i, Math.concatenate((Math.r_[0:i], Math.r_[i+1:n]))];
        (H, thisP) = Hbeta(Di, beta[i]);
            
        # Evaluate whether the perplexity is within tolerance
        Hdiff = H - logU;
        tries = 0;
        while Math.abs(Hdiff) > tol and tries < 50:
                
            # If not, increase or decrease precision
            if Hdiff > 0:
                betamin = beta[i];
                if betamax == Math.inf or betamax == -Math.inf:
                    beta[i] = beta[i] * 2;
                else:
                    beta[i] = (beta[i] + betamax) / 2;
            else:
                betamax = beta[i];
                if betamin == Math.inf or betamin == -Math.inf:
                    beta[i] = beta[i] / 2;
                else:
                    beta[i] = (beta[i] + betamin) / 2;
            
            # Recompute the values
            (H, thisP) = Hbeta(Di, beta[i]);
            Hdiff = H - logU;
            tries = tries + 1;
            
        # Set the final row of P
        P[i, Math.concatenate((Math.r_[0:i], Math.r_[i+1:n]))] = thisP;
    
    # Return final P-matrix
    print "Mean value of sigma: ", Math.mean(Math.sqrt(1 / beta))
    return P;
#endif    

    return P;
}


    
boost::multi_array<float, 2>
tsne(const boost::multi_array<float, 2> & probs,
     int num_dims = 2)
{
    int n = probs.shape()[0];
    if (n != probs.shape()[1])
        throw Exception("probabilities were the wrong shape");

    boost::multi_array<float, 2> Y(boost::extents[n][d]);
    return result;
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
