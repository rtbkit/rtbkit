/* tsne.h                                                          -*- C++ -*-
   Jeremy Barnes, 16 January 2010
   Copyright (c) 2010 Jeremy Barnes.  All rights reserved.

   Implementation of the TSNE dimensionality reduction algorithm, particularly
   useful for visualization of data.

   See http://ict.ewi.tudelft.nl/~lvandermaaten/t-SNE.html

   L.J.P. van der Maaten and G.E. Hinton.
   Visualizing High-Dimensional Data Using t-SNE.
   Journal of Machine Learning Research 9(Nov):2579-2605, 2008.
*/

#ifndef __jml__tsne__tsne_h__
#define __jml__tsne__tsne_h__

#include "jml/stats/distribution.h"
#include <boost/multi_array.hpp>
#include <boost/function.hpp>

namespace ML {

std::pair<double, distribution<float> >
perplexity_and_prob(const distribution<float> & D, double beta = 1.0,
                    int i = -1);

std::pair<double, distribution<double> >
perplexity_and_prob(const distribution<double> & D, double beta = 1.0,
                    int i = -1);

/** Given a matrix that gives the a number of points in a vector space of
    dimension d (ie, a number of points with coordinates of d dimensions),
    convert to a matrix that gives the square of the distance between
    each of the points.

                     2
    D   = ||X  - X || 
     ij      i    j

    params:
    X    a (n x d) matrix, where n is the number of points and d is the
         number of coordinates that each point has
    D    a (n x n) matrix that will be filled in with the distance between
         any of the two points.  Note that by definition the diagonal is
         zero and the matrix is symmetric; as a result only the lower
         diagonal needs to be filled in.
    fill_upper if set, the lower diagonal will be copied into the upper
               diagonal so that the entire matrix is filled in.
*/
void
vectors_to_distances(const boost::multi_array<float, 2> & X,
                     boost::multi_array<float, 2> & D,
                     bool fill_upper = true);

void
vectors_to_distances(const boost::multi_array<double, 2> & X,
                     boost::multi_array<double, 2> & D,
                     bool fill_upper = true);

inline boost::multi_array<float, 2>
vectors_to_distances(boost::multi_array<float, 2> & X,
                     bool fill_upper = true)
{
    int n = X.shape()[0];
    boost::multi_array<float, 2> result(boost::extents[n][n]);
    vectors_to_distances(X, result, fill_upper);
    return result;
}

inline boost::multi_array<double, 2>
vectors_to_distances(boost::multi_array<double, 2> & X,
                     bool fill_upper = true)
{
    int n = X.shape()[0];
    boost::multi_array<double, 2> result(boost::extents[n][n]);
    vectors_to_distances(X, result, fill_upper);
    return result;
}

boost::multi_array<float, 2>
distances_to_probabilities(boost::multi_array<float, 2> & D,
                           double tolerance = 1e-5,
                           double perplexity = 30.0);

/** Perform a principal component analysis.  This routine will reduce a
    (n x d) matrix to a (n x e) matrix, where e < d (and is possibly far less).
    The num_dims parameter gives the preferred value of e; it is possible that
    the routine will return a smaller value of e than this (where the rank of
    X is lower than the requested e value).
*/
boost::multi_array<float, 2>
pca(boost::multi_array<float, 2> & coords, int num_dims = 50);

struct TSNE_Params {
    
    TSNE_Params()
        : max_iter(1000),
          initial_momentum(0.5),
          final_momentum(0.8),
          eta(500),
          min_gain(0.01),
          min_prob(1e-12)
    {
    }

    int max_iter;
    double initial_momentum;
    double final_momentum;
    double eta;
    double min_gain;
    double min_prob;
};

// Function that will be used as a callback to provide progress to a calling
// process.  Arguments are:
// - int: iteration number
// - float: cost when last measured
// - const char *: phase name (of this iteration)
// - TSNE_Params &: parameters (may be modified)
// The return should be true to keep going, or false to stop (the most recent
// Y will then be returned).
typedef boost::function<bool (int, float, const char *)>
TSNE_Callback;

boost::multi_array<float, 2>
tsne(const boost::multi_array<float, 2> & probs,
     int num_dims = 2,
     const TSNE_Params & params = TSNE_Params(),
     const TSNE_Callback & callback = TSNE_Callback());


} // namespace ML

#endif /* __jml__tsne__tsne_h__ */
