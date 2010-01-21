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

#include "stats/distribution.h"
#include <boost/multi_array.hpp>


namespace ML {

std::pair<double, distribution<float> >
perplexity_and_prob(const distribution<float> & D, double beta = 1.0,
                    int i = -1);

std::pair<double, distribution<double> >
perplexity_and_prob(const distribution<double> & D, double beta = 1.0,
                    int i = -1);

boost::multi_array<float, 2>
vectors_to_distances(boost::multi_array<float, 2> & X);

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
          min_gain(0.01)
    {
    }

    int max_iter;
    float initial_momentum;
    float final_momentum;
    float eta;
    float min_gain;
};

boost::multi_array<float, 2>
tsne(const boost::multi_array<float, 2> & probs,
     int num_dims = 2,
     const TSNE_Params & params = TSNE_Params());


} // namespace ML

#endif /* __jml__tsne__tsne_h__ */
