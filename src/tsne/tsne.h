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

boost::multi_array<float, 2>
vectors_to_distances(boost::multi_array<float, 2> & X);


} // namespace ML

#endif /* __jml__tsne__tsne_h__ */
