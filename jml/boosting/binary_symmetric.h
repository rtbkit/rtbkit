/* binary_symmetric.h                                              -*- C++ -*-
   Jeremy Barnes, 17 March 2006
   Copyright (C) 2006 Jeremy Barnes.  All rights reserved.
   $Source$

   Code to deal with binary symmetric classifiers.
*/

#ifndef __boosting__binary_symmetric_h__
#define __boosting__binary_symmetric_h__

#include "config.h"
#include <boost/multi_array.hpp>
#include <vector>

namespace ML {


class Training_Data;
class Feature;


/** Check to see if the given dataset and weights are binary symmetric
    for the given set of features.  Will convert the weights array
    back and forth from a 1-dimensional to a two-dimensional depending
    upon the response.
*/
bool
convert_bin_sym(boost::multi_array<float, 2> & weights,
                const Training_Data & data,
                const Feature & predicted,
                const std::vector<Feature> & features);

/** Same as above, but doesn't convert it. */
bool
is_bin_sym(const boost::multi_array<float, 2> & weights,
           const Training_Data & data,
           const Feature & predicted,
           const std::vector<Feature> & features);

} // namespace ML


#endif /* __boosting__binary_symmetric_h__ */
