/* dense_layer.cc
   Jeremy Barnes, 2 November 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

   Dense neural network layer.
*/

#include "dense_layer.h"
#include "dense_layer_impl.h"

using namespace std;
using namespace ML::DB;

namespace ML {


template class Dense_Layer<float>;
template class Dense_Layer<double>;


} // namespace ML

