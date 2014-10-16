/* dense_shared_reverse_layer.cc
   Jeremy Barnes, 9 November 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

   Dense shared reverse layer.
*/

#include "dense_shared_reverse_layer_impl.h"

namespace ML {

extern template class Dense_Shared_Reverse_Layer<float>;
extern template class Dense_Shared_Reverse_Layer<double>;

} // namespace ML
