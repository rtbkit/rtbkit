/* feature_set.i                                                   -*- C++ -*-
   Jeremy Barnes, 24 September 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

   SWIG wrapper for the Training_Data class.
*/

%module jml 
%{
#include "boosting/training_data.h"
%}

%include "std_vector.i"

namespace ML {
} // namespace ML

