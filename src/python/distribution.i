/* distribution.i                                                  -*- C++ -*-
   Jeremy Barnes, 24 September 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

   SWIG wrapper for the distribution template.
*/

%module jml 
%{
#include "jml/stats/distribution.h"
%}

%include "std_vector.i"

%ignore wrong_sizes_exception;

%include "stats/distribution.h"

%template(fvector) std::vector<float>;
%template(dvector) std::vector<double>;
%template(bvector) std::vector<bool>;

%template(fdistribution) ML::Stats::distribution<float>;
%template(ddistribution) ML::Stats::distribution<double>;
%template(bdistribution) ML::Stats::distribution<bool>;

