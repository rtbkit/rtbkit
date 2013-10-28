%module "RTBKIT.api"
%import "std_vector.i"
%import "std_string.i"
#include "bidder.h"
namespace std {
   %template(vectors) vector<string>;
};
