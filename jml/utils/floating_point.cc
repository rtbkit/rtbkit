#include "floating_point.h"

namespace ML {

const float fp_traits<float>::max_exp_arg = 88.7228f;
const double fp_traits<double>::max_exp_arg = 709.782712893384;
const long double fp_traits<long double>::max_exp_arg = 11356.523406294143;

} // namespace ML
