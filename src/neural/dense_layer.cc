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

/*****************************************************************************/
/* MISSING_VALUES                                                            */
/*****************************************************************************/

BYTE_PERSISTENT_ENUM_IMPL(Missing_Values);

std::string print(Missing_Values mv)
{
    switch (mv) {
    case MV_NONE: return "NONE";
    case MV_ZERO: return "ZERO";
    case MV_INPUT: return "INPUT";
    case MV_DENSE: return "DENSE";
    default: return format("Missing_Values(%d)", mv);
    }
}

std::ostream & operator << (std::ostream & stream, Missing_Values mv)
{
    return stream << print(mv);
}

template class Dense_Layer<float>;
template class Dense_Layer<double>;


} // namespace ML

