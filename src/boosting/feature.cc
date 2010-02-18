/* feature.cc
   Jeremy Barnes, 19 March 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

*/

#include "feature.h"
#include "jml/arch/format.h"

namespace ML {


const Feature MISSING_FEATURE (-1, -1, 0);


/*****************************************************************************/
/* FEATURE                                                                   */
/*****************************************************************************/

std::string Feature::print() const
{
    return format("(%d %d %d)", type(), arg1(), arg2());
}

} // namespace ML
