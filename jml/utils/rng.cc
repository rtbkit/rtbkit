/* rng.cc
   Jeremy Barnes, 12 May 2012
   Copyright (c) 2012 Datacratic.  All rights reserved.

*/

#include "rng.h"

namespace ML {

/*****************************************************************************/
/* RNG                                                                       */
/*****************************************************************************/

RNG & RNG::defaultRNG()
{
    if (!defaultRNGs.get())
        defaultRNGs.reset(new RNG(0));
    return *defaultRNGs;
}

boost::thread_specific_ptr<RNG> RNG::defaultRNGs;

} // namespace ML
