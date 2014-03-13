/* rt.h                                                            -*- C++ -*-
   Jeremy Barnes, 11 January 2011
   Copyright (c) 2011 Datacratic.  All rights reserved.

   Real-time utilities.
*/

#ifndef __jml__arch__rt_h__
#define __jml__arch__rt_h__

#include <boost/thread.hpp>

namespace ML {


bool makeThreadRealTime(unsigned long long handle, int priority);

/** Make the given boost::thread into a realtime thread with the given
    priority (from zero upwards).  This will put it into the round-robin
    real time scheduling class for the given priority level.

    Note that either a root process or extra capabilities are required to
    enable this functionality.

    Returns whether or not the call succeeded.
*/

inline bool makeThreadRealTime(boost::thread & thread, int priority)
{
    return makeThreadRealTime(thread.native_handle(), priority);
}

} // namespace ML

#endif /* __jml__arch__rt_h__ */

