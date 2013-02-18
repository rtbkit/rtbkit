/* rt.cc
   Jeremy Barnes, 11 January 2011
   Copyright (c) 2011 Datacratic.  All rights reserved.

   Real-time utilities.
*/

#include "rt.h"

namespace ML {

bool makeThreadRealTime(unsigned long long handle,
                        int priority)
{
    struct sched_param sched;
    sched.sched_priority = priority;

    int res = pthread_setschedparam(handle, SCHED_RR, &sched);
    if (res != 0) {
        //cerr << "error setting realtime priority for thread: "
        //     << strerror(errno) << endl;
        return false;
    }

    return true;
}

} // namespace ML

