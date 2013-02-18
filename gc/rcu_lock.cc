/* rcu_lock.cc                                                      -*- C++ -*-
   Jeremy Barnes, 20 November 2011
   Copyright (c) 2011 Datacratic.  All rights reserved.

   Garbage collection lock using userspace RCU.
*/
#include "rcu_lock.h"

namespace Datacratic {


int RcuLock::currentIndex = 0;
ML::Thread_Specific<RcuLock::ThreadGcInfo> RcuLock::gcInfo;  ///< Thread-specific bookkeeping


} // namespace Datacratic
