/* cpu_info.h                                                      -*- C++ -*-
   Jeremy Barnes, 22 January 2010
   Copyright (c) 2010 Jeremy Barnes.  All rights reserved.

   Information about CPUs.
*/

#ifndef __jml__arch__cpu_info_h__
#define __jml__arch__cpu_info_h__

#include "jml/compiler/compiler.h"

namespace ML {

/** Returns the number of CPU cores installed in the system. */

extern int num_cpus_result;

void init_num_cpus();

JML_ALWAYS_INLINE int num_cpus()
{
    if (JML_UNLIKELY(!num_cpus_result)) init_num_cpus();
    return num_cpus_result;
}

} // namespace ML

#endif /* __jml__arch__cpu_info_h__ */
