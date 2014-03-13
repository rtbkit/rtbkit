/* hex_dump.h                                                      -*- C++ -*-
   Jeremy Barnes, 6 October 2010
   Copyright (c) 2010 Jeremy Barnes.  All rights reserved.
   Copyright (c) 2010 Datacratic.  All rights reserved.

   Routine to dump memory in hex format.
*/

#ifndef __utils__hex_dump_h__
#define __utils__hex_dump_h__

#include <stddef.h>

namespace ML {

/** Dump the given range of memory (up to a minimum of total_memory and
    max_size) as a hex/ascii dump to the screen.
*/
void hex_dump(const void * mem, size_t total_memory, size_t max_size = 1024);


} // namespace ML

#endif /* __utils__hex_dump_h__ */
