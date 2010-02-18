/* serialization_order.h                                           -*- C++ -*-
   Jeremy Barnes, 17 March 2005
   Copyright (c) 2005 Jeremy Barnes.  All rights reserved.
     
   This file is part of "Jeremy's Machine Learning Library", copyright (c)
   1999-2005 Jeremy Barnes.
   
   This program is available under the GNU General Public License, the terms
   of which are given by the file "license.txt" in the top level directory of
   the source code distribution.  If this file is missing, you have no right
   to use the program; please contact the author.

   This program is distributed in the hope that it will be useful, but
   WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
   or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
   for more details.

   ---

   Functions to convert between host and serialization order.  We define
   serialization order as the native byte order on x86 machines; this is
   the opposite to the hton and ntoh macros.
*/

#ifndef __db__serialization_order_h__
#define __db__serialization_order_h__

#include <stdint.h>
#include "jml/compiler/compiler.h"

namespace ML {
namespace DB {

// TODO: on ppc, these need to be byteswapped...

JML_ALWAYS_INLINE uint8_t native_order(uint8_t val)
{
    return val;
}

JML_ALWAYS_INLINE uint16_t native_order(uint16_t val)
{
    return val;
}

JML_ALWAYS_INLINE uint32_t native_order(uint32_t val)
{
    return val;
}

JML_ALWAYS_INLINE uint64_t native_order(uint64_t val)
{
    return val;
}

JML_ALWAYS_INLINE int8_t native_order(int8_t val)
{
    return val;
}

JML_ALWAYS_INLINE int16_t native_order(int16_t val)
{
    return val;
}

JML_ALWAYS_INLINE int32_t native_order(int32_t val)
{
    return val;
}

JML_ALWAYS_INLINE int64_t native_order(int64_t val)
{
    return val;
}

JML_ALWAYS_INLINE uint8_t serialization_order(uint8_t val)
{
    return val;
}

JML_ALWAYS_INLINE uint16_t serialization_order(uint16_t val)
{
    return val;
}

JML_ALWAYS_INLINE uint32_t serialization_order(uint32_t val)
{
    return val;
}

JML_ALWAYS_INLINE uint64_t serialization_order(uint64_t val)
{
    return val;
}

JML_ALWAYS_INLINE int8_t serialization_order(int8_t val)
{
    return val;
}

JML_ALWAYS_INLINE int16_t serialization_order(int16_t val)
{
    return val;
}

JML_ALWAYS_INLINE int32_t serialization_order(int32_t val)
{
    return val;
}

JML_ALWAYS_INLINE int64_t serialization_order(int64_t val)
{
    return val;
}

} // namespace DB
} // namespace ML

#endif /* __db__serialization_order_h__ */
