/* simd.h                                                          -*- C++ -*-
   Jeremy Barnes, 21 February 2007
   Copyright (c) 2007 Jeremy Barnes.  All rights reserved.

   Detection of SIMD (vector) unit and associated functions.
*/

#ifndef __arch__simd_h__
#define __arch__simd_h__


#include <string>
#include "cpuid.h"
#include "jml/arch/arch.h"

namespace ML {

#ifdef JML_INTEL_ISA
# ifndef JML_USE_SSE1
#   define JML_USE_SSE1 1
# endif
# ifndef JML_USE_SSE2
#   define JML_USE_SSE2 1
# endif
# ifndef JML_USE_SSE3
#   define JML_USE_SSE3 1
# endif
# ifndef JML_USE_SSE3
#   define JML_USE_SSE3 1
# endif

JML_ALWAYS_INLINE bool has_mmx() { return cpu_info().mmx; }

JML_ALWAYS_INLINE bool has_sse1() { return cpu_info().sse; }

JML_ALWAYS_INLINE bool has_sse2() { return cpu_info().sse2; }

JML_ALWAYS_INLINE bool has_sse3() { return cpu_info().sse3; }

JML_ALWAYS_INLINE bool has_pni() { return cpu_info().pni; }


#endif // __i686__

} // namespace ML

#endif /* __arch__simd_h__ */
