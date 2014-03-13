/* arch.h                                                          -*- C++ -*-
   Jeremy Barnes, 22 February 2007
   Copyright (c) 2007 Jeremy Barnes.  All rights reserved.

   Detection of the architecture.
*/

#ifndef __arch__arch_h__
#define __arch__arch_h__

#if defined(__i386__) || defined(__amd64__)
# define JML_INTEL_ISA 1
# if defined(__amd64__)
#  define JML_BITS 64
# else
#  define JML_BITS 32
# endif // 32/64 bits
#endif // intel ISA

#endif /* __arch__arch_h__ */
