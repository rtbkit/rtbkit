/* cuda.h                                                          -*- C++ -*-
   Jeremy Barnes, 10 March 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

   Implementation of the interface for the CUDA libraries.  Delayed loading
   to allow dlopening if they exist.
*/

#ifndef __arch__cuda_h__
#define __arch__cuda_h__

namespace ML {

void init_cuda();

} // namespace ML

#endif /* __arch__cuda_h__ */
