/* transfer_function.h                                             -*- C++ -*-
   Jeremy Barnes, 2 November 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

   Definition of transfer functions.
*/

#ifndef __jml__neural__transfer_function_h__
#define __jml__neural__transfer_function_h__

namespace ML {

/*****************************************************************************/
/* TRANSFER_FUNCTION                                                         */
/*****************************************************************************/

struct Transfer_Function {
    virtual ~Transfer_Function();
};


} // namespace ML

#endif /* __jml__neural__transfer_function_h__ */
