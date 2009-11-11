/* auto_encoder_stack.h                                            -*- C++ -*-
   Jeremy Barnes, 11 November 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

   Stack of auto-encoders.
*/

#ifndef __jml__neural__auto_encoder_stack_h__
#define __jml__neural__auto_encoder_stack_h__


#include "auto_encoder.h"
#include "layer_stack.h"

namespace ML {


/*****************************************************************************/
/* AUTO_ENCODER_STACK                                                        */
/*****************************************************************************/

struct Auto_Encoder_Stack : public Auto_Encoder {

    Layer_Stack<Auto_Encoder> layers;
};


} // namespace ML

#endif /* __jml__neural__auto_encoder_stack_h__ */
