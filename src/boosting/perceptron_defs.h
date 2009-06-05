/* perceptron_defs.h                                               -*- C++ -*-
   Jeremy Barnes, 25 May 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

   Definitions for the perceptron.
*/

#ifndef __jml__perceptron_defs_h__
#define __jml__perceptron_defs_h__


#include "utils/enum_info.h"
#include "db/persistent_fwd.h"

namespace ML {
    
/** Activation function for a layer */
enum Activation {
    ACT_LOGSIG,    ///< Log of sigmoid
    ACT_TANH,      
    ACT_TANHS,
    ACT_IDENTITY,
    ACT_LOGSOFTMAX
};

std::ostream & operator << (std::ostream & stream, Activation act);

BYTE_PERSISTENT_ENUM_DECL(Activation);

enum Sampling {
    SAMP_DETERMINISTIC,     /// Deterministic; always the same value
    SAMP_BINARY_STOCHASTIC, /// Stochastic; transfer is P(output == 1)
    SAMP_REAL_STOCHASTIC    /// Stochastic; normal dist with std=1 mean=transfer
};

std::ostream & operator << (std::ostream & stream, Sampling smp);

BYTE_PERSISTENT_ENUM_DECL(Sampling);

} // namespace ML

DECLARE_ENUM_INFO(ML::Activation, 5);
DECLARE_ENUM_INFO(ML::Sampling, 3);


#endif /* __jml__perceptron_defs_h__ */
