/* perceptron_defs.h                                               -*- C++ -*-
   Jeremy Barnes, 25 May 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

   Definitions for the perceptron.
*/

#ifndef __jml__perceptron_defs_h__
#define __jml__perceptron_defs_h__


#include "jml/utils/enum_info.h"
#include "jml/db/persistent_fwd.h"

namespace ML {
    
/** Transfer function for a neural network layer */
enum Transfer_Function_Type {
    TF_LOGSIG,      ///< Log of sigmoid
    TF_TANH,        ///< tanh function
    TF_TANHS,       ///< 1.7159 tanh(2/3x)
    TF_IDENTITY,    ///< Identity function
    TF_SOFTMAX,     ///< Pseudo-probabilistic
    TF_NONSTANDARD
};

std::string print(Transfer_Function_Type tf);

std::ostream & operator << (std::ostream & stream, Transfer_Function_Type fn);

BYTE_PERSISTENT_ENUM_DECL(Transfer_Function_Type);

enum Sampling {
    SAMP_DETERMINISTIC,     /// Deterministic; always the same value
    SAMP_BINARY_STOCHASTIC, /// Stochastic; transfer is P(output == 1)
    SAMP_REAL_STOCHASTIC    /// Stochastic; normal dist with std=1 mean=transfer
};

std::ostream & operator << (std::ostream & stream, Sampling smp);

BYTE_PERSISTENT_ENUM_DECL(Sampling);

} // namespace ML

DECLARE_ENUM_INFO(ML::Transfer_Function_Type, 6);
DECLARE_ENUM_INFO(ML::Sampling, 3);


#endif /* __jml__perceptron_defs_h__ */
