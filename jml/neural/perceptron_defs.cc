/* perceptron_defs.cc
   Jeremy Barnes, 25 May 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

   Definitions for the perceptron.
*/

#include "perceptron_defs.h"
#include <iostream>
#include "jml/utils/string_functions.h"
#include "jml/db/persistent.h"

using namespace std;

namespace ML {

std::string print(Transfer_Function_Type act)
{
    switch (act) {
    case TF_LOGSIG:   return "LOGSIG";
    case TF_TANH:     return "TANH";
    case TF_TANHS:    return "TANHS";
    case TF_IDENTITY: return "IDENTITY";
    case TF_SOFTMAX: return "SOFTMAX";
    case TF_NONSTANDARD: return "NONSTANDARD";

    default: return format("Transfer_Function_Type(%d)", act);
    }
}

std::ostream & operator << (std::ostream & stream, Transfer_Function_Type act)
{
    return stream << print(act);
}

BYTE_PERSISTENT_ENUM_IMPL(Transfer_Function_Type);

std::ostream & operator << (std::ostream & stream, Sampling smp)
{
    switch (smp) {
    case SAMP_DETERMINISTIC:     return stream << "DETERMINISTIC";
    case SAMP_BINARY_STOCHASTIC: return stream << "STOCHASTIC BINARY";
    case SAMP_REAL_STOCHASTIC:   return stream << "STOCHASTIC REAL";
    default: return stream << format("Sampling(%d)", smp);
    }
}

BYTE_PERSISTENT_ENUM_IMPL(Sampling);


} // namespace ML

ENUM_INFO_NAMESPACE

const Enum_Opt<ML::Transfer_Function_Type>
Enum_Info<ML::Transfer_Function_Type>::
OPT[Enum_Info<ML::Transfer_Function_Type>::NUM] = {
    { "logsig",      ML::TF_LOGSIG   },
    { "tanh",        ML::TF_TANH     },
    { "tanhs",       ML::TF_TANHS    },
    { "identity",    ML::TF_IDENTITY },
    { "softmax",     ML::TF_SOFTMAX },
    { "nonstandard", ML::TF_NONSTANDARD }
};

const char * Enum_Info<ML::Transfer_Function_Type>::NAME
    = "Transfer_Function_Type";

const Enum_Opt<ML::Sampling>
Enum_Info<ML::Sampling>::OPT[Enum_Info<ML::Sampling>::NUM] = {
    { "deterministic", ML::SAMP_DETERMINISTIC },
    { "stochastic_bin", ML::SAMP_BINARY_STOCHASTIC },
    { "stochastic_real", ML::SAMP_REAL_STOCHASTIC }
};

const char * Enum_Info<ML::Sampling>::NAME = "Sampling";

END_ENUM_INFO_NAMESPACE
