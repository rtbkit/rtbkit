/* perceptron_defs.cc
   Jeremy Barnes, 25 May 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

   Definitions for the perceptron.
*/

#include "perceptron_defs.h"
#include <iostream>
#include "utils/string_functions.h"
#include "db/persistent.h"

using namespace std;

namespace ML {

std::ostream & operator << (std::ostream & stream, Activation act)
{
    switch (act) {
    case ACT_LOGSIG:   return stream << "LOGSIG";
    case ACT_TANH:     return stream << "TANH";
    case ACT_TANHS:    return stream << "TANHS";
    case ACT_IDENTITY: return stream << "IDENTITY";
    case ACT_LOGSOFTMAX: return stream << "LOGSOFTMAX";

    default: return stream << format("Activation(%d)", act);
    }
}

BYTE_PERSISTENT_ENUM_IMPL(Activation);

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

const Enum_Opt<ML::Activation>
Enum_Info<ML::Activation>::OPT[Enum_Info<ML::Activation>::NUM] = {
    { "logsig",      ML::ACT_LOGSIG   },
    { "tanh",        ML::ACT_TANH     },
    { "tanhs",       ML::ACT_TANHS    },
    { "identity",    ML::ACT_IDENTITY },
    { "logsoftmax",  ML::ACT_LOGSOFTMAX }
};

const char * Enum_Info<ML::Activation>::NAME = "Activation";

const Enum_Opt<ML::Sampling>
Enum_Info<ML::Sampling>::OPT[Enum_Info<ML::Sampling>::NUM] = {
    { "deterministic", ML::SAMP_DETERMINISTIC },
    { "stochastic_bin", ML::SAMP_BINARY_STOCHASTIC },
    { "stochastic_real", ML::SAMP_REAL_STOCHASTIC }
};

const char * Enum_Info<ML::Sampling>::NAME = "Sampling";

END_ENUM_INFO_NAMESPACE
