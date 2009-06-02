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

} // namespace ML

ENUM_INFO_NAMESPACE

const Enum_Opt<ML::Activation>
Enum_Info<ML::Activation>::OPT[5] = {
    { "logsig",      ML::ACT_LOGSIG   },
    { "tanh",        ML::ACT_TANH     },
    { "tanhs",       ML::ACT_TANHS    },
    { "identity",    ML::ACT_IDENTITY },
    { "logsoftmax",  ML::ACT_LOGSOFTMAX }
};

const char * Enum_Info<ML::Activation>::NAME
   = "Activation";

END_ENUM_INFO_NAMESPACE
