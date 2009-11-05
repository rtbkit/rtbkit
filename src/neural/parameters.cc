/* parameters.cc                                                   -*- C++ -*-
   Jeremy Barnes, 3 November 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

   Implementation of the logic in the parameters class.
*/

#include "parameters.h"

namespace ML {


Parameters &
Parameters::
add(Parameter_Value * param)
{
    bool inserted = by_name.insert(make_pair(param->name(), param)).second;

    if (!inserted) {
        delete param;
        throw Exception("param with name "
                        + param->name() + " already existed");
    }

    params.push_back(param);
}


void
Parameters::
fill(float value)
{
}

void
Parameters::
random_fill(float limit, Thread_Context & context)
{
}
    
void
Parameters::
operator -= (const Parameters & other)
{
}

void
Parameters::
operator += (const Parameters & other)
{
}

double
Parameters::
two_norm() const
{
    double result = 0.0;

    return sqrt(result);
}

void
Parameters::
operator *= (double value)
{
}


} // namespace ML

