/* parameters_impl.h                                               -*- C++ -*-
   Jeremy Barnes, 6 November 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

   Implementation of parameters template members.
*/

#ifndef __jml__neural__parameters_impl_h__
#define __jml__neural__parameters_impl_h__

#include "parameters.h"
#include "layer.h"

namespace ML {


/*****************************************************************************/
/* PARAMETERS_COPY                                                           */
/*****************************************************************************/

template<class Float>
Parameters_Copy<Float>::
Parameters_Copy()
{
}

template<class Float>
Parameters_Copy<Float>::
Parameters_Copy(const Parameters_Copy & other)
{
    values.resize(other.parameter_count());

    // Copy all of the parameters in, creating the structure as we go
    Float * start = &values[0], * end = start + values.size();

    // Copy the structure over
    int i = 0;
    for (Params::const_iterator
             it = other.params.begin(),
             end = other.params.end();
         it != end;  ++it, ++i) {
        add(i, it->compatible_copy(start, end));
    }

    if (start != end)
        throw Exception("parameters_copy(): wrong length");
}

template<class Float>
Parameters_Copy<Float> &
Parameters_Copy<Float>::
operator = (const Parameters_Copy & other)
{
    return *this;
}

template<class Float>
Parameters_Copy<Float>::
Parameters_Copy(const Parameters & other)
{
    values.resize(other.parameter_count());

    // Copy all of the parameters in, creating the structure as we go
    Float * start = &values[0], * end = start + values.size();

    // Copy the structure over
    int i = 0;
    for (Params::const_iterator
             it = other.params.begin(),
             end = other.params.end();
         it != end;  ++it, ++i) {
        add(i, it->compatible_copy(start, end));
    }

    if (start != end)
        throw Exception("parameters_copy(): wrong length");
}

template<class Float>
Parameters_Copy<Float>::
Parameters_Copy(const Layer & layer)
{
    const Parameters_Ref & params
        = layer.parameters();
    values.resize(params.parameter_count());
}

template<class Float>
void
Parameters_Copy<Float>::
swap(Parameters_Copy & other)
{
    Parameters::swap(other);
    values.swap(other.values);
}

template<class Float>
float *
Parameters_Copy<Float>::
copy_to(float * where, float * limit) const
{
    if (limit - where < values.size())
        throw Exception("Parameters_Copy::copy_to(): not enough space");
    std::copy(values.begin(), values.end(), where);
    return where + values.size();
}

template<class Float>
double *
Parameters_Copy<Float>::
copy_to(double * where, double * limit) const
{
    if (limit - where < values.size())
        throw Exception("Parameters_Copy::copy_to(): not enough space");
    std::copy(values.begin(), values.end(), where);
    return where + values.size();
}

} // namespace ML


#endif /* __jml__neural__parameters_impl_h__ */
