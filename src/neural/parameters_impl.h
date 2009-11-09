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
    : Parameters("")
{
}

template<class Float>
Parameters_Copy<Float>::
Parameters_Copy(const Parameters_Copy & other)
    : Parameters(other.name())
{
    values.resize(other.parameter_count());

    // Copy all of the parameters in, creating the structure as we go
    Float * start = &values[0], * finish = start + values.size();

    // Copy the structure over
    int i = 0;
    for (Params::const_iterator
             it = other.params.begin(),
             end = other.params.end();
         it != end;  ++it, ++i) {
        add(i, it->compatible_copy(start, finish));
    }

    if (start != finish)
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
    : Parameters(other.name())
{
    values.resize(other.parameter_count());

    // Copy all of the parameters in, creating the structure as we go
    Float * start = &values[0], * finish = start + values.size();

    // Copy the structure over
    int i = 0;
    for (Params::const_iterator
             it = other.params.begin(),
             end = other.params.end();
         it != end;  ++it, ++i) {
        add(i, it->compatible_copy(start, finish));
    }

    if (start != finish)
        throw Exception("parameters_copy(): wrong length");
}

template<class Float>
Parameters_Copy<Float>::
Parameters_Copy(const Layer & layer)
    : Parameters(layer.name())
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

template<class Float>
void
Parameters_Copy<Float>::
set(const Parameter_Value & other)
{
    // Try to do it via a vector operation if possible
    {
        const Parameters_Copy<Float> * cast
            = dynamic_cast<const Parameters_Copy<Float> *>(&other);
        if (cast) {
            if (cast->values.size() != values.size())
                throw Exception("Parameters_Copy::set(): incompatible");
            std::copy(cast->values.begin(), cast->values.end(),
                      values.begin());
            return;
        }
    }

    {
        const Parameters_Copy<float> * cast
            = dynamic_cast<const Parameters_Copy<float> *>(&other);
        if (cast) {
            if (cast->values.size() != values.size())
                throw Exception("Parameters_Copy::set(): incompatible");
            std::copy(cast->values.begin(), cast->values.end(),
                      values.begin());
            return;
        }
    }

    {
        const Parameters_Copy<double> * cast
            = dynamic_cast<const Parameters_Copy<double> *>(&other);
        if (cast) {
            if (cast->values.size() != values.size())
                throw Exception("Parameters_Copy::set(): incompatible");
            std::copy(cast->values.begin(), cast->values.end(),
                      values.begin());
            return;
        }
    }

    // Otherwise, do it structurally
    Parameters::set(other);
}

} // namespace ML


#endif /* __jml__neural__parameters_impl_h__ */
