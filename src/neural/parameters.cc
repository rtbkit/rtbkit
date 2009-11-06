/* parameters.cc                                                   -*- C++ -*-
   Jeremy Barnes, 3 November 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

   Implementation of the logic in the parameters class.
*/

#include "parameters.h"
#include "parameters_impl.h"


namespace ML {


/*****************************************************************************/
/* PARAMETER_VALUE                                                           */
/*****************************************************************************/

Parameter_Value *
Parameter_Value::
compatible_copy(float * first, float * last) const
{
    auto_ptr<Parameter_Value> result(compatible_ref(first, last));
    copy_to(first, last);
    return result.release();
}

Parameter_Value *
Parameter_Value::
compatible_copy(double * first, double * last) const
{
    auto_ptr<Parameter_Value> result(compatible_ref(first, last));
    copy_to(first, last);
    return result.release();
}


/*****************************************************************************/
/* PARAMETERS                                                                */
/*****************************************************************************/

float *
Parameters::
copy_to(float * where, float * limit) const
{
    if (where > limit)
        throw Exception("Parameters::copy_to(): passed limit");

    float * current = where;

    for (Params::const_iterator
             it = params.begin(),
             end = params.end();
         it != end;  ++it) {
        current = it->copy_to(current, limit);
    }

    return current;
}

double *
Parameters::
copy_to(double * where, double * limit) const
{
    if (where > limit)
        throw Exception("Parameters::copy_to(): passed limit");

    double * current = where;

    for (Params::const_iterator
             it = params.begin(),
             end = params.end();
         it != end;  ++it) {
        current = it->copy_to(current, limit);
    }

    return current;
}

Parameters *
Parameters::
compatible_ref(float * first, float * last) const
{
    if (last > first)
        throw Exception("Parameters::compatible_ref(): range oob");

    auto_ptr<Parameters_Ref> result(new Parameters_Ref());

    int i = 0;
    for (Params::const_iterator
             it = params.begin(),
             end = params.end();
         it != end;  ++it) {
        size_t np = it->parameter_count();
        if (first + np > last)
            throw Exception("Parameters::compatible_ref(): bad size");

        result->add(i, it->compatible_ref(first, first + np));
    }

    if (last != first)
        throw Exception("Parameters::compatible_ref(): bad size");
    
    return result.release();
}

Parameters *
Parameters::
compatible_ref(double * first, double * last) const
{
    if (last > first)
        throw Exception("Parameters::compatible_ref(): range oob");

    auto_ptr<Parameters_Ref> result(new Parameters_Ref());

    int i = 0;
    for (Params::const_iterator
             it = params.begin(),
             end = params.end();
         it != end;  ++it) {
        size_t np = it->parameter_count();
        if (first + np > last)
            throw Exception("Parameters::compatible_ref(): bad size");

        result->add(i, it->compatible_ref(first, first + np));
    }

    if (last != first)
        throw Exception("Parameters::compatible_ref(): bad size");
    
    return result.release();
}

Parameters *
Parameters::
compatible_copy(float * first, float * last) const
{
    if (last > first)
        throw Exception("Parameters::compatible_copy(): range oob");

    auto_ptr<Parameters_Ref> result(new Parameters_Ref());

    int i = 0;
    for (Params::const_iterator
             it = params.begin(),
             end = params.end();
         it != end;  ++it) {
        size_t np = it->parameter_count();
        if (first + np > last)
            throw Exception("Parameters::compatible_copy(): bad size");

        result->add(i, it->compatible_copy(first, first + np));
    }

    if (last != first)
        throw Exception("Parameters::compatible_copy(): bad size");
    
    return result.release();
}

Parameters *
Parameters::
compatible_copy(double * first, double * last) const
{
    if (last > first)
        throw Exception("Parameters::compatible_copy(): range oob");

    auto_ptr<Parameters_Ref> result(new Parameters_Ref());

    int i = 0;
    for (Params::const_iterator
             it = params.begin(),
             end = params.end();
         it != end;  ++it) {
        size_t np = it->parameter_count();
        if (first + np > last)
            throw Exception("Parameters::compatible_copy(): bad size");

        result->add(i, it->compatible_copy(first, first + np));
    }

    if (last != first)
        throw Exception("Parameters::compatible_copy(): bad size");
    
    return result.release();
}

Parameters &
Parameters::
add(int index, Parameter_Value * param)
{
    bool inserted
        = by_name.insert(make_pair(param->name(), params.size())).second;

    if (!inserted) {
        delete param;
        throw Exception("param with name "
                        + param->name() + " already existed");
    }

    params.push_back(param);

    return *this;
}

void
Parameters::
serialize(DB::Store_Writer & store) const
{
}

void
Parameters::
reconstitute(DB::Store_Reader & store)
{
}

size_t
Parameters::
parameter_count() const
{
    size_t result = 0;

    for (Params::const_iterator it = params.begin(), end = params.end();
         it != end;  ++it) {
        result += it->parameter_count();
    }

    return result;
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
    return 0.0;
}

void
Parameters::
operator *= (double value)
{
}

void
Parameters::
update(const Parameters & other, double learning_rate)
{
}

Parameters &
Parameters::
subparams(int index, const std::string & name)
{
}

const Parameters &
Parameters::
subparams(int index, const std::string & name) const
{
}

void
Parameters::
add_subparams(int index, Layer & layer)
{
}
    
Parameters::
Parameters()
{
}

Parameters::
Parameters(const Parameters & other)
{
}

Parameters &
Parameters::
operator = (const Parameters & other)
{
    return *this;
}

void
Parameters::
swap(Parameters & other) const
{
}

void
Parameters::
clear()
{
}


/*****************************************************************************/
/* PARAMETER_REF                                                             */
/*****************************************************************************/

Parameters_Ref & subparams(int index, const std::string & name)
{
}

const Parameters_Ref &
subparams(int index, const std::string & name) const
{
}


/*****************************************************************************/
/* PARAMETER_COPY                                                            */
/*****************************************************************************/

template class Parameters_Copy<float>;
template class Parameters_Copy<double>;


} // namespace ML

