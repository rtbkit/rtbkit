/* parameters.cc                                                   -*- C++ -*-
   Jeremy Barnes, 3 November 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

   Implementation of the logic in the parameters class.
*/

#include "parameters.h"
#include "parameters_impl.h"
#include "jml/arch/demangle.h"
#include <typeinfo>


using namespace std;


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

Parameters &
Parameter_Value::
parameters()
{
    throw Exception("attempt to obtain parameters from Parameter_Value of "
                    "class " + demangle(typeid(*this).name()));
}

const Parameters &
Parameter_Value::
parameters() const
{
    throw Exception("attempt to obtain parameters from Parameter_Value of "
                    "class " + demangle(typeid(*this).name()));
}

Vector_Parameter &
Parameter_Value::
vector()
{
    throw Exception("attempt to obtain vector from Parameter_Value of "
                    "class " + demangle(typeid(*this).name()));
}

const Vector_Parameter &
Parameter_Value::
vector() const
{
    throw Exception("attempt to obtain vector from Parameter_Value of "
                    "class " + demangle(typeid(*this).name()));
}

Matrix_Parameter &
Parameter_Value::
matrix()
{
    throw Exception("attempt to obtain matrix from Parameter_Value of "
                    "class " + demangle(typeid(*this).name()));
}

const Matrix_Parameter &
Parameter_Value::
matrix() const
{
    throw Exception("attempt to obtain matrix from Parameter_Value of "
                    "class " + demangle(typeid(*this).name()));
}

void
Parameter_Value::
swap(Parameter_Value & other)
{
    name_.swap(other.name_);
}

void
Parameter_Value::
set_name(const std::string & name)
{
    name_ = name;
}

/*****************************************************************************/
/* VECTOR_PARAMETER                                                          */
/*****************************************************************************/

void
Vector_Parameter::
update(const distribution<float> & dist, float k)
{
    if (dist.size() != size())
        throw Exception("Vector_Parameter::update(): wrong size");
    update(&dist[0], k);
}

void
Vector_Parameter::
update(const distribution<double> & dist, double k)
{
    if (dist.size() != size())
        throw Exception("Vector_Parameter::update(): wrong size");
    update(&dist[0], k);
}


/*****************************************************************************/
/* MATRIX_PARAMETER                                                          */
/*****************************************************************************/

void
Matrix_Parameter::
update_row(int row, const distribution<float> & x, float k)
{
    if (x.size() != size2())
        throw Exception("Matrix_Parameter::update(): wrong size");
    update_row(row, &x[0], k);
}

void
Matrix_Parameter::
update_row(int row, const distribution<double> & x, float k)
{
    if (x.size() != size2())
        throw Exception("Vector_Parameter::update(): wrong size");
    update_row(row, &x[0], k);
}


/*****************************************************************************/
/* VECTOR_REFT                                                               */
/*****************************************************************************/

template struct Vector_RefT<float>;
template struct Vector_RefT<double>;


/*****************************************************************************/
/* MATRIX_REFT                                                               */
/*****************************************************************************/

template struct Matrix_RefT<float>;
template struct Matrix_RefT<double>;


/*****************************************************************************/
/* PARAMETERS                                                                */
/*****************************************************************************/

Vector_Parameter &
Parameters::
vector(int index, const std::string & name)
{
    if (index < 0 || index >= params.size())
        throw Exception("invalid index");
    if (params[index].name() != name) {
        cerr << "index = " << index << " parent = " << this->name()
             << " param name = " << params[index].name() << " wanted = "
             << name << endl;
        throw Exception("wrong name");
    }
    return params[index].vector();
}

const Vector_Parameter &
Parameters::
vector(int index, const std::string & name) const
{
    if (index < 0 || index >= params.size())
        throw Exception("invalid index");
    if (params[index].name() != name)
        throw Exception("wrong name");
    return params[index].vector();
}

Matrix_Parameter &
Parameters::
matrix(int index, const std::string & name)
{
    if (index < 0 || index >= params.size())
        throw Exception("invalid index");
    if (params[index].name() != name)
        throw Exception("wrong name");
    return params[index].matrix();
}

const Matrix_Parameter &
Parameters::
matrix(int index, const std::string & name) const
{
    if (index < 0 || index >= params.size())
        throw Exception("invalid index");
    if (params[index].name() != name)
        throw Exception("wrong name");
    return params[index].matrix();
}

template<typename F>
void
Parameters::
copy_to(F * where, F * limit) const
{
    if (where > limit)
        throw Exception("Parameters::copy_to(): passed limit");

    F * current = where;

    for (Params::const_iterator
             it = params.begin(),
             end = params.end();
         it != end;  ++it) {

        if (current > limit)
            throw Exception("Parameters::copy_to(): out of sync");

        F * cend = current + it->parameter_count();
        it->copy_to(current, cend);
        current = cend;
    }
    
    if (current != limit)
        throw Exception("Parameters::copy_to(): out of sync at end");
}

void
Parameters::
copy_to(float * where, float * limit) const
{
    copy_to<float>(where, limit);
}

void
Parameters::
copy_to(double * where, double * limit) const
{
    copy_to<double>(where, limit);
}

template<typename F>
Parameters *
Parameters::
compatible_ref(F * first, F * last) const
{
    if (first > last) {
        cerr << "name() = " << name() << endl;
        cerr << "type = " << demangle(typeid(*this).name()) << endl;
        cerr << "first = " << first << endl;
        cerr << "last = " << last << endl;
        cerr << "last - first = " << last - first << endl;
        throw Exception("Parameters::compatible_ref(): range oob");
    }

    auto_ptr<Parameters_Ref> result(new Parameters_Ref(name()));

    int i = 0;
    for (Params::const_iterator
             it = params.begin(),
             end = params.end();
         it != end;  ++it) {
        size_t np = it->parameter_count();
        if (first + np > last)
            throw Exception("Parameters::compatible_ref(): bad size");

        result->add(i, it->compatible_ref(first, first + np));

        first += np;
    }

    if (last != first)
        throw Exception("Parameters::compatible_ref(): bad size");
    
    return result.release();
}

Parameters *
Parameters::
compatible_ref(float * first, float * last) const
{
    return compatible_ref<float>(first, last);
}

Parameters *
Parameters::
compatible_ref(double * first, double * last) const
{
    return compatible_ref<double>(first, last);
}

template<typename F>
Parameters *
Parameters::
compatible_copy(F * first, F * last) const
{
    if (first > last) {
        cerr << "name() = " << name() << endl;
        cerr << "type = " << demangle(typeid(*this).name()) << endl;
        cerr << "first = " << first << endl;
        cerr << "last = " << last << endl;
        cerr << "last - first = " << last - first << endl;
        throw Exception("Parameters::compatible_copy(): range oob");
    }

    auto_ptr<Parameters_Ref> result(new Parameters_Ref(name()));

    int i = 0;
    for (Params::const_iterator
             it = params.begin(),
             end = params.end();
         it != end;  ++it) {

        //cerr << "  adding " << it->name() << " with " << it->parameter_count()
        //     << endl;

        size_t np = it->parameter_count();
        if (first + np > last)
            throw Exception("Parameters::compatible_copy(): bad size");

        result->add(i, it->compatible_copy(first, first + np));

        first += np;
    }

    if (last != first)
        throw Exception("Parameters::compatible_copy(): bad size");
    
    return result.release();
}

Parameters *
Parameters::
compatible_copy(float * first, float * last) const
{
    return compatible_copy<float>(first, last);
}

Parameters *
Parameters::
compatible_copy(double * first, double * last) const
{
    return compatible_copy<double>(first, last);
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

Parameters &
Parameters::
add(int index,
    const std::string & name,
    std::vector<float> & values)
{
    return add<float>(index, name, values);
}

Parameters &
Parameters::
add(int index,
    const std::string & name,
    std::vector<double> & values)
{
    return add<double>(index, name, values);
}

Parameters &
Parameters::
add(int index,
    const std::string & name,
    boost::multi_array<float, 2> & values)
{
    return add<float>(index, name, values);
}

Parameters &
Parameters::
add(int index,
    const std::string & name,
    boost::multi_array<double, 2> & values)
{
    return add<double>(index, name, values);
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
fill(double value)
{
    for (Params::iterator it = params.begin(), end = params.end();
         it != end;  ++it)
        it->fill(value);
}

void
Parameters::
update(const Parameter_Value & other, double learning_rate)
{
    if (name() != other.name()) {
        cerr << "name() = " << name() << endl;
        cerr << "other.name() = " << other.name() << endl;
        throw Exception("Parameters::update(): objects have different names");
    }

    const Parameters * cast
        = dynamic_cast<const Parameters *>(&other);
    if (!cast)
        throw Exception("Parameters::update(): other object is not Parameters");

    if (params.size() != cast->params.size()) {
        cerr << "name() = " << name() << endl;
        cerr << "params.size() = " << params.size() << endl;
        cerr << "cast->params.size() = " << cast->params.size() << endl;
        throw Exception("Parameters::update(): differing sizes");
    }

    // Iterate through the two parameters
    Params::iterator
        it = params.begin(),
        iend = params.end();
    Params::const_iterator
        jt = cast->params.begin(),
        jend = cast->params.end();

    for (; it != iend && jt != jend;  ++it, ++jt) {
        if (it->name() != jt->name())
            throw Exception("Parameters::update(): differing names");
        it->update(*jt, learning_rate);
    }
}

void
Parameters::
update(const Parameter_Value & other, const Parameter_Value & learning_rate)
{
    if (name() != other.name()) {
        cerr << "name() = " << name() << endl;
        cerr << "other.name() = " << other.name() << endl;
        throw Exception("Parameters::update(): objects have different names");
    }

    if (name() != learning_rate.name()) {
        cerr << "name() = " << name() << endl;
        cerr << "learning_rate.name() = " << learning_rate.name() << endl;
        throw Exception("Parameters::update(): learning rates have different "
                        "names");
    }

    const Parameters * cast1
        = dynamic_cast<const Parameters *>(&other);
    if (!cast1)
        throw Exception("Parameters::update(): other object is not Parameters");

    if (params.size() != cast1->params.size()) {
        cerr << "name() = " << name() << endl;
        cerr << "params.size() = " << params.size() << endl;
        cerr << "cast1->params.size() = " << cast1->params.size() << endl;
        throw Exception("Parameters::update(): differing sizes");
    }

    const Parameters * cast2
        = dynamic_cast<const Parameters *>(&learning_rate);
    if (!cast2)
        throw Exception("Parameters::update(): learning_rate object is not "
                        "Parameters");

    if (params.size() != cast2->params.size()) {
        cerr << "name() = " << name() << endl;
        cerr << "params.size() = " << params.size() << endl;
        cerr << "cast2->params.size() = " << cast2->params.size() << endl;
        throw Exception("Parameters::update(): differing sizes");
    }

    // Iterate through the two parameters
    Params::iterator
        it = params.begin(),
        iend = params.end();
    Params::const_iterator
        jt = cast1->params.begin(),
        jend = cast1->params.end(),
        lt = cast2->params.begin(),
        lend = cast2->params.end();

    for (; it != iend && jt != jend && lt != lend;  ++it, ++jt, ++lt) {
        if (it->name() != jt->name())
            throw Exception("Parameters::update(): differing names");
        if (it->name() != lt->name())
            throw Exception("Parameters::update(): differing names for lr");
        it->update(*jt, *lt);
    }
}

void
Parameters::
update_sqr(const Parameter_Value & other, double learning_rate)
{
    if (name() != other.name()) {
        cerr << "name() = " << name() << endl;
        cerr << "other.name() = " << other.name() << endl;
        throw Exception("Parameters::update(): objects have different names");
    }

    const Parameters * cast
        = dynamic_cast<const Parameters *>(&other);
    if (!cast)
        throw Exception("Parameters::update(): other object is not Parameters");

    if (params.size() != cast->params.size()) {
        cerr << "name() = " << name() << endl;
        cerr << "params.size() = " << params.size() << endl;
        cerr << "cast->params.size() = " << cast->params.size() << endl;
        throw Exception("Parameters::update(): differing sizes");
    }

    // Iterate through the two parameters
    Params::iterator
        it = params.begin(),
        iend = params.end();
    Params::const_iterator
        jt = cast->params.begin(),
        jend = cast->params.end();

    for (; it != iend && jt != jend;  ++it, ++jt) {
        if (it->name() != jt->name())
            throw Exception("Parameters::update(): differing names");
        it->update_sqr(*jt, learning_rate);
    }
}

Parameters &
Parameters::
subparams(int index, const std::string & name)
{
    if (index < 0 || index > params.size()) {
        cerr << "index = " << index << endl;
        cerr << "name = " << name << endl;
        cerr << "params.size() = " << params.size() << endl;
        throw Exception("Parameters::subparams(): invalid parameters");
    }
    if (index == params.size()) {
        params.push_back(new Parameters_Ref(name));
    }

    return params[index].parameters();
}

const Parameters &
Parameters::
subparams(int index, const std::string & name) const
{
    if (index < 0 || index >= params.size()) {
        cerr << "index = " << index << endl;
        cerr << "name = " << name << endl;
        cerr << "params.size() = " << params.size() << endl;
        throw Exception("Parameters::subparams(): invalid parameters");
    }
    return params[index].parameters();
}

Parameters::
Parameters(const std::string & name)
    : Parameter_Value(name)
{
}

Parameters::
Parameters(const Parameters & other)
    : Parameter_Value(other.name())
{
    int i = 0;
    for (Params::const_iterator
             it = other.params.begin(),
             end = other.params.end();
         it != end;  ++it, ++i) {
        add(i, it->make_copy());
    }
}

Parameters &
Parameters::
operator = (const Parameters & other)
{
    Parameters new_me(other);
    swap(new_me);
    return *this;
}

void
Parameters::
swap(Parameters & other)
{
    Parameter_Value::swap(other);
    params.swap(other.params);
}

void
Parameters::
clear()
{
    params.clear();
}

Parameters *
Parameters::
make_copy() const
{
    return new Parameters(*this);
}

void
Parameters::
set(const Parameter_Value & other)
{
    if (name() != other.name()) {
        cerr << "name() = " << name() << endl;
        cerr << "other.name() = " << other.name() << endl;
        throw Exception("Parameters::set(): objects have different names");
    }

    const Parameters * cast
        = dynamic_cast<const Parameters *>(&other);
    if (!cast)
        throw Exception("Parameters::set(): other object is not Parameters");

    if (params.size() != cast->params.size()) {
        cerr << "name() = " << name() << endl;
        cerr << "params.size() = " << params.size() << endl;
        cerr << "cast->params.size() = " << cast->params.size() << endl;
        throw Exception("Parameters::set(): differing sizes");
    }

    // Iterate through the two parameters
    Params::iterator
        it = params.begin(),
        iend = params.end();
    Params::const_iterator
        jt = cast->params.begin(),
        jend = cast->params.end();

    for (; it != iend && jt != jend;  ++it, ++jt) {
        if (it->name() != jt->name()) {
            cerr << "it->name() = " << it->name() << endl;
            cerr << "jt->name() = " << jt->name() << endl;
            throw Exception("Parameters::set(): differing names");
        }
        it->set(*jt);
    }
}

std::string
Parameters::
parameter_info(int index) const
{
    if (index < 0 || index >= parameter_count())
        throw Exception("parameter_info(): invalid index");

    for (Params::const_iterator it = params.begin(), end = params.end();
         it != end;  index -= it->parameter_count(),  ++it)
        if (index < it->parameter_count())
            return name() + "." + it->parameter_info(index);

    throw Exception("parameter_info(): out of sync");
}


/*****************************************************************************/
/* PARAMETERS_REF                                                            */
/*****************************************************************************/

Parameters_Ref::
Parameters_Ref()
    : Parameters("")
{
}

Parameters_Ref::
Parameters_Ref(const std::string & name)
    : Parameters(name)
{
}

Parameters_Ref *
Parameters_Ref::
make_copy() const
{
    return new Parameters_Ref(*this);
}


/*****************************************************************************/
/* PARAMETERS_COPY                                                           */
/*****************************************************************************/

template class Parameters_Copy<float>;
template class Parameters_Copy<double>;


} // namespace ML

