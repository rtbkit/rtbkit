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
/* VECTOR_REFT                                                               */
/*****************************************************************************/

template<typename Underlying>
struct Vector_RefT : public Vector_Parameter {

    Vector_RefT(const std::string & name,
                Underlying * array, size_t size)
        : Vector_Parameter(name), array_(array), size_(size)
    {
    }

    virtual size_t parameter_count() const
    {
        return size_;
    }

    virtual void copy_to(float * where, float * limit) const
    {
        if (limit - where != size_)
            throw Exception("copy_to(): wrong size array");
        std::copy(array_, array_ + size_, where);
    }

    virtual void copy_to(double * where, double * limit) const
    {
        if (limit - where != size_)
            throw Exception("copy_to(): wrong size array");
        std::copy(array_, array_ + size_, where);
    }

    virtual Parameter_Value *
    compatible_ref(float * first, float * last) const
    {
        if (first + size_ != last)
            throw Exception("Vector_Ref::compatible_ref(): wrong size");
        return new Vector_RefT<float>(name(), first, size_);
    }

    virtual Parameter_Value *
    compatible_ref(double * first, double * last) const
    {
        if (first + size_ != last)
            throw Exception("Vector_Ref::compatible_ref(): wrong size");
        return new Vector_RefT<double>(name(), first, size_);
    }

    virtual void update(const float * x, float k)
    {
        SIMD::vec_add(array_, k, x, array_, size_);
    }

    virtual void update(const double * x, double k)
    {
        SIMD::vec_add(array_, k, x, array_, size_);
    }

    virtual void set(const Parameter_Value & other)
    {
        if (name() != other.name())
            throw Exception("VectorRefT::set(): objects have different names");
        other.copy_to(array_, array_ + size_);
    }

    virtual void fill(double value)
    {
        std::fill(array_, array_ + size_, value);
    }

    virtual void update_element(int element, float update_by)
    {
        if (element < 0 || element >= size_)
            throw Exception("update_element(): out of range");
        array_[element] += update_by;
    }

    virtual void update_element(int element, double update_by)
    {
        if (element < 0 || element >= size_)
            throw Exception("update_element(): out of range");
        array_[element] += update_by;
    }
    
protected:
    Underlying * array_;
    size_t size_;
};

extern template struct Vector_RefT<float>;
extern template struct Vector_RefT<double>;


/*****************************************************************************/
/* MATRIX_REFT                                                               */
/*****************************************************************************/

template<typename Underlying>
struct Matrix_RefT : public Matrix_Parameter {

    Matrix_RefT(const std::string & name,
                Underlying * array, size_t size1, size_t size2)
        : Matrix_Parameter(name),
          array_(array), size1_(size1), size2_(size2)
    {
    }

    virtual size_t parameter_count() const
    {
        return size1_ * size2_;
    }

    virtual void copy_to(float * where, float * limit) const
    {
        size_t n = parameter_count();
        if (limit - where != n)
            throw Exception("copy_to(): wrong size matrix");
        std::copy(array_, array_ + n, where);
    }

    virtual void copy_to(double * where, double * limit) const
    {
        size_t n = parameter_count();
        if (limit - where != n)
            throw Exception("copy_to(): wrong size matrix");
        std::copy(array_, array_ + n, where);
    }

    virtual void set(const Parameter_Value & other)
    {
        if (name() != other.name())
            throw Exception("MatrixRefT::set(): objects have different names");
        size_t n = parameter_count();
        other.copy_to(array_, array_ + n);
    }

    virtual void fill(double value)
    {
        std::fill(array_, array_ + size1_ * size2_, value);
    }

    virtual Parameter_Value *
    compatible_ref(float * first, float * last) const
    {
        size_t n = parameter_count();
        if (first + n != last) {
            using namespace std;
            cerr << "n = " << n << endl;
            cerr << "last - first = " << last - first << endl;
            cerr << "name() = " << name() << endl;
            throw Exception("Matrix_Ref::compatible_ref(): wrong size");
        }
        return new Matrix_RefT<float>(name(), first, size1_, size2_);
    }

    virtual Parameter_Value *
    compatible_ref(double * first, double * last) const
    {
        size_t n = parameter_count();
        if (first + n != last)
            throw Exception("Matrix_Ref::compatible_ref(): wrong size");
        return new Matrix_RefT<double>(name(), first, size1_, size2_);
    }

    virtual void update_row(int row, const float * x, float k = 1.0)
    {
        if (row < 0 || row >= size1_)
            throw Exception("update_row: invalid row");
        SIMD::vec_add(array_ + (size1_ * row), k, x,
                      array_ + (size1_ * row), size2_);
    }

    virtual void update_row(int row, const double * x, double k = 1.0)
    {
        if (row < 0 || row >= size1_)
            throw Exception("update_row: invalid row");
        SIMD::vec_add(array_ + (size1_ * row), k, x,
                      array_ + (size1_ * row), size2_);
    }
    
protected:
    Underlying * array_;
    size_t size1_, size2_;
};

extern template struct Matrix_RefT<float>;
extern template struct Matrix_RefT<double>;


/*****************************************************************************/
/* PARAMETERS                                                                */
/*****************************************************************************/

template<class Float>
Parameters &
Parameters::
add(int index,
    const std::string & name,
    std::vector<Float> & values)
{
    return add(index,
               new Vector_RefT<Float>(name, &values[0], values.size()));
}
    
template<class Float>
Parameters &
Parameters::
add(int index,
    const std::string & name,
    boost::multi_array<Float, 2> & values)
{
    return add(index,
               new Matrix_RefT<Float>(name, values.data(),
                                      values.shape()[0],
                                      values.shape()[1]));
}


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
    values = other.values;

    // Copy all of the parameters in, creating the structure as we go
    Float * start = &values[0], * finish = start + values.size(),
        * current = start;

    // Copy the structure over
    int i = 0;
    for (Params::const_iterator
             it = other.params.begin(),
             end = other.params.end();
         it != end;  ++it, ++i) {
        size_t n = it->parameter_count();

        add(i, it->compatible_copy(current, current + n));
        current += n;
    }

    if (current != finish)
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
    Float * start = &values[0], * finish = start + values.size(),
        * current = start;

    // Copy the structure over
    int i = 0;
    for (Params::const_iterator
             it = other.params.begin(),
             end = other.params.end();
         it != end;  ++it, ++i) {
        size_t n = it->parameter_count();

        add(i, it->compatible_copy(current, current + n));
        current += n;
    }

    if (current != finish)
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
void
Parameters_Copy<Float>::
copy_to(float * where, float * limit) const
{
    if (limit - where != values.size())
        throw Exception("Parameters_Copy::copy_to(): wrong sized space");
    std::copy(values.begin(), values.end(), where);
}

template<class Float>
void
Parameters_Copy<Float>::
copy_to(double * where, double * limit) const
{
    if (limit - where != values.size())
        throw Exception("Parameters_Copy::copy_to(): wrong sized space");
    std::copy(values.begin(), values.end(), where);
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

template<class Float>
void
Parameters_Copy<Float>::
fill(double value)
{
    values.fill(value);
}

} // namespace ML


#endif /* __jml__neural__parameters_impl_h__ */
