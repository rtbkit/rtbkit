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


namespace {

template<typename F>
bool need_update(const F * vals, size_t size)
{
#if 0 // check the whole range for NaN
    bool result = false;
    for (unsigned i = 0;  i < size;  ++i) {
        if (isnan(vals[i]))
            throw Exception("updating with range containing NaN");
        if (vals[i] != 0.0)
            result = true;
    }
    return result;
#else // check for NaN until we find a single non-zero value
    for (unsigned i = 0;  i < size;  ++i) {
        if (isnan(vals[i]))
            throw Exception("updating with range containing NaN");
        if (vals[i] != 0.0) return true;
    }
    return false;
#endif
}

} // file scope


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

    virtual size_t size() const
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

    virtual void
    update(const Parameter_Value & other, double learning_rate)
    {
        const Vector_Parameter & vp = other.vector();

        // Try to do it via a vector operation if possible
        {
            const Vector_RefT<Underlying> * cast
                = dynamic_cast<const Vector_RefT<Underlying> *>(&vp);
            if (cast) {
                if (cast->size_ != size_)
                    throw Exception("Parameters_Copy::set(): incompatible");

                if (need_update(cast->array_, size_))
                    SIMD::vec_add(array_, learning_rate, cast->array_, array_,
                                  size_);
                return;
            }
        }

        {
            const Vector_RefT<float> * cast
                = dynamic_cast<const Vector_RefT<float> *>(&vp);
            if (cast) {
                if (cast->size_ != size_)
                    throw Exception("Parameters_Copy::set(): incompatible");

                if (need_update(cast->array_, size_))
                    SIMD::vec_add(array_, learning_rate, cast->array_, array_,
                                  size_);
                return;
            }
        }

        {
            const Vector_RefT<double> * cast
                = dynamic_cast<const Vector_RefT<double> *>(&vp);
            if (cast) {
                if (cast->size_ != size_)
                    throw Exception("Parameters_Copy::set(): incompatible");

                if (need_update(cast->array_, size_))
                    SIMD::vec_add(array_, learning_rate, cast->array_, array_,
                                  size_);
                return;
            }
        }
    
        // Otherwise, do it via a copy through a known type (here, double)
        double tmp[size_];
        vp.copy_to(tmp, tmp + size_);
        
        if (need_update(tmp, size_))
            SIMD::vec_add(array_, learning_rate, tmp, array_, size_);
    }

    virtual void
    update(const Parameter_Value & other,
           const Parameter_Value & learning_rate)
    {
        const Vector_Parameter & v_other = other.vector();
        const Vector_Parameter & v_learning_rate = learning_rate.vector();

        if (name() != v_other.name() || size_ != v_other.size())
            throw Exception("update with incompatible object");

        if (name() != v_learning_rate.name() || size_ != v_learning_rate.size())
            throw Exception("update with incompatible learning_rate");

        double tmp1[size_];
        v_other.copy_to(tmp1, tmp1 + size_);
        
        double tmp2[size_];
        v_learning_rate.copy_to(tmp2, tmp2 + size_);

        if (need_update(tmp1, size_))
            SIMD::vec_add(array_, tmp1, tmp2, array_, size_);
    }

    virtual void
    update_sqr(const Parameter_Value & other, double learning_rate)
    {
        const Vector_Parameter & vp = other.vector();

        // Try to do it via a vector operation if possible
        {
            const Vector_RefT<Underlying> * cast
                = dynamic_cast<const Vector_RefT<Underlying> *>(&vp);
            if (cast) {
                if (cast->size_ != size_)
                    throw Exception("Parameters_Copy::set(): incompatible");

                if (need_update(cast->array_, size_))
                    SIMD::vec_add_sqr(array_, learning_rate, cast->array_,
                                      array_, size_);
                return;
            }
        }

        {
            const Vector_RefT<float> * cast
                = dynamic_cast<const Vector_RefT<float> *>(&vp);
            if (cast) {
                if (cast->size_ != size_)
                    throw Exception("Parameters_Copy::set(): incompatible");

                if (need_update(cast->array_, size_))
                    SIMD::vec_add_sqr(array_, learning_rate, cast->array_,
                                      array_, size_);
                return;
            }
        }

        {
            const Vector_RefT<double> * cast
                = dynamic_cast<const Vector_RefT<double> *>(&vp);
            if (cast) {
                if (cast->size_ != size_)
                    throw Exception("Parameters_Copy::set(): incompatible");

                if (need_update(cast->array_, size_))
                    SIMD::vec_add_sqr(array_, learning_rate, cast->array_,
                                      array_, size_);
                return;
            }
        }
    
        // Otherwise, do it via a copy through a known type (here, double)
        double tmp[size_];
        vp.copy_to(tmp, tmp + size_);
        
        if (need_update(tmp, size_))
            SIMD::vec_add_sqr(array_, learning_rate, tmp, array_, size_);
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
        if (k != 0.0 && need_update(x, size_))
            SIMD::vec_add(array_, k, x, array_, size_);
    }

    virtual void update(const double * x, double k)
    {
        if (k != 0.0 && need_update(x, size_))
            SIMD::vec_add(array_, k, x, array_, size_);
    }

    virtual void update_sqr(const float * x, float k)
    {
        if (k != 0.0 && need_update(x, size_))
            SIMD::vec_add_sqr(array_, k, x, array_, size_);
    }

    virtual void update_sqr(const double * x, double k)
    {
        if (k != 0.0 && need_update(x, size_))
            SIMD::vec_add_sqr(array_, k, x, array_, size_);
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

    virtual Vector_RefT * make_copy() const
    {
        return new Vector_RefT(*this);
    }

    virtual std::string parameter_info(int index) const
    {
        if (index < 0 || index >= size_)
            throw Exception("size was wrong");
        return format("%s (vector) element %d",
                      name().c_str(), index);
    }
    
protected:
    Underlying * array_;
    size_t size_;
    template<typename U> friend class Vector_RefT;
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

    virtual size_t size1() const { return size1_; }
    virtual size_t size2() const { return size2_; }

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

    virtual void
    update(const Parameter_Value & other, double learning_rate)
    {
        size_t n = size1_ * size2_;

        const Matrix_Parameter & mp = other.matrix();

        // Try to do it via a vector operation if possible
        {
            const Matrix_RefT<Underlying> * cast
                = dynamic_cast<const Matrix_RefT<Underlying> *>(&mp);
            if (cast) {
                if ((size1_ != cast->size1_) || (size2_ != cast->size2_))
                    throw Exception("Parameters_Copy::set(): incompatible");
                if (need_update(cast->array_, n))
                    SIMD::vec_add(array_, learning_rate, cast->array_, array_,
                                  n);
                return;
            }
        }

        {
            const Matrix_RefT<float> * cast
                = dynamic_cast<const Matrix_RefT<float> *>(&mp);
            if (cast) {
                if ((size1_ != cast->size1_) || (size2_ != cast->size2_))
                    throw Exception("Parameters_Copy::set(): incompatible");
                if (need_update(cast->array_, n))
                    SIMD::vec_add(array_, learning_rate, cast->array_, array_,
                                  n);
                return;
            }
        }

        {
            const Matrix_RefT<double> * cast
                = dynamic_cast<const Matrix_RefT<double> *>(&mp);
            if (cast) {
                if ((size1_ != cast->size1_) || (size2_ != cast->size2_))
                    throw Exception("Parameters_Copy::set(): incompatible");
                if (need_update(cast->array_, n))
                    SIMD::vec_add(array_, learning_rate, cast->array_, array_,
                                  n);
                return;
            }
        }
    
        // Otherwise, do it via a copy through a known type (here, double)
        double tmp[n];
        mp.copy_to(tmp, tmp + n);
        
        if (need_update(tmp, n))
            SIMD::vec_add(array_, learning_rate, tmp, array_, n);
    }

    virtual void
    update(const Parameter_Value & other,
           const Parameter_Value & learning_rate)
    {
        size_t n = size1_ * size2_;

        const Matrix_Parameter & m_other = other.matrix();
        const Matrix_Parameter & m_learning_rate = learning_rate.matrix();

        if (m_other.size1() != size1_ || m_other.size2() != size2_)
            throw Exception("Matrix_Parameter::update(): incompatible sizes");

        if (m_learning_rate.size1() != size1_
            || m_learning_rate.size2() != size2_)
            throw Exception("Matrix_Parameter::update(): incompatible sizes");

        double tmp1[n];
        m_other.copy_to(tmp1, tmp1 + n);
        
        double tmp2[n];
        m_learning_rate.copy_to(tmp2, tmp2 + n);

        if (need_update(tmp1, n))
            SIMD::vec_add(array_, tmp1, tmp2, array_, n);
    }

    virtual void
    update_sqr(const Parameter_Value & other, double learning_rate)
    {
        size_t n = size1_ * size2_;

        const Matrix_Parameter & mp = other.matrix();

        // Try to do it via a vector operation if possible
        {
            const Matrix_RefT<Underlying> * cast
                = dynamic_cast<const Matrix_RefT<Underlying> *>(&mp);
            if (cast) {
                if ((size1_ != cast->size1_) || (size2_ != cast->size2_))
                    throw Exception("Parameters_Copy::set(): incompatible");
                if (need_update(cast->array_, n))
                    SIMD::vec_add_sqr(array_, learning_rate, cast->array_,
                                      array_, n);
                return;
            }
        }

        {
            const Matrix_RefT<float> * cast
                = dynamic_cast<const Matrix_RefT<float> *>(&mp);
            if (cast) {
                if ((size1_ != cast->size1_) || (size2_ != cast->size2_))
                    throw Exception("Parameters_Copy::set(): incompatible");
                if (need_update(cast->array_, n))
                    SIMD::vec_add_sqr(array_, learning_rate, cast->array_,
                                      array_, n);
                return;
            }
        }

        {
            const Matrix_RefT<double> * cast
                = dynamic_cast<const Matrix_RefT<double> *>(&mp);
            if (cast) {
                if ((size1_ != cast->size1_) || (size2_ != cast->size2_))
                    throw Exception("Parameters_Copy::set(): incompatible");
                if (need_update(cast->array_, n))
                    SIMD::vec_add_sqr(array_, learning_rate, cast->array_,
                                      array_, n);
                return;
            }
        }
    
        // Otherwise, do it via a copy through a known type (here, double)
        double tmp[n];
        mp.copy_to(tmp, tmp + n);
        
        if (need_update(tmp, n))
            SIMD::vec_add_sqr(array_, learning_rate, tmp, array_, n);
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
        if (k != 0.0 && need_update(x, size2_))
            SIMD::vec_add(array_ + (size2_ * row), k, x,
                          array_ + (size2_ * row), size2_);
    }

    virtual void update_row(int row, const double * x, double k = 1.0)
    {
        if (row < 0 || row >= size1_)
            throw Exception("update_row: invalid row");
        if (k != 0.0 && need_update(x, size2_))
            SIMD::vec_add(array_ + (size2_ * row), k, x,
                          array_ + (size2_ * row), size2_);
    }
    
    virtual void update_row_sqr(int row, const float * x, float k = 1.0)
    {
        if (row < 0 || row >= size1_)
            throw Exception("update_row: invalid row");
        if (k != 0.0 && need_update(x, size2_))
            SIMD::vec_add_sqr(array_ + (size2_ * row), k, x,
                              array_ + (size2_ * row), size2_);
    }

    virtual void update_row_sqr(int row, const double * x, double k = 1.0)
    {
        if (row < 0 || row >= size1_)
            throw Exception("update_row: invalid row");
        if (k != 0.0 && need_update(x, size2_))
            SIMD::vec_add_sqr(array_ + (size2_ * row), k, x,
                              array_ + (size2_ * row), size2_);
    }
    
    virtual Matrix_RefT * make_copy() const
    {
        return new Matrix_RefT(*this);
    }

    virtual std::string parameter_info(int index) const
    {
        if (index < 0 || index >= size1_ * size2_)
            throw Exception("size was wrong");
        return format("%s (matrix) row %ld column %ld",
                      name().c_str(), index / size1_, index % size1_);
    }

protected:
    Underlying * array_;
    size_t size1_, size2_;
    template<typename U> friend class Matrix_RefT;
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
void
Parameters_Copy<Float>::
init(const Parameters & other,
     bool copy)
{
    if (values.size() != other.parameter_count())
        throw Exception("Parameters_Copy::init(): values wasn't sized "
                        "properly");

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

        if (copy) add(i, it->compatible_copy(current, current + n));
        else add(i, it->compatible_ref(current, current + n));

        current += n;
    }

    if (current != finish)
        throw Exception("parameters_copy(): wrong length");
}

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
    init(other, false /* copy */);
}

template<class Float>
Parameters_Copy<Float>::
Parameters_Copy(const Parameters_Copy & other, double fill_with)
    : Parameters(other.name())
{
    values.resize(other.values.size(), fill_with);
    init(other, false /* copy */);
}

template<class Float>
Parameters_Copy<Float> &
Parameters_Copy<Float>::
operator = (const Parameters_Copy & other)
{
    Parameters_Copy new_me(other);
    swap(new_me);
    return *this;
}

template<class Float>
Parameters_Copy<Float>::
Parameters_Copy(const Parameters & other)
    : Parameters(other.name())
{
    values.resize(other.parameter_count());
    init(other, true /* copy */);
}

template<class Float>
Parameters_Copy<Float>::
Parameters_Copy(const Parameters & other, double fill_with)
    : Parameters(other.name())
{
    values.resize(other.parameter_count(), fill_with);
    init(other, false /* copy */);
}

template<class Float>
Parameters_Copy<Float>::
Parameters_Copy(const Layer & layer)
    : Parameters(layer.name())
{
    values.resize(layer.parameters().parameter_count());
    init(layer.parameters(), true /* copy */);
}

template<class Float>
Parameters_Copy<Float>::
Parameters_Copy(const Layer & layer, double fill_with)
    : Parameters(layer.name())
{
    values.resize(layer.parameters().parameter_count(), fill_with);
    init(layer.parameters(), false /* copy */);
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

template<class Float>
void
Parameters_Copy<Float>::
update(const Parameter_Value & other, double learning_rate)
{
    // Try to do it via a vector operation if possible
    {
        const Parameters_Copy<Float> * cast
            = dynamic_cast<const Parameters_Copy<Float> *>(&other);
        if (cast) {
            if (cast->values.size() != values.size())
                throw Exception("Parameters_Copy::set(): incompatible");
            if (need_update(&cast->values[0], values.size()))
                SIMD::vec_add(&values[0], learning_rate, &cast->values[0],
                              &values[0], values.size());
            return;
        }
    }

    {
        const Parameters_Copy<float> * cast
            = dynamic_cast<const Parameters_Copy<float> *>(&other);
        if (cast) {
            if (cast->values.size() != values.size())
                throw Exception("Parameters_Copy::set(): incompatible");
            if (need_update(&cast->values[0], values.size()))
                SIMD::vec_add(&values[0], learning_rate, &cast->values[0],
                              &values[0], values.size());
            return;
        }
    }

    {
        const Parameters_Copy<double> * cast
            = dynamic_cast<const Parameters_Copy<double> *>(&other);
        if (cast) {
            if (cast->values.size() != values.size())
                throw Exception("Parameters_Copy::set(): incompatible");
            if (need_update(&cast->values[0], values.size()))
                SIMD::vec_add(&values[0], learning_rate, &cast->values[0],
                              &values[0], values.size());
            return;
        }
    }

    // Otherwise, do it structurally
    Parameters::update(other, learning_rate);
}

template<typename Float1, typename Float2, typename Float>
bool try_update(Parameters_Copy<Float> & params,
                const Parameter_Value & other,
                const Parameter_Value & learning_rate)
{
    size_t nvals = params.values.size();

    if (const Parameters_Copy<Float1> * cast
        = dynamic_cast<const Parameters_Copy<Float1> *>(&other)) {
        if (const Parameters_Copy<Float2> * cast2
            = dynamic_cast<const Parameters_Copy<Float2> *>(&learning_rate)) {
            if (cast->values.size() != nvals)
                throw Exception("Parameters_Copy::update(): incompatible");
            if (cast2->values.size() != nvals)
                throw Exception("Parameters_Copy::update(): incompatible lr");
            if (need_update(&cast->values[0], nvals))
                SIMD::vec_add(&params.values[0], &cast2->values[0],
                              &cast->values[0], &params.values[0], nvals);
            return true;
        }
    }
    return false;
}

template<class Float>
void
Parameters_Copy<Float>::
update(const Parameter_Value & other,
       const Parameter_Value & learning_rate)
{
    if (name() != other.name()
        || name() != learning_rate.name())
        throw Exception("Parameters_Copy::update(): incompatible names");

    // Try to do it via a vector operation if possible
    if (try_update<double, double>(*this, other, learning_rate)) return;
    if (try_update<float, double>(*this, other, learning_rate)) return;
    if (try_update<double, float>(*this, other, learning_rate)) return;
    if (try_update<float, float>(*this, other, learning_rate)) return;

    // Otherwise, do it structurally
    Parameters::update(other, learning_rate);
}

template<class Float>
void
Parameters_Copy<Float>::
update_sqr(const Parameter_Value & other, double learning_rate)
{
    // Try to do it via a vector operation if possible
    {
        const Parameters_Copy<Float> * cast
            = dynamic_cast<const Parameters_Copy<Float> *>(&other);
        if (cast) {
            if (cast->values.size() != values.size())
                throw Exception("Parameters_Copy::set(): incompatible");
            if (need_update(&cast->values[0], values.size()))
                SIMD::vec_add_sqr(&values[0], learning_rate, &cast->values[0],
                                  &values[0], values.size());
            return;
        }
    }

    {
        const Parameters_Copy<float> * cast
            = dynamic_cast<const Parameters_Copy<float> *>(&other);
        if (cast) {
            if (cast->values.size() != values.size())
                throw Exception("Parameters_Copy::set(): incompatible");
            if (need_update(&cast->values[0], values.size()))
                SIMD::vec_add_sqr(&values[0], learning_rate, &cast->values[0],
                                  &values[0], values.size());
            return;
        }
    }

    {
        const Parameters_Copy<double> * cast
            = dynamic_cast<const Parameters_Copy<double> *>(&other);
        if (cast) {
            if (cast->values.size() != values.size())
                throw Exception("Parameters_Copy::set(): incompatible");
            if (need_update(&cast->values[0], values.size()))
                SIMD::vec_add_sqr(&values[0], learning_rate, &cast->values[0],
                                  &values[0], values.size());
            return;
        }
    }

    // Otherwise, do it structurally
    Parameters::update_sqr(other, learning_rate);
}

template<class Float>
Parameters_Copy<Float> *
Parameters_Copy<Float>::
make_copy() const
{
    return new Parameters_Copy(*this);
}

} // namespace ML


#endif /* __jml__neural__parameters_impl_h__ */
