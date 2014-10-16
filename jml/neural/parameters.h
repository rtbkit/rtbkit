/* parameters.h                                                    -*- C++ -*-
   Jeremy Barnes, 2 November 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

   Desciption of parameters.  Used to allow polymorphic updates of
   parameters.
*/

#ifndef __neural__parameters_h__
#define __neural__parameters_h__

#include <vector>
#include <boost/multi_array.hpp>
#include <boost/ptr_container/ptr_vector.hpp>
#include "jml/utils/hash_map.h"
#include "jml/boosting/thread_context.h"
#include "jml/db/persistent_fwd.h"
#include "jml/stats/distribution.h"
#include "jml/arch/simd_vector.h"

namespace ML {


class Layer;
class Parameters;
class Vector_Parameter;
class Matrix_Parameter;


/*****************************************************************************/
/* LOCKING_POLICY                                                            */
/*****************************************************************************/

/** Describes how the locking is performed on the object when multiple threads
    can update.  They have different tradeoffs for thread occupancy versus
    efficiency.

    In a single threaded context, no locking is needed.
*/
enum Locking_Policy {
    LP_NONE,    ///< No locking (single threaded)
    LP_ATOMIC,  ///< Use atomic instructions
    LP_COARSE,  ///< Use one (coarse grained) lock
    LP_FINE     ///< Use fine grained locking per row (spinlock)
};


/*****************************************************************************/
/* PARAMETER_VALUE                                                           */
/*****************************************************************************/

struct Parameter_Value {
    Parameter_Value(const std::string & name)
        : name_(name)
    {
    }

    virtual ~Parameter_Value() {}

    virtual size_t parameter_count() const = 0;

    virtual void copy_to(float * where, float * limit) const = 0;
    virtual void copy_to(double * where, double * limit) const = 0;
    

    /** Create a compatible parameters object, that refers to the data range
        given, not the current range.  The given range is not modified.  */
    virtual Parameter_Value *
    compatible_ref(float * first, float * last) const = 0;
    virtual Parameter_Value *
    compatible_ref(double * first, double * last) const = 0;

    /** Create a compatible parameters object, that refers to the data range
        given, not the current range.  The given range is initialized with
        the current values via copy_to.  */
    virtual Parameter_Value *
    compatible_copy(float * first, float * last) const;
    virtual Parameter_Value *
    compatible_copy(double * first, double * last) const;

    /** Set our values from another set of values */
    virtual void set(const Parameter_Value & other) = 0;

    /** Fill with the given value */
    virtual void fill(double value) = 0;

    virtual void update(const Parameter_Value & other,
                        double learning_rate) = 0;
    virtual void update(const Parameter_Value & other,
                        const Parameter_Value & learning_rate) = 0;

    virtual void update_sqr(const Parameter_Value & other,
                            double learning_rate) = 0;

    std::string name() const { return name_; }

    virtual Parameters & parameters();
    virtual const Parameters & parameters() const;

    virtual Vector_Parameter & vector();
    virtual const Vector_Parameter & vector() const;

    virtual Matrix_Parameter & matrix();
    virtual const Matrix_Parameter & matrix() const;

    virtual Parameter_Value * make_copy() const = 0;

    // Change the name.  This might confuse the parent; should mostly be
    // used as part of the implementation.
    virtual void set_name(const std::string & name);

    /** Describe what the parameter is (name, type, etc) */
    virtual std::string parameter_info(int index) const = 0;

protected:
    std::string name_;
    void swap(Parameter_Value & other);
};


/*****************************************************************************/
/* VECTOR_PARAMETER                                                          */
/*****************************************************************************/

struct Vector_Parameter : public Parameter_Value {
    Vector_Parameter(const std::string & name)
        : Parameter_Value(name)
    {
    }

    // Update: y += kx
    virtual void update(const float * x, float k) = 0;
    virtual void update(const double * x, double k) = 0;

    // Update: y += k x^2 (useful for 2nd derivs)
    virtual void update_sqr(const float * x, float k) = 0;
    virtual void update_sqr(const double * x, double k) = 0;

    void update(const distribution<float> & dist, float k);
    void update(const distribution<double> & dist, double k);

    using Parameter_Value::update_sqr;

    void update_sqr(const distribution<float> & dist, float k);
    void update_sqr(const distribution<double> & dist, double k);

    using Parameter_Value::update;

    virtual void update_element(int element, float update_by) = 0;
    virtual void update_element(int element, double update_by) = 0;

    virtual Vector_Parameter & vector() { return *this; }
    virtual const Vector_Parameter & vector() const { return *this; }

    virtual size_t size() const = 0;
};


/*****************************************************************************/
/* MATRIX_PARAMETER                                                          */
/*****************************************************************************/

struct Matrix_Parameter : public Parameter_Value {
    Matrix_Parameter(const std::string & name)
        : Parameter_Value(name)
    {
    }

    virtual void update_row(int row, const float * x, float k = 1.0) = 0;
    virtual void update_row(int row, const double * x, double k = 1.0) = 0;

    virtual void update_row_sqr(int row, const float * x, float k = 1.0) = 0;
    virtual void update_row_sqr(int row, const double * x, double k = 1.0) = 0;

    void update_row(int row, const distribution<float> & x, float k);
    void update_row(int row, const distribution<double> & x, float k);

    void update_row_sqr(int row, const distribution<float> & x, float k);
    void update_row_sqr(int row, const distribution<double> & x, float k);

    virtual Matrix_Parameter & matrix() { return *this; }
    virtual const Matrix_Parameter & matrix() const { return *this; }

    virtual size_t size1() const = 0;
    virtual size_t size2() const = 0;
};


/*****************************************************************************/
/* PARAMETERS                                                                */
/*****************************************************************************/

struct Parameters : public Parameter_Value {

    /** Add a vector of values to the parameters */

    Parameters & add(int index, Parameter_Value * param);

    template<class Float>
    Parameters &
    add(int index,
        const std::string & name,
        std::vector<Float> & values);

    Parameters &
    add(int index,
        const std::string & name,
        std::vector<float> & values);
    Parameters &
    add(int index,
        const std::string & name,
        std::vector<double> & values);
    
    /** Add a matrix of values to the parameters */
    template<class Float>
    Parameters &
    add(int index,
        const std::string & name,
        boost::multi_array<Float, 2> & values);

    Parameters &
    add(int index,
        const std::string & name,
        boost::multi_array<float, 2> & values);
    Parameters &
    add(int index,
        const std::string & name,
        boost::multi_array<double, 2> & values);

    size_t parameter_count() const;

    void serialize(DB::Store_Writer & store) const;

    /** Reconstitutes the object, not the parameters.  To reconstitute the
        parameters, first reconstitute a new object and then assign the
        new version. */
    void reconstitute(DB::Store_Reader & store);

    virtual void fill(double value);

    virtual void update(const Parameter_Value & other, double learning_rate);
    virtual void update(const Parameter_Value & other,
                        const Parameter_Value & learning_rate);
    virtual void update_sqr(const Parameter_Value & other,
                            double learning_rate);

    virtual Parameters & subparams(int index, const std::string & name);
    virtual const Parameters &
    subparams(int index, const std::string & name) const;

    size_t subparam_count() const { return params.size(); }

    Vector_Parameter & vector(int index, const std::string & name);
    const Vector_Parameter &
    vector(int index, const std::string & name) const;
    using Parameter_Value::vector;


    Matrix_Parameter & matrix(int index, const std::string & name);
    const Matrix_Parameter &
    matrix(int index, const std::string & name) const;
    using Parameter_Value::matrix;


    /** Concrete copy_to implementations */
    template<typename F>
    void copy_to(F * where, F * limit) const;

    virtual void copy_to(float * where, float * limit) const;
    virtual void copy_to(double * where, double * limit) const;

    /** Create a compatible parameters object, that refers to the data range
        given, not the current range.  The given range is not modified.  */
    template<typename F>
    Parameters *
    compatible_ref(F * first, F * last) const;

    virtual Parameters *
    compatible_ref(float * first, float * last) const;
    virtual Parameters *
    compatible_ref(double * first, double * last) const;

    /** Create a compatible parameters object, that refers to the data range
        given, not the current range.  The given range is initialized with
        the current values via copy_to.  */
    template<typename F>
    Parameters *
    compatible_copy(F * first, F * last) const;

    virtual Parameters *
    compatible_copy(float * first, float * last) const;

    virtual Parameters *
    compatible_copy(double * first, double * last) const;

    virtual Parameters & parameters() { return *this; }
    virtual const Parameters & parameters() const { return *this; }

    Parameters * make_copy() const;

    /** Remove all parameter references from this object.  Doesn't actually
        modify any of the parameter values. */
    void clear();

    /** Set these parameters from another parameters object. */
    virtual void set(const Parameter_Value & other);

    /** Describe what the parameter is (name, type, etc) */
    virtual std::string parameter_info(int index) const;

protected:
    Parameters(const std::string & name);
    Parameters(const Parameters & other);

    Parameters & operator = (const Parameters & other);

    void swap(Parameters & other);

    std::hash_map<std::string, int> by_name;
    typedef boost::ptr_vector<Parameter_Value> Params;
    Params params;

    template<class Float> friend class Parameters_Copy;
};


/*****************************************************************************/
/* PARAMETERS_REF                                                            */
/*****************************************************************************/

/** Parameters that are stored somewhere else but referenced here. */

struct Parameters_Ref : public Parameters {
    Parameters_Ref();
    Parameters_Ref(const std::string & name);

    virtual Parameters_Ref * make_copy() const;
};


/*****************************************************************************/
/* PARAMETERS_COPY                                                           */
/*****************************************************************************/

/** Storage of a value for each parameter, in the given type. */

template<class Float>
struct Parameters_Copy : public Parameters {
    Parameters_Copy();
    
    Parameters_Copy(const Parameters & other);
    Parameters_Copy(const Parameters_Copy & other);

    Parameters_Copy(const Parameters & other, double fill_with);
    Parameters_Copy(const Parameters_Copy & other, double fill_with);

    Parameters_Copy & operator = (const Parameters_Copy & other);

    Parameters_Copy(const Layer & layer);
    Parameters_Copy(const Layer & layer, double fill_with);

    void swap(Parameters_Copy & other);

    virtual Parameters_Copy * make_copy() const;

    /** Concrete copy_to implementations */
    virtual void copy_to(float * where, float * limit) const;
    virtual void copy_to(double * where, double * limit) const;

    /** Set these parameters from another parameters object. */
    virtual void set(const Parameter_Value & other);

    virtual void fill(double value);

    /** For all of our parameters, apply param += learning_rate * other.param */
    virtual void update(const Parameter_Value & other, double learning_rate);
    virtual void update(const Parameter_Value & other,
                        const Parameter_Value & learning_rate);
    virtual void update_sqr(const Parameter_Value & other,
                            double learning_rate);

    // The actual values, stored contiguously for efficiency.  The client
    // should not resize them, but is free to access them as an anonymous
    // parameter vector.
    distribution<Float> values;

private:
    /** Initialize from the other parameters, into a set of values which
        should already be the right size.  If copy is true, the parameter
        values from other will be copied into values; otherwise values is
        left untouched. */
    void init(const Parameters & other, bool copy);
};


extern template class Parameters_Copy<float>;
extern template class Parameters_Copy<double>;


} // namespace ML

#endif /* __neural__parameters_h__ */
