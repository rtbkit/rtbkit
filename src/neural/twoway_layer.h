/* twoway_layer.h                                                  -*- C++ -*-
   Jeremy Barnes, 4 November 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

   Two way layer, that can generate its data.
*/

#ifndef __jml__neural__twoway_layer_h__
#define __jml__neural__twoway_layer_h__


#include "dense_layer.h"
#include "layer_stack.h"


namespace ML {

template<class LayerT> class Layer_Stack;



/*****************************************************************************/
/* TWOWAY_LAYER                                                              */
/*****************************************************************************/

/** A perceptron layer that has both a forward and a reverse direction.  It's
    both a discriminative model (in the forward direction) and a generative
    model (in the reverse direction).

    This layer shares the transfer function and activation weights between
    the forward and reverse directions.  However, there are some adjstments
    made:

    forward:   o = f( Wi + b )
    backwards: i = f( d WT o e + c )

    WT is W transpose.  c is a bias vector with separate weights.  d and e
    are diagonal scaling matrices on the left and right of WT; they can
    compensate for W having have non-unitary singular values.

    Note that in the inverse direction, no missing values are accepted.
*/

struct Twoway_Layer : public Dense_Layer<float> {
    typedef Dense_Layer<float> Base;

    Twoway_Layer();

    Twoway_Layer(const std::string & name,
                 size_t inputs, size_t outputs,
                 Transfer_Function_Type transfer,
                 Missing_Values missing_values,
                 Thread_Context & context,
                 float limit = -1.0);

    Twoway_Layer(const std::string & name,
                 size_t inputs, size_t outputs,
                 Transfer_Function_Type transfer,
                 Missing_Values missing_values);

    /*************************************************************************/
    /* INVERSE DIRECTION                                                     */
    /*************************************************************************/

    /// Bias for the reverse direction
    distribution<float> ibias;

    /// Scaling factors for the reverse direction
    distribution<float> iscales;
    distribution<float> oscales;
    
    template<typename F>
    distribution<F> iapply(const distribution<F> & output) const;

    virtual distribution<double>
    iapply(const distribution<double> & output) const;
    virtual distribution<float>
    iapply(const distribution<float> & output) const;

    /** Return the amount of space necessary to save temporary results for the
        forward prop.  There will be an array of the given precision (double
        or single) provided.

        Default implementation returns outputs().
    */

    virtual size_t ifprop_temporary_space_required() const;

    /** These functions perform a forward propagation.  They also save whatever
        information is necessary to perform an efficient backprop at a later
        period in time.

        Default implementation calls apply() and saves the outputs only in the
        temporary space.
    */
    template<typename F>
    void
    ifprop(const F * inputs,
           F * temp_space, size_t temp_space_size,
           F * outputs) const;

    virtual void
    ifprop(const float * outputs,
           float * temp_space, size_t temp_space_size,
           float * inputs) const;

    /** \copydoc ifprop */
    virtual void
    ifprop(const double * outputs,
           double * temp_space, size_t temp_space_size,
           double * inputs) const;

    
    /** Perform a back propagation.  Given the derivative of the error with
        respect to each of the errors, they compute the gradient of the
        parameter space.
    */

    template<typename F>
    void ibprop(const F * outputs,
                const F * inputs,
                const F * temp_space, size_t temp_space_size,
                const F * input_errors,
                F * output_errors,
                Parameters & gradient,
                double example_weight) const;

    virtual void ibprop(const float * outputs,
                        const float * inputs,
                        const float * temp_space, size_t temp_space_size,
                        const float * input_errors,
                        float * output_errors,
                        Parameters & gradient,
                        double example_weight) const;
    
    /** \copydoc ibprop */
    virtual void ibprop(const double * outputs,
                        const double * inputs,
                        const double * temp_space, size_t temp_space_size,
                        const double * input_errors,
                        double * output_errors,
                        Parameters & gradient,
                        double example_weight) const;
    

    /*************************************************************************/
    /* RECONSTRUCTION                                                        */
    /*************************************************************************/

    virtual distribution<float>
    reconstruct(const distribution<float> & input) const;
    virtual distribution<double>
    reconstruct(const distribution<double> & input) const;

    /** Return the amount of space necessary to save temporary results for the
        forward reconstruction.  There will be an array of the given precision
        (double or single) provided.
    */

    virtual size_t rfprop_temporary_space_required() const;
    
    /** These functions perform a forward reconstruction.  They also save
        whatever information is necessary to perform an efficient backprop
        of the reconstruction error at a later period in time.

        Returns the reconstructed input.
    */
    template<typename F>
    void
    rfprop(const F * inputs,
           F * temp_space, size_t temp_space_size,
           F * reconstruction) const;

    /** \copydoc rfprop */
    virtual void
    rfprop(const float * inputs,
           float * temp_space, size_t temp_space_size,
           float * reconstruction) const;

    /** \copydoc rfprop */
    virtual void
    rfprop(const double * inputs,
           double * temp_space, size_t temp_space_size,
           double * reconstruction) const;
    
    /** Perform a back propagation.  Given the derivative of the error with
        respect to each of the errors, they compute the gradient of the
        parameter space.
    */
    template<typename F>
    void rbprop(const F * inputs,
                const F * reconstruction,
                const F * temp_space,
                size_t temp_space_size,
                const F * reconstruction_errors,
                Parameters & gradient,
                double example_weight) const;
    
    /** \copydoc rbprop */
    virtual void rbprop(const float * inputs,
                        const float * reconstruction,
                        const float * temp_space,
                        size_t temp_space_size,
                        const float * reconstruction_errors,
                        Parameters & gradient,
                        double example_weight) const;
    
    /** \copydoc rbprop */
    virtual void rbprop(const double * inputs,
                        const double * reconstruction,
                        const double * temp_space,
                        size_t temp_space_size,
                        const double * reconstruction_errors,
                        Parameters & gradient,
                        double example_weight) const;

    /** Dump as ASCII.  This will be big. */
    virtual std::string print() const;
    
    /** Return the name of the type */
    virtual std::string class_id() const;

    /** Check that all parameters are reasonable and invariants are met.
        Will throw an exception if there is a problem. */
    virtual void validate() const;

    virtual bool equal_impl(const Layer & other) const;

    /** Add all of our parameters to the given parameters object. */
    virtual void add_parameters(Parameters & params);

    /** Return the number of parameters (degrees of freedom) for the
        layer. */
    virtual size_t parameter_count() const;

    virtual void serialize(DB::Store_Writer & store) const;
    virtual void reconstitute(DB::Store_Reader & store);

    virtual void random_fill(float limit, Thread_Context & context);

    virtual void zero_fill();

    virtual Twoway_Layer * make_copy() const { return new Twoway_Layer(*this); }

    virtual Twoway_Layer * deep_copy() const { return new Twoway_Layer(*this); }

    bool operator == (const Twoway_Layer & other) const;
    bool operator != (const Twoway_Layer & other) const
    {
        return ! operator == (other);
    }
};

IMPL_SERIALIZE_RECONSTITUTE(Twoway_Layer);

extern template class Layer_Stack<Twoway_Layer>;

} // namespace ML


#endif /* __jml__neural__twoway_layer_h__ */
