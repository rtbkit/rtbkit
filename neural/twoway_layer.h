/* twoway_layer.h                                                  -*- C++ -*-
   Jeremy Barnes, 4 November 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

   Two way layer, that can generate its data.
*/

#ifndef __jml__neural__twoway_layer_h__
#define __jml__neural__twoway_layer_h__


#include "dense_layer.h"
#include "layer_stack.h"
#include "auto_encoder.h"

namespace ML {


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

struct Twoway_Layer : public Auto_Encoder {
    typedef Auto_Encoder Base;
    typedef float Float;
    typedef Dense_Layer<Float> Forward;

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

    Twoway_Layer(const Twoway_Layer & other);

    Twoway_Layer & operator = (const Twoway_Layer & other);

    void swap(Twoway_Layer & other);


    /*************************************************************************/
    /* FORWARD DIRECTION                                                     */
    /*************************************************************************/

    /// Layer for the forward direction
    Forward forward;

    /** All of these methods simply forward to the ones in the forward
        layer. */

    virtual std::pair<float, float> targets(float maximum) const;

    virtual bool supports_missing_inputs() const;

    using Layer::apply;

    virtual void apply(const float * input, float * output) const;
    virtual void apply(const double * input, double * output) const;


    virtual size_t fprop_temporary_space_required() const;

    using Layer::fprop;

    virtual void
    fprop(const float * inputs,
          float * temp_space, size_t temp_space_size,
          float * outputs) const;

    virtual void
    fprop(const double * inputs,
          double * temp_space, size_t temp_space_size,
          double * outputs) const;

    using Layer::bprop;
    
    virtual void bprop(const float * inputs,
                       const float * outputs,
                       const float * temp_space, size_t temp_space_size,
                       const float * output_errors,
                       float * input_errors,
                       Parameters & gradient,
                       double example_weight) const;

    virtual void bprop(const double * inputs,
                       const double * outputs,
                       const double * temp_space, size_t temp_space_size,
                       const double * output_errors,
                       double * input_errors,
                       Parameters & gradient,
                       double example_weight) const;

    using Layer::bbprop;

    virtual void bbprop(const float * inputs,
                        const float * outputs,
                        const float * temp_space, size_t temp_space_size,
                        const float * output_errors,
                        const float * d2output_errors,
                        float * input_errors,
                        float * d2input_errors,
                        Parameters & gradient,
                        Parameters * dgradient,
                        double example_weight) const;
 
    virtual void bbprop(const double * inputs,
                        const double * outputs,
                        const double * temp_space, size_t temp_space_size,
                        const double * output_errors,
                        const double * d2output_errors,
                        double * input_errors,
                        double * d2input_errors,
                        Parameters & gradient,
                        Parameters * dgradient,
                        double example_weight) const;
    

    /*************************************************************************/
    /* INVERSE DIRECTION                                                     */
    /*************************************************************************/

    /// Bias for the reverse direction
    distribution<Float> ibias;

    /// Scaling factors for the reverse direction
    distribution<Float> iscales;
    distribution<Float> oscales;

    virtual std::pair<float, float> itargets(float maximum) const;

    /** When running in the inverse direction, are missing outputs (NaN values)
        supported? */
        
    virtual bool supports_missing_outputs() const;
    
    using Auto_Encoder::iapply;

    template<typename F>
    void iapply(const F * outputs, F * inputs) const;

    virtual void iapply(const float * outputs, float * inputs) const;
    virtual void iapply(const double * outputs, double * inputs) const;


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

    using Auto_Encoder::ifprop;

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
    
    using Auto_Encoder::ibprop;

    virtual void ibbprop(const float * outputs,
                         const float * inputs,
                         const float * temp_space, size_t temp_space_size,
                         const float * input_errors,
                         const float * d2input_errors,
                         float * output_errors,
                         float * d2output_errors,
                         Parameters & gradient,
                         Parameters * dgradient,
                         double example_weight) const;
 
    virtual void ibbprop(const double * outputs,
                         const double * inputs,
                         const double * temp_space, size_t temp_space_size,
                         const double * input_errors,
                         const double * d2input_errors,
                         double * output_errors,
                         double * d2output_errors,
                         Parameters & gradient,
                         Parameters * dgradient,
                         double example_weight) const;

    template<typename F>
    void ibbprop(const F * outputs,
                 const F * inputs,
                 const F * temp_space, size_t temp_space_size,
                 const F * input_errors,
                 const F * d2input_errors,
                 F * output_errors,
                 F * d2output_errors,
                 Parameters & gradient,
                 Parameters * dgradient,
                 double example_weight) const;

    /** Perform a back propagation.  Given the derivative of the error with
        respect to each of the errors, they compute the gradient of the
        parameter space.
    */
    virtual void rbprop(const float * inputs,
                        const float * reconstruction,
                        const float * temp_space,
                        size_t temp_space_size,
                        const float * reconstruction_errors,
                        float * input_errors_out,
                        Parameters & gradient,
                        double example_weight) const;
    
    /** \copydoc rbprop */
    virtual void rbprop(const double * inputs,
                        const double * reconstruction,
                        const double * temp_space,
                        size_t temp_space_size,
                        const double * reconstruction_errors,
                        double * input_errors_out,
                        Parameters & gradient,
                        double example_weight) const;

    template<typename F>
    void rbprop(const F * inputs,
                const F * reconstruction,
                const F * temp_space,
                size_t temp_space_size,
                const F * reconstruction_errors,
                F * input_errors_out,
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
