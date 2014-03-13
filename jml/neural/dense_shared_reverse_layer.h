/* dense_shared_reverse_layer.h                                    -*- C++ -*-
   Jeremy Barnes, 9 November 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

   A reverse layer that shares connection weights with a dense layer.
*/

#ifndef __jml__neural__dense_shared_reverse_layer_h__
#define __jml__neural__dense_shared_reverse_layer_h__

#include "dense_layer.h"

namespace ML {


/*****************************************************************************/
/* DENSE_SHARED_REVERSE_LAYER                                                */
/*****************************************************************************/

/** A reversed layer that shares the weights of a forward layer. */

template<typename Float>
struct Dense_Shared_Reverse_Layer : public Layer {

    Dense_Shared_Reverse_Layer();

    /** Initialize to zero */
    Dense_Shared_Reverse_Layer(const std::string & name,
                               Dense_Layer<Float>..
                size_t inputs, size_t units,
                Transfer_Function_Type transfer_function,
                Missing_Values missing_values);

    /** Initialize with random values */
    Dense_Shared_Reverse_Layer(const std::string & name,
                size_t ninputs, size_t units,
                Transfer_Function_Type transfer_function,
                Missing_Values missing_values,
                Thread_Context & thread_context,
                float limit = -1.0);

    Dense_Shared_Reverse_Layer(const Dense_Shared_Reverse_Layer & other);

    Dense_Shared_Reverse_Layer & operator = (const Dense_Shared_Reverse_Layer & other);

    void swap(Dense_Shared_Reverse_Layer & other);

    /// Bias for the reverse direction
    distribution<Float> ibias;

    /// Scaling factors for the reverse direction
    distribution<Float> iscales;
    distribution<Float> hscales;

    /// The forward layer that we share with
    Dense_Layer<Float> * forward;


    /*************************************************************************/
    /* APPLY                                                                 */
    /*************************************************************************/


    virtual void apply(const float * input, float * output) const;
    virtual void apply(const double * input, double * output) const;

    using Layer::apply;


    /*************************************************************************/
    /* ACTIVATION                                                            */
    /*************************************************************************/

    /* Calculate the activation function for the output neurons */

    virtual void activation(const float * input,
                            float * activation) const;

    virtual void activation(const double * input,
                            double * activation) const;

    distribution<float> activation(const distribution<float> & input) const;
    distribution<double> activation(const distribution<double> & input) const;


    /*************************************************************************/
    /* FPROP                                                                 */
    /*************************************************************************/

    /** Return the amount of space necessary to save temporary results for the
        forward prop.  There will be an array of the given precision (double
        or single) provided.

        Default implementation returns outputs().
    */

    virtual size_t fprop_temporary_space_required() const;

    /** These functions perform a forward propagation.  They also save whatever
        information is necessary to perform an efficient backprop at a later
        period in time.

        Default implementation calls apply() and saves the outputs only in the
        temporary space.
    */
    virtual distribution<float>
    fprop(const distribution<float> & inputs,
          float * temp_space, size_t temp_space_size) const;

    virtual distribution<double>
    fprop(const distribution<double> & inputs,
          double * temp_space, size_t temp_space_size) const;
    
               

    /*************************************************************************/
    /* BPROP                                                                 */
    /*************************************************************************/

    /** Perform a back propagation.  Given the derivative of the error with
        respect to each of the errors, they compute the gradient of the
        parameter space.
    */

    virtual void bprop(const distribution<float> & output_errors,
                       float * temp_space, size_t temp_space_size,
                       Parameters & gradient,
                       distribution<float> & input_errors,
                       double example_weight,
                       bool calculate_input_errors) const;

    virtual void bprop(const distribution<double> & output_errors,
                       double * temp_space, size_t temp_space_size,
                       Parameters & gradient,
                       distribution<double> & input_errors,
                       double example_weight,
                       bool calculate_input_errors) const;


    /** Add in our parameters to the params object. */
    virtual void add_parameters(Parameters & params);

    /** Dump as ASCII.  This will be big. */
    virtual std::string print() const;
    
    /** Return the name of the type */
    virtual std::string class_id() const;

    virtual void serialize(DB::Store_Writer & store) const;
    virtual void reconstitute(DB::Store_Reader & store);

    /** Fill with random weights. */
    virtual void random_fill(float limit, Thread_Context & context);

    /** Fill with zero values */
    virtual void zero_fill();

    /** Return the number of parameters (degrees of freedom) for the
        layer. */
    virtual size_t parameter_count() const;

    virtual std::pair<float, float> targets(float maximum) const;

    virtual Dense_Shared_Reverse_Layer *
    make_copy() const { return new Dense_Shared_Reverse_Layer(*this); }

    virtual Dense_Shared_Reverse_Layer *
    deep_copy() const { return new Dense_Shared_Reverse_Layer(*this); }

    virtual void validate() const;

    virtual bool equal_impl(const Layer & other) const;

    // For testing purposes
    bool operator == (const Dense_Shared_Reverse_Layer & other) const;
    bool operator != (const Dense_Shared_Reverse_Layer & other) const
    {
        return ! operator == (other);
    }

private:
    struct RegisterMe;
    static RegisterMe register_me;
};

JML_IMPL_SERIALIZE_RECONSTITUTE_TEMPLATE(typename Float, Dense_Shared_ReverseLayer<Float>);


extern template class Dense_Shared_Reverse_Layer<float>;
extern template class Dense_Shared_Reverse_Layer<double>;

} // namespace ML


#endif /* __jml__neural__dense_shared_reverse_layer_h__ */
