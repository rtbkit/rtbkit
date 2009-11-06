/* dense_layer.h                                                   -*- C++ -*-
   Jeremy Barnes, 2 November 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

*/

#ifndef __neural__dense_layer_h__
#define __neural__dense_layer_h__


#include "layer.h"
#include "transfer_function.h"
#include "utils/enum_info.h"
#include "layer_stack.h"


namespace ML {


/*****************************************************************************/
/* MISSING_VALUES                                                            */
/*****************************************************************************/

enum Missing_Values {
    MV_NONE,  ///< Missing values are not accepted
    MV_ZERO,  ///< Missing values are replaced with a zero
    MV_INPUT, ///< Missing values are replaced with a value per input
    MV_DENSE  ///< Missing inputs use a different activation matrix
};

BYTE_PERSISTENT_ENUM_DECL(Missing_Values);

std::string print(Missing_Values mv);

std::ostream & operator << (std::ostream & stream, Missing_Values mv);


/*****************************************************************************/
/* DENSE_LAYER                                                               */
/*****************************************************************************/

/** A simple one way layer with dense connections. */

template<typename Float>
struct Dense_Layer : public Layer {

    Dense_Layer();

    /** Initialize to zero */
    Dense_Layer(const std::string & name,
                size_t inputs, size_t units,
                Transfer_Function_Type transfer_function,
                Missing_Values missing_values);

    /** Initialize with random values */
    Dense_Layer(const std::string & name,
                size_t ninputs, size_t units,
                Transfer_Function_Type transfer_function,
                Missing_Values missing_values,
                Thread_Context & thread_context,
                float limit = -1.0);

    /// Transfer function for the output
    boost::shared_ptr<const Transfer_Function> transfer_function;

    /// How to treat missing values in the input
    Missing_Values missing_values;
        
    /// Network parameters: activation weights
    boost::multi_array<Float, 2> weights;

    /// Network parameters: bias
    distribution<Float> bias;

    /// missing_values == MV_INPUT: Input value to use instead of missing
    distribution<Float> missing_replacements;

    /// missing_values == MV_DENSE: Activation matrix to use when missing
    boost::multi_array<Float, 2> missing_activations;


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
    virtual boost::shared_ptr<Parameters> parameters();

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

    virtual Dense_Layer * make_copy() const { return new Dense_Layer(*this); }

    virtual Dense_Layer * deep_copy() const { return new Dense_Layer(*this); }

    virtual void validate() const;

    // For testing purposes
    bool operator == (const Dense_Layer & other) const;

private:
    struct RegisterMe;
    static RegisterMe register_me;
};

JML_IMPL_SERIALIZE_RECONSTITUTE_TEMPLATE(typename Float, Dense_Layer<Float>);


extern template class Dense_Layer<float>;
extern template class Dense_Layer<double>;

extern template class Layer_Stack<Dense_Layer<float> >;
extern template class Layer_Stack<Dense_Layer<double> >;

} // namespace ML

DECLARE_ENUM_INFO(ML::Missing_Values, 4);

#endif /* __neural__dense_layer_h__ */
