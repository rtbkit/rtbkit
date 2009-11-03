/* dense_layer.h                                                   -*- C++ -*-
   Jeremy Barnes, 2 November 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

*/

#ifndef __neural__dense_layer_h__
#define __neural__dense_layer_h__

#include "layer.h"

namespace ML {

/*****************************************************************************/
/* DENSE_LAYER                                                               */
/*****************************************************************************/

/** A simple one way layer with dense connections. */

template<typename Float>
struct Dense_Layer : public Layer {
    Dense_Layer();

    /** Initialize to zero */
    Dense_Layer(size_t inputs, size_t units,
                Transfer_Function_Type transfer_function);

    /** Initialize with random values */
    Dense_Layer(size_t ninputs, size_t units,
                Transfer_Function_Type transfer_function,
                Thread_Context & thread_context,
                float limit = -1.0);

    /// Network parameters
    boost::multi_array<Float, 2> weights;
    distribution<Float> bias;

    /// Values to use for input when the value is missing (NaN)
    distribution<Float> missing_replacements;


    /*************************************************************************/
    /* APPLY                                                                 */
    /*************************************************************************/


    virtual void apply(const float * input, float * output) const;
    virtual void apply(const double * input, double * output) const;



    /*************************************************************************/
    /* PREPROCESS                                                            */
    /*************************************************************************/

    /** Performs pre-processing of the input including any shift and scaling
        factors, potential decorrelation and the replacement of any missing
        values with their replacements.

        Default implementation does nothing.
    */

    virtual void preprocess(const float * input,
                            float * preprocessed) const;

    virtual void preprocess(const double * input,
                            double * preprocessed) const;

    distribution<float> preprocess(const distribution<float> & input) const;
    distribution<double> preprocess(const distribution<double> & input) const;


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
    /* TRANSFER                                                              */
    /*************************************************************************/

    /* Apply the transfer function to the activations. */

    /** Transform the given value according to the transfer function. */
    template<typename FloatIn>
    static void transfer(const FloatIn * activation, FloatIn * outputs,
                         int nvals,
                         Transfer_Function_Type transfer_function);
        

    void transfer(const float * activation, float * outputs) const;
    void transfer(const double * activation, double * outputs) const;

    distribution<float> transfer(const distribution<float> & activation) const;
    distribution<double> transfer(const distribution<double> & activation) const;

    
    /*************************************************************************/
    /* DERIVATIVE                                                            */
    /*************************************************************************/

    /** Return the derivative of the given value according to the transfer
        function.  Note that only the output of the activation function is
        provided; this is sufficient for most cases.
    */
    distribution<float> derivative(const distribution<float> & outputs) const;
    distribution<double> derivative(const distribution<double> & outputs) const;

    template<class FloatIn>
    static void derivative(const FloatIn * outputs, FloatIn * deriv, int nvals,
                           Transfer_Function_Type transfer_function);
    
    virtual void derivative(const float * outputs,
                            float * derivatives) const;
    virtual void derivative(const double * outputs,
                            double * derivatives) const;

    /** These are the same, but they operate on the second derivative. */

    distribution<float>
    second_derivative(const distribution<float> & outputs) const;

    distribution<double>
    second_derivative(const distribution<double> & outputs) const;

    template<class FloatIn>
    static void second_derivative(const FloatIn * outputs,
                                  FloatIn * second_deriv, int nvals,
                                  Transfer_Function_Type transfer_function);

    virtual void second_derivative(const float * outputs,
                                   float * second_derivatives) const;
    virtual void second_derivative(const double * outputs,
                                   double * second_derivatives) const;

    
    /*************************************************************************/
    /* DELTAS                                                                */
    /*************************************************************************/

    /* deltas = derivative * error */

    void deltas(const float * outputs, const float * errors,
                float * deltas) const;
    void deltas(const double * outputs, const double * errors,
                double * deltas) const;







    /** Dump as ASCII.  This will be big. */
    virtual std::string print() const;
    
    /** Return the name of the type */
    virtual std::string type() const;

    virtual void serialize(DB::Store_Writer & store) const;
    virtual void reconstitute(DB::Store_Reader & store);

    /** Fill with random weights. */
    virtual void random_fill(float limit, Thread_Context & context);

    /** Fill with zero values */
    virtual void zero_fill();

    /** Return the number of parameters (degrees of freedom) for the
        layer. */
    virtual size_t parameter_count() const;

    virtual size_t inputs() const { return weights.shape()[0]; }
    virtual size_t outputs() const { return weights.shape()[1]; }

    virtual Dense_Layer * make_copy() const { return new Dense_Layer(*this); }

    virtual void validate() const;

    // For testing purposes
    bool operator == (const Dense_Layer & other) const;
};

JML_IMPL_SERIALIZE_RECONSTITUTE_TEMPLATE(typename Float, Dense_Layer<Float>);


extern template class Dense_Layer<float>;
extern template class Dense_Layer<double>;


} // namespace ML

#endif /* __neural__dense_layer_h__ */
