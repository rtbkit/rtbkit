/* layer.h                                                         -*- C++ -*-
   Jeremy Barnes, 20 October 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

   Layers for perceptrons and other similar beasts.
*/

#ifndef __jml__layer_h__
#define __jml__layer_h__

#include "perceptron_defs.h"
#include "boosting/thread_context.h"
#include "stats/distribution.h"
#include <boost/multi_array.hpp>

namespace ML {


/*****************************************************************************/
/* TRANSFER_FUNCTION                                                         */
/*****************************************************************************/

struct Transfer_Function {
    virtual ~Transfer_Function();
};


/*****************************************************************************/
/* LAYER                                                                     */
/*****************************************************************************/

/** A basic layer of a neural network.  Other kinds of layers can be built on
    this base.
*/

class Layer {
public:
    Layer();

    Transfer_Function_Type transfer_function;
        
    /** Dump as ASCII.  This will be big. */
    virtual std::string print() const = 0;
    
    /** Return the name of the type */
    virtual std::string type() const = 0;

    virtual void serialize(DB::Store_Writer & store) const = 0;
    virtual void reconstitute(DB::Store_Reader & store) = 0;


    /*************************************************************************/
    /* APPLY                                                                 */
    /*************************************************************************/

    /* These functions take an input, preprocess it, compute the activations,
       apply the transfer function and return the result.
       
       Equivalent to transfer(activation(preprocess(input))).
    */

    /** Apply the layer to the input and return an output. */
    distribution<float> apply(const distribution<float> & input) const;
    distribution<double> apply(const distribution<double> & input) const;
        
    void apply(const distribution<float> & input,
               distribution<float> & output) const;

    void apply(const distribution<double> & input,
               distribution<double> & output) const;

    void apply(const float * input,
               float * output) const;

    void apply(const double * input,
               double * output) const;


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
                            double * preprocessed) const = 0;

    distribution<float> preprocess(const distribution<float> & input) const;
    distribution<double> preprocess(const distribution<double> & input) const;
    

    /*************************************************************************/
    /* ACTIVATION                                                            */
    /*************************************************************************/

    /* Calculate the activation function for the output neurons */

    virtual void activation(const float * input,
                            float * activation) const;

    virtual void activation(const double * input,
                            double * activation) const = 0;

    distribution<float> activation(const distribution<float> & input) const;
    distribution<double> activation(const distribution<double> & input) const;


    /*************************************************************************/
    /* TRANSFER                                                              */
    /*************************************************************************/

    /* Apply the transfer function to the activations. */

    /** Transform the given value according to the transfer function. */
    template<typename Float>
    static void transfer(const Float * activation, Float * outputs, int nvals,
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

    template<class Float>
    static void derivative(const Float * outputs, Float * deriv, int nvals,
                           Transfer_Function_Type transfer_function);
    
    virtual void derivative(const float * outputs,
                            float * derivatives) const;
    virtual void derivative(const double * outputs,
                            double * derivatives) const;

    
    /*************************************************************************/
    /* DELTAS                                                                */
    /*************************************************************************/

    /* deltas = derivative * error */

    void deltas(const float * outputs, const float * errors,
                float * deltas) const;
    void deltas(const double * outputs, const double * errors,
                double * deltas) const;

    /** Fill with random weights. */
    virtual void random_fill(float limit, Thread_Context & context) = 0;

    virtual void zero_fill() = 0;

    /** Return the number of parameters (degrees of freedom) for the
        layer. */
    virtual size_t parameter_count() const = 0;

    virtual size_t inputs() const = 0;
    virtual size_t outputs() const = 0;

    /** Check that all parameters are reasonable and invariants are met.
        Will throw an exception if there is a problem. */
    virtual void validate() const;

    virtual Layer * make_copy() const = 0;
    
    /** Given the activation function and the maximum amount of the range
        that we want to use (eg, 0.8 for asymptotic functions), what are
        the minimum and maximum values that we want to use.

        For example, tanh goes from -1 to 1, but asymptotically.  We would
        normally want to go from -0.8 to 0.8, to leave a bit of space for
        expansion.
    */
    static std::pair<float, float>
    targets(float maximum, Transfer_Function_Type transfer_function);
};

inline std::ostream & operator << (std::ostream & stream, const Layer & layer)
{
    return stream << layer.print();
}

} // namespace ML

#if 0  // for later if we want to do RBMs
    void apply_stochastic(const distribution<float> & input,
                          distribution<float> & output,
                          Thread_Context & context) const;

    void apply(const float * input, float * output) const;

    void apply_stochastic(const float * input, float * output,
                          Thread_Context & context) const;

    /** Generate a single stochastic Gibbs sample from the stocastic
        distribution, starting from the given input values.  It will
        modify both the input and the output of the new sample.

        Performs the given number of iterations.
    */
    void sample(distribution<float> & input,
                distribution<float> & output,
                Thread_Context & context,
                int num_iterations) const;
#endif // RBMs


#endif /* __jml__layer_h__ */
