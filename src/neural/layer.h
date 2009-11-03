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

    virtual void apply(const float * input, float * output) const = 0;
    virtual void apply(const double * input, double * output) const = 0;

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

#endif /* __jml__layer_h__ */
