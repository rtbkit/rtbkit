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
#include "parameters.h"


namespace ML {


/*****************************************************************************/
/* LAYER                                                                     */
/*****************************************************************************/

/** A basic layer of a neural network.  Other kinds of layers can be built on
    this base.
*/

class Layer {
public:
    Layer(const std::string & name,
          size_t inputs, size_t outputs);

    /*************************************************************************/
    /* INFO                                                                  */
    /*************************************************************************/

    /** Dump as ASCII.  This will be big. */
    virtual std::string print() const = 0;
    
    /** Return the name of the type */
    virtual std::string class_id() const = 0;

    size_t inputs() const { return inputs_; }
    size_t outputs() const { return outputs_; }

    /** Given the activation function and the maximum amount of the range
        that we want to use (eg, 0.8 for asymptotic functions), what are
        the minimum and maximum values that we want to use.

        For example, tanh goes from -1 to 1, but asymptotically.  We would
        normally want to go from -0.8 to 0.8, so that we didn't force too
        hard to get there.
    */
    virtual std::pair<float, float> targets(float maximum) const = 0;

    /** Check that all parameters are reasonable and invariants are met.
        Will throw an exception if there is a problem. */
    virtual void validate() const;


    /*************************************************************************/
    /* PARAMETERS                                                            */
    /*************************************************************************/

    /** Return a reference to a parameters object that describes this layer's
        parameters.  It should provide a reference. */
    virtual boost::shared_ptr<Parameters> parameters() = 0;

    /** Return the number of parameters (degrees of freedom) for the
        layer. */
    virtual size_t parameter_count() const = 0;

    /** Fill with random weights. */
    virtual void random_fill(float limit, Thread_Context & context) = 0;

    virtual void zero_fill() = 0;

    
    /*************************************************************************/
    /* SERIALIZATION                                                         */
    /*************************************************************************/

    virtual void serialize(DB::Store_Writer & store) const = 0;
    virtual void reconstitute(DB::Store_Reader & store) = 0;

    virtual Layer * make_copy() const = 0;

    /** Make a copy that is not connected to those underneath. */
    virtual Layer * deep_copy() const = 0;

    void poly_serialize(ML::DB::Store_Writer & store) const;

    static boost::shared_ptr<Layer>
    poly_reconstitute(ML::DB::Store_Reader & store);


    /*************************************************************************/
    /* APPLY                                                                 */
    /*************************************************************************/

    /* These functions take an input and return the output.  Note that,
       although they perform the same function as a fprop, they don't
       attempt to save information that is necessary for the bprop later, and
       so are more efficient.
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


    /*************************************************************************/
    /* FPROP                                                                 */
    /*************************************************************************/

    /** Return the amount of space necessary to save temporary results for the
        forward prop.  There will be an array of the given precision (double
        or single) provided.

        Default implementation returns outputs().
    */

    virtual size_t fprop_temporary_space_required() const = 0;

    /** These functions perform a forward propagation.  They also save whatever
        information is necessary to perform an efficient backprop at a later
        period in time.

        Default implementation calls apply() and saves the outputs only in the
        temporary space.
    */
    virtual distribution<float>
    fprop(const distribution<float> & inputs,
          float * temp_space, size_t temp_space_size) const = 0;

    virtual distribution<double>
    fprop(const distribution<double> & inputs,
          double * temp_space, size_t temp_space_size) const = 0;


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
                       bool calculate_input_errors) const = 0;

    virtual void bprop(const distribution<double> & output_errors,
                       double * temp_space, size_t temp_space_size,
                       Parameters & gradient,
                       distribution<double> & input_errors,
                       double example_weight,
                       bool calculate_input_errors) const = 0;


#if 0
    /*************************************************************************/
    /* BBPROP                                                                */
    /*************************************************************************/

    /** Second derivative of the parameters with respect to the errors.  Used
        to determine an individual learning rate for each of the parameters.
    */

    /** Does this implement bbprop? */
    virtual bool has_bbprop() const = 0;
 
    /* (todo) */
#endif


protected:
    std::string name_;
    size_t inputs_, outputs_;
};

inline std::ostream & operator << (std::ostream & stream, const Layer & layer)
{
    return stream << layer.print();
}

} // namespace ML

#endif /* __jml__layer_h__ */
