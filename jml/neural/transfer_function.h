/* transfer_function.h                                             -*- C++ -*-
   Jeremy Barnes, 2 November 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

   Definition of transfer functions.
*/

#ifndef __jml__neural__transfer_function_h__
#define __jml__neural__transfer_function_h__

#include "perceptron_defs.h"
#include "jml/stats/distribution.h"
#include "jml/db/persistent_fwd.h"
#include <boost/shared_ptr.hpp>

namespace ML {


/*****************************************************************************/
/* RANGE_TYPE                                                                */
/*****************************************************************************/

enum Range_Type {
    RT_PROB,    ///< A probability: from zero to one
    RT_PM_ONE,  ///< From -1 to +1
    RT_PM_INF,  ///< From -infinity to plus infinity
    RT_OTHER    ///< Something else
};


/*****************************************************************************/
/* RANGE                                                                     */
/*****************************************************************************/

/** Gives the range of a function (the set of values that its output can
    take).
*/

struct Range {
    float min;  ///< Minimum value it can take on
    float max;  ///< Maximum value it can take on
    float neutral;  ///< A "neutral" value for the range
    bool min_asymptotic;  ///< Approaches but can never reach the minimum
    bool max_asymptotic;  ///< Approaches but can never reach the maximum
    Range_Type type;      ///< Break it down into a category for simplicity

    std::string print() const;
};


/*****************************************************************************/
/* TRANSFER_FUNCTION                                                         */
/*****************************************************************************/

/** A transfer function takes a vector of inputs and transforms them into
    a vector of outputs.  It is most frequently used as the non-linearity
    after a linear activation step in a neural network.

    These objects can also provide the first and second derivatives of their
    input.
*/

struct Transfer_Function {

    virtual ~Transfer_Function() {}

    /*************************************************************************/
    /* INFORMATION                                                           */
    /*************************************************************************/

    virtual Range range() const = 0;

    virtual std::string print() const = 0;

    /** Given the activation function and the maximum amount of the range
        that we want to use (eg, 0.8 for asymptotic functions), what are
        the minimum and maximum values that we want to use.

        For example, tanh goes from -1 to 1, but asymptotically.  We would
        normally want to go from -0.8 to 0.8, to leave a bit of space for
        expansion.

        Mostly used for classification to transform a label number into a
        real valued vector for training and classification.
    */
    virtual std::pair<float, float> targets(float maximum) const = 0;

    virtual bool equal(const Transfer_Function & other) const = 0;

    bool operator == (const Transfer_Function & other) const
    {
        return equal(other);
    }

    bool operator != (const Transfer_Function & other) const
    {
        return ! operator == (other);
    }

    /*************************************************************************/
    /* SERIALIZATION                                                         */
    /*************************************************************************/

    /** Save the type informatinon and the data to the store.  Note that only
        poly_reconstitute can reconstitute the object in this case; the normal
        reconstitute will not work. */
    void poly_serialize(DB::Store_Writer & store) const;

    /** Read the type information from the store, create an object of the
        correct type and reconstitute its data members from the store.  This
        is the companion function to poly_serialize. */
    static std::shared_ptr<Transfer_Function>
    poly_reconstitute(DB::Store_Reader & store);

    /** Serialize the data for this type. */
    virtual void serialize(DB::Store_Writer & store) const = 0;

    /** Reconstitute the data for this type. */
    virtual void reconstitute(DB::Store_Reader & store) = 0;

    virtual std::string class_id() const = 0;


    /*************************************************************************/
    /* TRANSFER                                                              */
    /*************************************************************************/

    /* Apply the transfer function to the activations. */

    virtual void transfer(const float * activation,
                          float * outputs,
                          size_t n) const = 0;
    virtual void transfer(const double * activation,
                          double * outputs,
                          size_t n) const = 0;

    distribution<float>
    transfer(const distribution<float> & activation) const;

    distribution<double>
    transfer(const distribution<double> & activation) const;

    
    /*************************************************************************/
    /* DERIVATIVE                                                            */
    /*************************************************************************/

    /** Return the derivative of the given value according to the transfer
        function.  Note that only the output of the activation function is
        provided; this is sufficient for most cases.
    */
    distribution<float> derivative(const distribution<float> & outputs) const;
    distribution<double> derivative(const distribution<double> & outputs) const;

    virtual void derivative(const float * outputs,
                            float * derivatives,
                            size_t n) const = 0;
    virtual void derivative(const double * outputs,
                            double * derivatives,
                            size_t n) const = 0;


    /*************************************************************************/
    /* SECOND_DERIVATIVE                                                     */
    /*************************************************************************/

    /** These are the same, but they operate on the second derivative. */

    distribution<float>
    second_derivative(const distribution<float> & outputs) const;

    distribution<double>
    second_derivative(const distribution<double> & outputs) const;

    virtual void second_derivative(const float * outputs,
                                   float * second_derivatives,
                                   size_t n) const = 0;
    virtual void second_derivative(const double * outputs,
                                   double * second_derivatives,
                                   size_t n) const = 0;
};


/*****************************************************************************/
/* STANDARD_TRANSFER_FUNCTION                                                */
/*****************************************************************************/

/** A transfer function that implements, in a switched manner, the standard
    ones.  The only parameter is the Transfer_Function_Type which tells
    us which one is being used.
*/

struct Standard_Transfer_Function : public Transfer_Function {
    Standard_Transfer_Function(Transfer_Function_Type transfer_function
                                   = TF_IDENTITY);

    Transfer_Function_Type transfer_function;

    virtual ~Standard_Transfer_Function() {}

    virtual std::string print() const;

    virtual Range range() const;

    virtual std::pair<float, float> targets(float maximum) const;

    virtual void serialize(DB::Store_Writer & store) const;

    virtual void reconstitute(DB::Store_Reader & store);

    virtual std::string class_id() const;

    virtual bool equal(const Transfer_Function & other) const;


    /*************************************************************************/
    /* TRANSFER                                                              */
    /*************************************************************************/

    template<typename FloatIn>
    static void transfer(const FloatIn * activation, FloatIn * outputs,
                         int nvals,
                         Transfer_Function_Type transfer_function);
        
    virtual void transfer(const float * activation, float * outputs,
                          size_t n) const;
    virtual void transfer(const double * activation, double * outputs,
                          size_t n) const;

    using Transfer_Function::transfer;
    
    
    /*************************************************************************/
    /* DERIVATIVE                                                            */
    /*************************************************************************/

    template<class FloatIn>
    static void derivative(const FloatIn * outputs, FloatIn * deriv, int nvals,
                           Transfer_Function_Type transfer_function);
    
    virtual void derivative(const float * outputs,
                            float * derivatives,
                            size_t n) const;
    virtual void derivative(const double * outputs,
                            double * derivatives,
                            size_t n) const;

    using Transfer_Function::derivative;


    /*************************************************************************/
    /* SECOND_DERIVATIVE                                                     */
    /*************************************************************************/

    template<class FloatIn>
    static void second_derivative(const FloatIn * outputs,
                                  FloatIn * second_deriv, int nvals,
                                  Transfer_Function_Type transfer_function);

    virtual void second_derivative(const float * outputs,
                                   float * second_derivatives,
                                   size_t n) const;
    virtual void second_derivative(const double * outputs,
                                   double * second_derivatives,
                                   size_t n) const;

    using Transfer_Function::second_derivative;
};



/*****************************************************************************/
/* FACTORY                                                                   */
/*****************************************************************************/

std::shared_ptr<Transfer_Function>
create_transfer_function(const Transfer_Function_Type & function);

std::shared_ptr<Transfer_Function>
create_transfer_function(const std::string & name);


} // namespace ML

#endif /* __jml__neural__transfer_function_h__ */
