/* transfer_function.h                                             -*- C++ -*-
   Jeremy Barnes, 2 November 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

   Definition of transfer functions.
*/

#ifndef __jml__neural__transfer_function_h__
#define __jml__neural__transfer_function_h__

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

struct Range {
    float min;  ///< Minimum value it can take on
    float max;  ///< Maximum value it can take on
    float neutral;  ///< A "neutral" value for the range
    bool min_asymptotic;
    bool max_asymptotic;
    Range_Type type;
};


/*****************************************************************************/
/* TRANSFER_FUNCTION                                                         */
/*****************************************************************************/

struct Transfer_Function {
    virtual ~Transfer_Function();


    /*************************************************************************/
    /* TRANSFER                                                              */
    /*************************************************************************/

    /* Apply the transfer function to the activations. */

    /** Transform the given value according to the transfer function. */
    template<typename FloatIn>
    static void transfer(const FloatIn * activation, FloatIn * outputs,
                         int nvals,
                         Transfer_Function_Type transfer_function);
        
    virtual Range range() const;


    void transfer(const float * activation, float * outputs) const;
    void transfer(const double * activation, double * outputs) const;

    distribution<float> transfer(const distribution<float> & activation) const;
    distribution<double> transfer(const distribution<double> & activation) const;
    
    /** Given the activation function and the maximum amount of the range
        that we want to use (eg, 0.8 for asymptotic functions), what are
        the minimum and maximum values that we want to use.

        For example, tanh goes from -1 to 1, but asymptotically.  We would
        normally want to go from -0.8 to 0.8, to leave a bit of space for
        expansion.
    */
    static std::pair<float, float>
    targets(float maximum, Transfer_Function_Type transfer_function);

    
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
};


/*****************************************************************************/
/* FACTORY                                                                   */
/*****************************************************************************/

boost::shared_ptr<Transfer_Function>
create_transfer_function(const Transfer_Function_Type & function);


} // namespace ML

#endif /* __jml__neural__transfer_function_h__ */
