/* auto_encoder.h                                                  -*- C++ -*-
   Jeremy Barnes, 11 November 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

   Abstract class for an auto encoder.
*/

#ifndef __jml__neural__auto_encoder_h__
#define __jml__neural__auto_encoder_h__

#include "layer.h"

namespace ML {


/*****************************************************************************/
/* AUTO_ENCODER                                                              */
/*****************************************************************************/

/** A neural network layer that acts as an auto-encoder.  It can both predict
    the output from the input and predict the input from the output.
*/

struct Auto_Encoder : public Layer {

    Auto_Encoder();
    Auto_Encoder(const std::string & name, int inputs, int outputs);

    virtual Auto_Encoder * make_copy() const = 0;
    virtual Auto_Encoder * deep_copy() const = 0;


    /*************************************************************************/
    /* INVERSE DIRECTION                                                     */
    /*************************************************************************/

    /** \name Inverse Direction

        These methods do the same as their counterparts with no i, but in the
        opposite direction (from the "outputs" to the "inputs").

        @{
    */

    /** Given the activation function and the maximum amount of the range
        that we want to use (eg, 0.8 for asymptotic functions), what are
        the minimum and maximum values that we want to use.

        For example, tanh goes from -1 to 1, but asymptotically.  We would
        normally want to go from -0.8 to 0.8, so that we didn't force too
        hard to get there.
    */
    virtual std::pair<float, float> itargets(float maximum) const = 0;

    /** When running in the inverse direction, are missing outputs (NaN values)
        supported? */
        
    virtual bool supports_missing_outputs() const = 0;

    /** The backwards counterpart of the apply() function. */
    virtual void iapply(const float * input, float * output) const = 0;
    virtual void iapply(const double * input, double * output) const = 0;

    distribution<double>
    iapply(const distribution<double> & output) const;
    distribution<float>
    iapply(const distribution<float> & output) const;

    /** Return the amount of space necessary to save temporary results for the
        inverse forward prop.  There will be an array of the given precision
        (double or single) provided.
    */
    virtual size_t ifprop_temporary_space_required() const = 0;

    /** These functions perform an inverse forward propagation.  They also
        save whatever information is necessary to perform an efficient
        inverse backprop at a later period in time.

        They are the inverse counterparts of the fprop() function.
    */
    virtual void
    ifprop(const float * outputs,
           float * temp_space, size_t temp_space_size,
           float * inputs) const = 0;

    /** \copydoc ifprop */
    virtual void
    ifprop(const double * outputs,
           double * temp_space, size_t temp_space_size,
           double * inputs) const = 0;

    /** Perform an inverse back propagation.  Given the derivative of the
        error with
        respect to each of the errors, they compute the gradient of the
        parameter space.
    */
    virtual void ibprop(const float * outputs,
                        const float * inputs,
                        const float * temp_space, size_t temp_space_size,
                        const float * input_errors,
                        float * output_errors,
                        Parameters & gradient,
                        double example_weight) const = 0;
    
    /** \copydoc ibprop */
    virtual void ibprop(const double * outputs,
                        const double * inputs,
                        const double * temp_space, size_t temp_space_size,
                        const double * input_errors,
                        double * output_errors,
                        Parameters & gradient,
                        double example_weight) const = 0;

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
    void ibbprop_jacobian(const F * outputs,
                          const F * inputs,
                          const F * temp_space, size_t temp_space_size,
                          const F * input_errors,
                          const F * d2input_errors,
                          F * output_errors,
                          F * d2output_errors,
                          Parameters & gradient,
                          Parameters * dgradient,
                          double example_weight) const;
    
    /// @}


    /*************************************************************************/
    /* RECONSTRUCTION                                                        */
    /*************************************************************************/

    /** \name Reconstruction

        These methods are associated with a reconstruction pass: where we
        take an input, convert it into an internal representation and then
        try to reconstruct the input again from this internal representation.

        @{
    */

    virtual void reconstruct(const float * input, float * output) const;
    virtual void reconstruct(const double * input, double * output) const;

    distribution<float>
    reconstruct(const distribution<float> & input) const;

    distribution<double>
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

    template<typename F>
    void
    rfprop(const F * inputs,
           F * temp_space, size_t temp_space_size,
           F * reconstruction) const;
    
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

    virtual void rbbprop(const float * inputs,
                         const float * reconstruction,
                         const float * temp_space, size_t temp_space_size,
                         const float * reconstruction_errors,
                         const float * d2reconstruction_errors,
                         float * input_errors,
                         float * d2input_errors,
                         Parameters & gradient,
                         Parameters * dgradient,
                         double example_weight) const;
 
    virtual void rbbprop(const double * inputs,
                         const double * reconstruction,
                         const double * temp_space, size_t temp_space_size,
                         const double * reconstruction_errors,
                         const double * d2reconstruction_errors,
                         double * input_errors,
                         double * d2input_errors,
                         Parameters & gradient,
                         Parameters * dgradient,
                         double example_weight) const;

    template<typename F>
    void rbbprop(const F * inputs,
                 const F * reconstruction,
                 const F * temp_space, size_t temp_space_size,
                 const F * reconstruction_errors,
                 const F * d2reconstruction_errors,
                 F * input_errors,
                 F * d2input_errors,
                 Parameters & gradient,
                 Parameters * dgradient,
                 double example_weight) const;
    
    /// @}
};

} // namespace ML


#endif /* __jml__neural__auto_encoder_h__ */
