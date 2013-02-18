/* reverse_layer_adaptor.h                                         -*- C++ -*-
   Jeremy Barnes, 12 November 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

   An adaptor to reverse an Auto_Encoder and make it behave in the
   opposite direction.
*/

#ifndef __jml__neural__reverse_layer_adaptor_h__
#define __jml__neural__reverse_layer_adaptor_h__

#include "auto_encoder.h"

namespace ML {


/*****************************************************************************/
/* REVERSE_LAYER_ADAPTOR                                                     */
/*****************************************************************************/

/** Takes an auto-encoder and reverses its direction, both for training and
    for application. */

struct Reverse_Layer_Adaptor : public Auto_Encoder {

    Reverse_Layer_Adaptor();
    Reverse_Layer_Adaptor(std::shared_ptr<Auto_Encoder>);

    Reverse_Layer_Adaptor(const Reverse_Layer_Adaptor & other);
    Reverse_Layer_Adaptor & operator = (const Reverse_Layer_Adaptor & other);

    void swap(Reverse_Layer_Adaptor & other);


    /*************************************************************************/
    /* INFO                                                                  */
    /*************************************************************************/

    virtual std::string print() const;
    
    virtual std::string class_id() const;

    virtual size_t max_width() const;

    virtual std::pair<float, float> targets(float maximum) const;

    virtual void validate() const;

    virtual bool equal_impl(const Layer & other) const;

    virtual bool supports_missing_inputs() const;


    /*************************************************************************/
    /* PARAMETERS                                                            */
    /*************************************************************************/

    virtual void add_parameters(Parameters & params);

    virtual size_t parameter_count() const;

    virtual void random_fill(float limit, Thread_Context & context);

    virtual void zero_fill();

    
    /*************************************************************************/
    /* SERIALIZATION                                                         */
    /*************************************************************************/

    virtual void serialize(DB::Store_Writer & store) const;

    virtual void reconstitute(DB::Store_Reader & store);

    virtual Reverse_Layer_Adaptor * make_copy() const;

    virtual Reverse_Layer_Adaptor * deep_copy() const;


    /*************************************************************************/
    /* APPLY                                                                 */
    /*************************************************************************/

    virtual void apply(const float * input, float * output) const;
    virtual void apply(const double * input, double * output) const;

    using Auto_Encoder::apply;


    /*************************************************************************/
    /* FPROP                                                                 */
    /*************************************************************************/

    virtual size_t fprop_temporary_space_required() const;
    virtual void
    fprop(const float * inputs,
          float * temp_space, size_t temp_space_size,
          float * outputs) const;

    /** \copydoc fprop */
    virtual void
    fprop(const double * inputs,
          double * temp_space, size_t temp_space_size,
          double * outputs) const;

    using Auto_Encoder::fprop;


    /*************************************************************************/
    /* BPROP                                                                 */
    /*************************************************************************/

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

    using Auto_Encoder::bprop;

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

    virtual std::pair<float, float> itargets(float maximum) const;
    virtual bool supports_missing_outputs() const;

    virtual void iapply(const float * input, float * output) const;
    virtual void iapply(const double * input, double * output) const;

    using Auto_Encoder::iapply;

    virtual size_t ifprop_temporary_space_required() const;

    virtual void
    ifprop(const float * outputs,
           float * temp_space, size_t temp_space_size,
           float * inputs) const;

    virtual void
    ifprop(const double * outputs,
           double * temp_space, size_t temp_space_size,
           double * inputs) const;

    virtual void ibprop(const float * outputs,
                        const float * inputs,
                        const float * temp_space, size_t temp_space_size,
                        const float * input_errors,
                        float * output_errors,
                        Parameters & gradient,
                        double example_weight) const;
    
    virtual void ibprop(const double * outputs,
                        const double * inputs,
                        const double * temp_space, size_t temp_space_size,
                        const double * input_errors,
                        double * output_errors,
                        Parameters & gradient,
                        double example_weight) const;

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


    /*************************************************************************/
    /* RECONSTRUCTION                                                        */
    /*************************************************************************/

#if 0 // default versions from Auto_Encoder are OK
    virtual void reconstruct(const float * input, float * output) const;
    virtual void reconstruct(const double * input, double * output) const;

    virtual size_t rfprop_temporary_space_required() const;
    virtual void
    rfprop(const float * inputs,
           float * temp_space, size_t temp_space_size,
           float * reconstruction) const;

    virtual void
    rfprop(const double * inputs,
           double * temp_space, size_t temp_space_size,
           double * reconstruction) const;

    virtual void rbprop(const float * inputs,
                        const float * reconstruction,
                        const float * temp_space,
                        size_t temp_space_size,
                        const float * reconstruction_errors,
                        Parameters & gradient,
                        double example_weight) const;

    virtual void rbprop(const double * inputs,
                        const double * reconstruction,
                        const double * temp_space,
                        size_t temp_space_size,
                        const double * reconstruction_errors,
                        Parameters & gradient,
                        double example_weight) const;

#endif // default versions OK

    std::shared_ptr<Auto_Encoder> ae_;

};

} // namespace ML

#endif /* __jml__neural__reverse_layer_adaptor_h__ */
