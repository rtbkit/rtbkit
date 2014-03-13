/* reconstruct_layer_adaptor.h                                     -*- C++ -*-
   Jeremy Barnes, 13 November 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

   Adaptor that turns an autoencoder's reconstruct (encode-decode) cycle into
   a single layer's action.
*/

#ifndef __jml__neural__reconstruct_layer_adaptor_h__
#define __jml__neural__reconstruct_layer_adaptor_h__

#include "auto_encoder.h"

namespace ML {


/*****************************************************************************/
/* RECONSTRUCT_LAYER_ADAPTOR                                                 */
/*****************************************************************************/

/** Takes an auto-encoder and reconstructs its direction, both for training and
    for application. */

struct Reconstruct_Layer_Adaptor : public Layer {

    Reconstruct_Layer_Adaptor();
    Reconstruct_Layer_Adaptor(std::shared_ptr<Auto_Encoder>);

    Reconstruct_Layer_Adaptor(const Reconstruct_Layer_Adaptor & other);
    Reconstruct_Layer_Adaptor &
    operator = (const Reconstruct_Layer_Adaptor & other);

    void swap(Reconstruct_Layer_Adaptor & other);


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

    virtual Reconstruct_Layer_Adaptor * make_copy() const;

    virtual Reconstruct_Layer_Adaptor * deep_copy() const;


    /*************************************************************************/
    /* APPLY                                                                 */
    /*************************************************************************/

    virtual void apply(const float * input, float * output) const;
    virtual void apply(const double * input, double * output) const;

    using Layer::apply;


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

    using Layer::fprop;


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

    using Layer::bprop;

    std::shared_ptr<Auto_Encoder> ae_;
};

} // namespace ML

#endif /* __jml__neural__reconstruct_layer_adaptor_h__ */
