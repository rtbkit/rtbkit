/* auto_encoder.cc
   Jeremy Barnes, 11 November 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

   Auto-encoder class.
*/

#include "auto_encoder.h"
#include "reverse_layer_adaptor.h"
#include "reconstruct_layer_adaptor.h"

namespace ML {


/*****************************************************************************/
/* AUTO_ENCODER                                                              */
/*****************************************************************************/

Auto_Encoder::
Auto_Encoder()
    : Layer("", 0, 0)
{
}

Auto_Encoder::
Auto_Encoder(const std::string & name, int inputs, int outputs)
    : Layer(name, inputs, outputs)
{
}

distribution<double>
Auto_Encoder::
iapply(const distribution<double> & outputs) const
{
    if (outputs.size() != this->outputs())
        throw Exception("iapply(): output provided was wrong size");
    distribution<double> inputs(this->inputs());
    iapply(&outputs[0], &inputs[0]);
    return inputs;
}

distribution<float>
Auto_Encoder::
iapply(const distribution<float> & outputs) const
{
    if (outputs.size() != this->outputs())
        throw Exception("iapply(): output provided was wrong size");
    distribution<float> inputs(this->inputs());
    iapply(&outputs[0], &inputs[0]);
    return inputs;
}

void
Auto_Encoder::
ibbprop(const float * outputs,
        const float * inputs,
        const float * temp_space, size_t temp_space_size,
        const float * input_errors,
        const float * d2input_errors,
        float * output_errors,
        float * d2output_errors,
        Parameters & gradient,
        Parameters * dgradient,
        double example_weight) const
{
    ibbprop_jacobian(outputs, inputs, temp_space, temp_space_size,
                     input_errors, d2input_errors, output_errors,
                     d2output_errors, gradient, dgradient, example_weight);
}
 
void
Auto_Encoder::
ibbprop(const double * outputs,
        const double * inputs,
        const double * temp_space, size_t temp_space_size,
        const double * input_errors,
        const double * d2input_errors,
        double * output_errors,
        double * d2output_errors,
        Parameters & gradient,
        Parameters * dgradient,
        double example_weight) const
{
    ibbprop_jacobian(outputs, inputs, temp_space, temp_space_size,
                     input_errors, d2input_errors, output_errors,
                     d2output_errors, gradient, dgradient, example_weight);
}
 
template<typename F>
void
Auto_Encoder::
ibbprop_jacobian(const F * outputs,
                 const F * inputs,
                 const F * temp_space, size_t temp_space_size,
                 const F * input_errors,
                 const F * d2input_errors,
                 F * output_errors,
                 F * d2output_errors,
                 Parameters & gradient,
                 Parameters * dgradient,
                 double example_weight) const
{
    // We reverse ourself and call the layer version
    // TODO: undo hack here; make layer adaptor know about constness
    Reverse_Layer_Adaptor adaptor(make_unowned_sp(const_cast<Auto_Encoder &>(*this)));
    adaptor.bbprop_jacobian(outputs, inputs, temp_space, temp_space_size,
                            input_errors, d2input_errors, output_errors,
                            d2output_errors, gradient, dgradient,
                            example_weight);
}

void
Auto_Encoder::
reconstruct(const float * inputs, float * reconstructed) const
{
    float outputs[this->outputs()];
    apply(inputs, outputs);
    iapply(outputs, reconstructed);
}

void
Auto_Encoder::
reconstruct(const double * inputs, double * reconstructed) const
{
    double outputs[this->outputs()];
    apply(inputs, outputs);
    iapply(outputs, reconstructed);
}

distribution<float>
Auto_Encoder::
reconstruct(const distribution<float> & input) const
{
    if (input.size() != this->inputs())
        throw Exception("reconstruct(): input was wrong size");
    distribution<float> reconstructed(this->inputs());
    reconstruct(&input[0], &reconstructed[0]);
    return reconstructed;
}

distribution<double>
Auto_Encoder::
reconstruct(const distribution<double> & inputs) const
{
    if (inputs.size() != this->inputs())
        throw Exception("reconstruct(): input was wrong size");
    distribution<double> reconstructed(this->inputs());
    reconstruct(&inputs[0], &reconstructed[0]);
    return reconstructed;
}

size_t
Auto_Encoder::
rfprop_temporary_space_required() const
{
    // We need to save the hidden outputs, plus whatever is needed for the
    // fprop in the two directions.
    return outputs()
        + fprop_temporary_space_required()
        + ifprop_temporary_space_required();
}

template<typename F>
void
Auto_Encoder::
rfprop(const F * inputs,
       F * temp_space, size_t temp_space_size,
       F * reconstruction) const
{
    // Temporary space:
    // 
    // +-----------+-------------+---------------+
    // |  fprop    | outputs     |   ifprop      |
    // +-----------+-------------+---------------+
    // |<- fspace->|<-   no    ->|<-  ifspace  ->|

    size_t fspace = fprop_temporary_space_required();
    size_t ifspace = ifprop_temporary_space_required();

    if (temp_space_size != this->outputs() + fspace + ifspace)
        throw Exception("wrong temporary space size");

    F * outputs = temp_space + fspace;
    F * itemp_space = outputs + this->outputs();

    fprop(inputs, temp_space, fspace, outputs);
    ifprop(outputs, itemp_space, ifspace, reconstruction);
}

void
Auto_Encoder::
rfprop(const float * inputs,
       float * temp_space, size_t temp_space_size,
       float * reconstruction) const
{
    return rfprop<float>(inputs, temp_space, temp_space_size, reconstruction);
}

void
Auto_Encoder::
rfprop(const double * inputs,
       double * temp_space, size_t temp_space_size,
       double * reconstruction) const
{
    return rfprop<double>(inputs, temp_space, temp_space_size, reconstruction);
}

template<typename F>
void
Auto_Encoder::
rbprop(const F * inputs,
       const F * reconstruction,
       const F * temp_space,
       size_t temp_space_size,
       const F * reconstruction_errors,
       F * input_errors_out,
       Parameters & gradient,
       double example_weight) const
{
    // Temporary space:
    // 
    // +-----------+-------------+---------------+
    // |  fprop    | outputs     |   ifprop      |
    // +-----------+-------------+---------------+
    // |<- fspace->|<-   no    ->|<-  ifspace  ->|

    size_t fspace = fprop_temporary_space_required();
    size_t ifspace = ifprop_temporary_space_required();

    if (temp_space_size != this->outputs() + fspace + ifspace)
        throw Exception("wrong temporary space size");

    const F * outputs = temp_space + fspace;
    const F * itemp_space = outputs + this->outputs();

    // output error gradients
    F output_errors[this->outputs()];

    ibprop(outputs, reconstruction, itemp_space, ifspace,
           reconstruction_errors, output_errors, gradient, example_weight);
    
    bprop(inputs, outputs, temp_space, fspace, output_errors, input_errors_out,
          gradient, example_weight);
}
    
void
Auto_Encoder::
rbprop(const float * inputs,
       const float * reconstruction,
       const float * temp_space,
       size_t temp_space_size,
       const float * reconstruction_errors,
       float * input_errors_out,
       Parameters & gradient,
       double example_weight) const
{
    return rbprop<float>(inputs, reconstruction, temp_space, temp_space_size,
                         reconstruction_errors, input_errors_out,
                         gradient, example_weight);
}
    
void
Auto_Encoder::
rbprop(const double * inputs,
       const double * reconstruction,
       const double * temp_space,
       size_t temp_space_size,
       const double * reconstruction_errors,
       double * input_errors_out,
       Parameters & gradient,
       double example_weight) const
{
    return rbprop<double>(inputs, reconstruction, temp_space, temp_space_size,
                          reconstruction_errors, input_errors_out,
                          gradient, example_weight);
}

void
Auto_Encoder::
rbbprop(const float * inputs,
        const float * reconstruction,
        const float * temp_space, size_t temp_space_size,
        const float * reconstruction_errors,
        const float * d2reconstruction_errors,
        float * input_errors,
        float * d2input_errors,
        Parameters & gradient,
        Parameters * dgradient,
        double example_weight) const
{
    rbbprop<float>(inputs, reconstruction, temp_space, temp_space_size,
                   reconstruction_errors, d2reconstruction_errors,
                   input_errors, d2input_errors, gradient, dgradient,
                   example_weight);
}
 
void
Auto_Encoder::
rbbprop(const double * inputs,
        const double * reconstruction,
        const double * temp_space, size_t temp_space_size,
        const double * reconstruction_errors,
        const double * d2reconstruction_errors,
        double * input_errors,
        double * d2input_errors,
        Parameters & gradient,
        Parameters * dgradient,
        double example_weight) const
{
    rbbprop<double>(inputs, reconstruction, temp_space, temp_space_size,
                    reconstruction_errors, d2reconstruction_errors,
                    input_errors, d2input_errors, gradient, dgradient,
                    example_weight);
}

template<typename F>
void
Auto_Encoder::
rbbprop(const F * inputs,
        const F * reconstruction,
        const F * temp_space, size_t temp_space_size,
        const F * reconstruction_errors,
        const F * dreconstruction_errors,
        F * input_errors,
        F * dinput_errors,
        Parameters & gradient,
        Parameters * dgradient,
        double example_weight) const
{


    if (dinput_errors == 0 && dgradient == 0)
        return rbprop(inputs, reconstruction, temp_space, temp_space_size,
                      reconstruction_errors, input_errors, gradient,
                      example_weight);

#if 0
    Reconstruct_Layer_Adaptor adaptor(make_unowned_sp(const_cast<Auto_Encoder & >(*this)));

    return adaptor.bbprop_jacobian(inputs, reconstruction, temp_space, temp_space_size,
                          reconstruction_errors, dreconstruction_errors,
                          input_errors, dinput_errors,
                          gradient, dgradient, example_weight);

    return adaptor.bbprop(inputs, reconstruction, temp_space, temp_space_size,
                          reconstruction_errors, dreconstruction_errors,
                          input_errors, dinput_errors,
                          gradient, dgradient, example_weight);
#endif
        

    // Temporary space:
    // 
    // +-----------+-------------+---------------+
    // |  fprop    | outputs     |   ifprop      |
    // +-----------+-------------+---------------+
    // |<- fspace->|<-   no    ->|<-  ifspace  ->|

    size_t fspace = fprop_temporary_space_required();
    size_t ifspace = ifprop_temporary_space_required();

    if (temp_space_size != this->outputs() + fspace + ifspace)
        throw Exception("wrong temporary space size");

    const F * outputs = temp_space + fspace;
    const F * itemp_space = outputs + this->outputs();

    // output error gradients
    F output_errors[this->outputs()];
    F doutput_errors[this->outputs()];

    ibbprop(outputs, reconstruction, itemp_space, ifspace,
            reconstruction_errors, dreconstruction_errors,
            output_errors, doutput_errors, gradient, dgradient, example_weight);
    
    bbprop(inputs, outputs, temp_space, fspace, output_errors, doutput_errors,
           input_errors, dinput_errors, gradient, dgradient, example_weight);
}
    
} // namespace ML
