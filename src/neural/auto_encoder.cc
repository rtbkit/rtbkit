/* auto_encoder.cc
   Jeremy Barnes, 11 November 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

   Auto-encoder class.
*/

#include "auto_encoder.h"


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

} // namespace ML
