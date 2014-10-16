/* reconstruct_layer_adaptor.cc
   Jeremy Barnes, 13 November 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

   Layer adaptor to make the reconstruction look like a normal layer.
*/

#include "reconstruct_layer_adaptor.h"
#include "jml/db/persistent.h"


using namespace std;
using namespace ML::DB;


namespace ML {


/*****************************************************************************/
/* RECONSTRUCT_LAYER_ADAPTOR                                                 */
/*****************************************************************************/

Reconstruct_Layer_Adaptor::
Reconstruct_Layer_Adaptor()
    : Layer("", 0, 0)
{
}

Reconstruct_Layer_Adaptor::
Reconstruct_Layer_Adaptor(std::shared_ptr<Auto_Encoder> enc)
    : Layer(enc->name(), enc->inputs(), enc->inputs()), ae_(enc)
{
    update_parameters();
}

Reconstruct_Layer_Adaptor::
Reconstruct_Layer_Adaptor(const Reconstruct_Layer_Adaptor & other)
    : Layer(other), ae_(other.ae_)
{
    update_parameters();
}

Reconstruct_Layer_Adaptor &
Reconstruct_Layer_Adaptor::
operator = (const Reconstruct_Layer_Adaptor & other)
{
    Reconstruct_Layer_Adaptor new_me(other);
    swap(new_me);
    return *this;
}

void
Reconstruct_Layer_Adaptor::
swap(Reconstruct_Layer_Adaptor & other)
{
    Layer::swap(other);
    ae_.swap(other.ae_);
}

std::string
Reconstruct_Layer_Adaptor::
print() const
{
    return "Reconstructed version of " + ae_->print();
}
    
std::string
Reconstruct_Layer_Adaptor::
class_id() const
{
    return "Reconstruct_Layer_Adaptor";
}

size_t
Reconstruct_Layer_Adaptor::
max_width() const
{
    return ae_->max_width();
}

std::pair<float, float>
Reconstruct_Layer_Adaptor::
targets(float maximum) const
{
    return ae_->targets(maximum);
}

void
Reconstruct_Layer_Adaptor::
validate() const
{
    Layer::validate();
    ae_->validate();
}

bool
Reconstruct_Layer_Adaptor::
equal_impl(const Layer & other) const
{
    const Reconstruct_Layer_Adaptor & cast
        = dynamic_cast<const Reconstruct_Layer_Adaptor &>(other);
    return operator == (cast);
}

bool
Reconstruct_Layer_Adaptor::
supports_missing_inputs() const
{
    return ae_->supports_missing_inputs();
}

void
Reconstruct_Layer_Adaptor::
add_parameters(Parameters & params)
{
    ae_->add_parameters(params);
}

size_t
Reconstruct_Layer_Adaptor::
parameter_count() const
{
    return ae_->parameter_count();
}

void
Reconstruct_Layer_Adaptor::
random_fill(float limit, Thread_Context & context)
{
    ae_->random_fill(limit, context);
}

void
Reconstruct_Layer_Adaptor::
zero_fill()
{
    ae_->zero_fill();
}

void
Reconstruct_Layer_Adaptor::
serialize(DB::Store_Writer & store) const
{
    store << (char)1; // version
    ae_->poly_serialize(store);
}

void
Reconstruct_Layer_Adaptor::
reconstitute(DB::Store_Reader & store)
{
    char version;
    store >> version;
    if (version != 1)
        throw Exception("Reconstruct_Layer_Adaptor::reconstitute(): "
                        "unknown version");
    ae_ = std::dynamic_pointer_cast<Auto_Encoder>(poly_reconstitute(store));
    if (!ae_)
        throw Exception("Reconstruct_Layer_Adaptor::reconstiute(): "
                        "couldn't reconstitute contained object");
}

Reconstruct_Layer_Adaptor *
Reconstruct_Layer_Adaptor::
make_copy() const
{
    return new Reconstruct_Layer_Adaptor(*this);
}

Reconstruct_Layer_Adaptor *
Reconstruct_Layer_Adaptor::
deep_copy() const
{
    return new Reconstruct_Layer_Adaptor(make_sp(ae_->deep_copy()));
}

void
Reconstruct_Layer_Adaptor::
apply(const float * input, float * output) const
{
    
    ae_->reconstruct(input, output);
}

void
Reconstruct_Layer_Adaptor::
apply(const double * input, double * output) const
{
    ae_->reconstruct(input, output);
}

size_t
Reconstruct_Layer_Adaptor::
fprop_temporary_space_required() const
{
    return ae_->rfprop_temporary_space_required();
}

void
Reconstruct_Layer_Adaptor::
fprop(const float * inputs,
      float * temp_space, size_t temp_space_size,
      float * outputs) const
{
    ae_->rfprop(inputs, temp_space, temp_space_size, outputs);
}

void
Reconstruct_Layer_Adaptor::
fprop(const double * inputs,
      double * temp_space, size_t temp_space_size,
      double * outputs) const
{
    ae_->rfprop(inputs, temp_space, temp_space_size, outputs);
}

void
Reconstruct_Layer_Adaptor::
bprop(const float * inputs,
      const float * outputs,
      const float * temp_space, size_t temp_space_size,
      const float * output_errors,
      float * input_errors,
      Parameters & gradient,
      double example_weight) const
{
    ae_->rbprop(inputs, outputs, temp_space, temp_space_size, output_errors,
                input_errors, gradient, example_weight);
}

void
Reconstruct_Layer_Adaptor::
bprop(const double * inputs,
      const double * outputs,
      const double * temp_space, size_t temp_space_size,
      const double * output_errors,
      double * input_errors,
      Parameters & gradient,
      double example_weight) const
{
    ae_->rbprop(inputs, outputs, temp_space, temp_space_size, output_errors,
                input_errors, gradient, example_weight);
}

void
Reconstruct_Layer_Adaptor::
bbprop(const float * inputs,
       const float * outputs,
       const float * temp_space, size_t temp_space_size,
       const float * output_errors,
       const float * d2output_errors,
       float * input_errors,
       float * d2input_errors,
       Parameters & gradient,
       Parameters * dgradient,
       double example_weight) const
{
    ae_->rbbprop(inputs, outputs, temp_space, temp_space_size,
                 output_errors, d2output_errors, input_errors, d2input_errors,
                 gradient, dgradient, example_weight);
}

void
Reconstruct_Layer_Adaptor::
bbprop(const double * inputs,
       const double * outputs,
       const double * temp_space, size_t temp_space_size,
       const double * output_errors,
       const double * d2output_errors,
       double * input_errors,
       double * d2input_errors,
       Parameters & gradient,
       Parameters * dgradient,
       double example_weight) const
{
    ae_->rbbprop(inputs, outputs, temp_space, temp_space_size,
                 output_errors, d2output_errors, input_errors, d2input_errors,
                 gradient, dgradient, example_weight);
}

} // namespace ML
