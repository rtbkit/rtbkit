/* reverse_layer_adaptor.cc
   Jeremy Barnes, 12 November 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

   Adaptor to reverse a layer.
*/

#include "reverse_layer_adaptor.h"
#include "jml/db/persistent.h"


using namespace std;
using namespace ML::DB;


namespace ML {


/*****************************************************************************/
/* REVERSE_LAYER_ADAPTOR                                                     */
/*****************************************************************************/

Reverse_Layer_Adaptor::
Reverse_Layer_Adaptor()
    : Auto_Encoder("", 0, 0)
{
}

Reverse_Layer_Adaptor::
Reverse_Layer_Adaptor(std::shared_ptr<Auto_Encoder> enc)
    : Auto_Encoder(enc->name(), enc->outputs(), enc->inputs()), ae_(enc)
{
    update_parameters();
}

Reverse_Layer_Adaptor::
Reverse_Layer_Adaptor(const Reverse_Layer_Adaptor & other)
    : Auto_Encoder(other), ae_(other.ae_)
{
    update_parameters();
}

Reverse_Layer_Adaptor &
Reverse_Layer_Adaptor::
operator = (const Reverse_Layer_Adaptor & other)
{
    Reverse_Layer_Adaptor new_me(other);
    swap(new_me);
    return *this;
}

void
Reverse_Layer_Adaptor::
swap(Reverse_Layer_Adaptor & other)
{
    Auto_Encoder::swap(other);
    ae_.swap(other.ae_);
}

std::string
Reverse_Layer_Adaptor::
print() const
{
    return "Reversed version of " + ae_->print();
}
    
std::string
Reverse_Layer_Adaptor::
class_id() const
{
    return "Reverse_Layer_Adaptor";
}

size_t
Reverse_Layer_Adaptor::
max_width() const
{
    return ae_->max_width();
}

std::pair<float, float>
Reverse_Layer_Adaptor::
targets(float maximum) const
{
    return ae_->itargets(maximum);
}

std::pair<float, float>
Reverse_Layer_Adaptor::
itargets(float maximum) const
{
    return ae_->targets(maximum);
}

void
Reverse_Layer_Adaptor::
validate() const
{
    Auto_Encoder::validate();
    ae_->validate();
}

bool
Reverse_Layer_Adaptor::
equal_impl(const Layer & other) const
{
    const Reverse_Layer_Adaptor & cast
        = dynamic_cast<const Reverse_Layer_Adaptor &>(other);
    return operator == (cast);
}

bool
Reverse_Layer_Adaptor::
supports_missing_inputs() const
{
    return ae_->supports_missing_outputs();
}

bool
Reverse_Layer_Adaptor::
supports_missing_outputs() const
{
    return ae_->supports_missing_inputs();
}

void
Reverse_Layer_Adaptor::
add_parameters(Parameters & params)
{
    ae_->add_parameters(params);
}

size_t
Reverse_Layer_Adaptor::
parameter_count() const
{
    return ae_->parameter_count();
}

void
Reverse_Layer_Adaptor::
random_fill(float limit, Thread_Context & context)
{
    ae_->random_fill(limit, context);
}

void
Reverse_Layer_Adaptor::
zero_fill()
{
    ae_->zero_fill();
}

void
Reverse_Layer_Adaptor::
serialize(DB::Store_Writer & store) const
{
    store << (char)1; // version
    ae_->poly_serialize(store);
}

void
Reverse_Layer_Adaptor::
reconstitute(DB::Store_Reader & store)
{
    char version;
    store >> version;
    if (version != 1)
        throw Exception("Reverse_Layer_Adatptor::reconstitute(): "
                        "unknown version");
    ae_ = std::dynamic_pointer_cast<Auto_Encoder>(poly_reconstitute(store));
    if (!ae_)
        throw Exception("Reverse_Layer_Adaptor::reconstiute(): "
                        "couldn't reconstitute contained object");
}

Reverse_Layer_Adaptor *
Reverse_Layer_Adaptor::
make_copy() const
{
    return new Reverse_Layer_Adaptor(*this);
}

Reverse_Layer_Adaptor *
Reverse_Layer_Adaptor::
deep_copy() const
{
    return new Reverse_Layer_Adaptor(make_sp(ae_->deep_copy()));
}

void
Reverse_Layer_Adaptor::
apply(const float * input, float * output) const
{
    return ae_->iapply(input, output);
}

void
Reverse_Layer_Adaptor::
apply(const double * input, double * output) const
{
    return ae_->iapply(input, output);
}

size_t
Reverse_Layer_Adaptor::
fprop_temporary_space_required() const
{
    return ae_->ifprop_temporary_space_required();
}

void
Reverse_Layer_Adaptor::
fprop(const float * inputs,
      float * temp_space, size_t temp_space_size,
      float * outputs) const
{
    ae_->ifprop(inputs, temp_space, temp_space_size, outputs);
}

void
Reverse_Layer_Adaptor::
fprop(const double * inputs,
      double * temp_space, size_t temp_space_size,
      double * outputs) const
{
    ae_->ifprop(inputs, temp_space, temp_space_size, outputs);
}

void
Reverse_Layer_Adaptor::
bprop(const float * inputs,
      const float * outputs,
      const float * temp_space, size_t temp_space_size,
      const float * output_errors,
      float * input_errors,
      Parameters & gradient,
      double example_weight) const
{
    ae_->ibprop(inputs, outputs, temp_space, temp_space_size, output_errors,
                input_errors, gradient, example_weight);
}

void
Reverse_Layer_Adaptor::
bprop(const double * inputs,
      const double * outputs,
      const double * temp_space, size_t temp_space_size,
      const double * output_errors,
      double * input_errors,
      Parameters & gradient,
      double example_weight) const
{
    ae_->ibprop(inputs, outputs, temp_space, temp_space_size, output_errors,
                input_errors, gradient, example_weight);
}

void
Reverse_Layer_Adaptor::
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
    ae_->ibbprop(inputs, outputs, temp_space, temp_space_size,
                 output_errors, d2output_errors, input_errors, d2input_errors,
                 gradient, dgradient, example_weight);
}

void
Reverse_Layer_Adaptor::
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
    ae_->ibbprop(inputs, outputs, temp_space, temp_space_size,
                 output_errors, d2output_errors, input_errors, d2input_errors,
                 gradient, dgradient, example_weight);
}

void
Reverse_Layer_Adaptor::
iapply(const float * outputs, float * inputs) const
{
    ae_->apply(outputs, inputs);
}

void
Reverse_Layer_Adaptor::
iapply(const double * outputs, double * inputs) const
{
    ae_->apply(outputs, inputs);
}

size_t
Reverse_Layer_Adaptor::
ifprop_temporary_space_required() const
{
    return ae_->fprop_temporary_space_required();
}

void
Reverse_Layer_Adaptor::
ifprop(const float * outputs,
       float * temp_space, size_t temp_space_size,
       float * inputs) const
{
    ae_->fprop(outputs, temp_space, temp_space_size, inputs);
}

void
Reverse_Layer_Adaptor::
ifprop(const double * outputs,
       double * temp_space, size_t temp_space_size,
       double * inputs) const
{
    ae_->fprop(outputs, temp_space, temp_space_size, inputs);
}

void
Reverse_Layer_Adaptor::
ibprop(const float * outputs,
       const float * inputs,
       const float * temp_space, size_t temp_space_size,
       const float * input_errors,
       float * output_errors,
       Parameters & gradient,
       double example_weight) const
{
    ae_->bprop(outputs, inputs, temp_space, temp_space_size, input_errors,
               output_errors, gradient, example_weight);
}
    
void
Reverse_Layer_Adaptor::
ibprop(const double * outputs,
       const double * inputs,
       const double * temp_space, size_t temp_space_size,
       const double * input_errors,
       double * output_errors,
       Parameters & gradient,
       double example_weight) const
{
    ae_->bprop(outputs, inputs, temp_space, temp_space_size, input_errors,
               output_errors, gradient, example_weight);
}

void
Reverse_Layer_Adaptor::
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
    ae_->bbprop(outputs, inputs, temp_space, temp_space_size,
                input_errors, d2input_errors, output_errors, d2output_errors,
                gradient, dgradient, example_weight);
}

void
Reverse_Layer_Adaptor::
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
    ae_->bbprop(outputs, inputs, temp_space, temp_space_size,
                input_errors, d2input_errors, output_errors, d2output_errors,
                gradient, dgradient, example_weight);
}

#if 0 // default versions from Auto_Encoder are OK
void
Reverse_Layer_Adaptor::
reconstruct(const float * input, float * output) const
{
}

void
Reverse_Layer_Adaptor::
reconstruct(const double * input, double * output) const
{
}

size_t
Reverse_Layer_Adaptor::
rfprop_temporary_space_required() const
{
}

void
Reverse_Layer_Adaptor::
rfprop(const float * inputs,
       float * temp_space, size_t temp_space_size,
       float * reconstruction) const
{
}

void
Reverse_Layer_Adaptor::
rfprop(const double * inputs,
       double * temp_space, size_t temp_space_size,
       double * reconstruction) const
{
}

void
Reverse_Layer_Adaptor::
rbprop(const float * inputs,
       const float * reconstruction,
       const float * temp_space,
       size_t temp_space_size,
       const float * reconstruction_errors,
       Parameters & gradient,
       double example_weight) const
{
}

void
Reverse_Layer_Adaptor::
rbprop(const double * inputs,
       const double * reconstruction,
       const double * temp_space,
       size_t temp_space_size,
       const double * reconstruction_errors,
       Parameters & gradient,
       double example_weight) const
{
}

#endif // default versions OK

} // namespace ML
