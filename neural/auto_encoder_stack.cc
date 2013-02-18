/* auto_encoder_stack.cc                                           -*- C++ -*-
   Jeremy Barnes, 11 November 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

   Implementation of a stack of auto-encoders.
*/

#undef NDEBUG

#include "auto_encoder_stack.h"
#include "jml/boosting/registry.h"
#include "layer_stack_impl.h"
#include "jml/utils/check_not_nan.h"


using namespace std;


namespace ML {


/*****************************************************************************/
/* AUTO_ENCODER_STACK                                                        */
/*****************************************************************************/

/** A stack of auto-encoders, that acts as a whole like an auto-encoder. */

Auto_Encoder_Stack::
Auto_Encoder_Stack()
{
}

Auto_Encoder_Stack::
Auto_Encoder_Stack(const std::string & name)
    : layers_(name)
{
}

Auto_Encoder_Stack::
Auto_Encoder_Stack(const Auto_Encoder_Stack & other, Deep_Copy_Tag)
    : Auto_Encoder(other),
      layers_(other.layers_, Deep_Copy_Tag())
{
    update_parameters();
}

void
Auto_Encoder_Stack::
swap(Auto_Encoder_Stack & other)
{
    Auto_Encoder::swap(other);
    layers_.swap(other.layers_);
}

void
Auto_Encoder_Stack::
clear()
{
    layers_.clear();
    update_parameters();
}

void
Auto_Encoder_Stack::
add(Auto_Encoder * layer)
{
    layers_.add(layer);
    Layer::init(name(), layers_.inputs(), layers_.outputs());
    update_parameters();
}

void
Auto_Encoder_Stack::
add(std::shared_ptr<Auto_Encoder> layer)
{
    layers_.add(layer);
    Layer::init(name(), layers_.inputs(), layers_.outputs());
    update_parameters();
}

std::string
Auto_Encoder_Stack::
print() const
{
    return layers_.print();
}

std::string
Auto_Encoder_Stack::
class_id() const
{
    return "Auto_Encoder_Stack";
}

void
Auto_Encoder_Stack::
serialize(DB::Store_Writer & store) const
{
    store << (char)1 // version
          << layers_;
}

void
Auto_Encoder_Stack::
reconstitute(DB::Store_Reader & store)
{
    char version;
    store >> version;
    if (version != 1)
        throw Exception("Auto_Encoder_Stack::reconstitute(): invalid version");

    store >> layers_;

    //cerr << "reconstituted auto encoder stack; layers_.inputs() = "
    //     << layers_.inputs() << endl;

    Layer::init(name(), layers_.inputs(), layers_.outputs());
    update_parameters();
}

void
Auto_Encoder_Stack::
add_parameters(Parameters & params)
{
    layers_.add_parameters(params);
}

void
Auto_Encoder_Stack::
apply(const float * input, float * output) const
{
    return layers_.apply(input, output);
}

void
Auto_Encoder_Stack::
apply(const double * input, double * output) const
{
    return layers_.apply(input, output);
}

size_t
Auto_Encoder_Stack::
fprop_temporary_space_required() const
{
    return layers_.fprop_temporary_space_required();
}

void
Auto_Encoder_Stack::
fprop(const float * inputs,
      float * temp_space, size_t temp_space_size,
      float * outputs) const
{
    layers_.fprop(inputs, temp_space, temp_space_size, outputs);
}

void
Auto_Encoder_Stack::
fprop(const double * inputs,
      double * temp_space, size_t temp_space_size,
      double * outputs) const
{
    layers_.fprop(inputs, temp_space, temp_space_size, outputs);
}

void
Auto_Encoder_Stack::
bprop(const float * inputs,
      const float * outputs,
      const float * temp_space, size_t temp_space_size,
      const float * output_errors,
      float * input_errors,
      Parameters & gradient,
      double example_weight) const
{
    layers_.bprop(inputs, outputs, temp_space, temp_space_size,
                  output_errors, input_errors, gradient, example_weight);
}

void
Auto_Encoder_Stack::
bprop(const double * inputs,
      const double * outputs,
      const double * temp_space, size_t temp_space_size,
      const double * output_errors,
      double * input_errors,
      Parameters & gradient,
      double example_weight) const
{
    layers_.bprop(inputs, outputs, temp_space, temp_space_size,
                  output_errors, input_errors, gradient, example_weight);
}

std::pair<float, float>
Auto_Encoder_Stack::
itargets(float maximum) const
{
    if (layers_.empty())
        throw Exception("itargets(): no layers");
    return layers_.back().itargets(maximum);
}

bool
Auto_Encoder_Stack::
supports_missing_outputs() const
{
    if (layers_.empty())
        throw Exception("itargets(): no layers");
    return layers_.back().supports_missing_outputs();
}

void
Auto_Encoder_Stack::
iapply(const float * output, float * input) const
{
    iapply<float>(output, input);
}

void
Auto_Encoder_Stack::
iapply(const double * output, double * input) const
{
    iapply<double>(output, input);
}

template<typename F>
void
Auto_Encoder_Stack::
iapply(const F * output, F * input) const
{
    F tmp[layers_.max_internal_width()];

    for (int l = layers_.size() - 1;  l >= 0;  --l) {
        F * i = (l == 0 ? input : tmp);
        const F * o = (l == layers_.size() - 1 ? output : tmp);

        layers_[l].iapply(o, i);
    }
}

size_t
Auto_Encoder_Stack::
ifprop_temporary_space_required() const
{
    // We need: inputs of all layers except the first,
    // temporary space of all layers in between.
    //
    // +-----------+-------+-------------+--------+---...
    // |   l0 tmp  | l0 out|  l1 tmp     |  l1 out| l2 tmp
    // +-----------+-------+-------------+--------+---...

    size_t result = 0;

    for (unsigned i = 0;  i < size();  ++i) {
        if (i != 0) result += layers_[i].inputs();
        result += layers_[i].ifprop_temporary_space_required();
    }

    return result;
}

void
Auto_Encoder_Stack::
ifprop(const float * outputs,
       float * temp_space, size_t temp_space_size,
       float * inputs) const
{
    ifprop<float>(outputs, temp_space, temp_space_size, inputs);
}

void
Auto_Encoder_Stack::
ifprop(const double * outputs,
       double * temp_space, size_t temp_space_size,
       double * inputs) const
{
    ifprop<double>(outputs, temp_space, temp_space_size, inputs);
}

template<typename F>
void
Auto_Encoder_Stack::
ifprop(const F * outputs,
       F * temp_space, size_t temp_space_size,
       F * inputs) const
{
    F * temp_space_start = temp_space;
    F * temp_space_end = temp_space_start + temp_space_size;

    const F * curr_outputs = outputs;

    for (int i = size() - 1;  i >= 0;  --i) {
        int layer_temp_space_size
            = layers_[i].ifprop_temporary_space_required();

        F * curr_inputs
            = (i == 0
               ? inputs
               : temp_space + layer_temp_space_size);
        
        layers_[i].ifprop(curr_outputs, temp_space, layer_temp_space_size,
                           curr_inputs);

        curr_outputs = curr_inputs;

        temp_space += layer_temp_space_size;
        if (i != 0) temp_space += layers_[i].inputs();

        if (temp_space > temp_space_end
            || (i == 0 && temp_space != temp_space_end))
            throw Exception("temp space out of sync");
    }
}

void
Auto_Encoder_Stack::
ibprop(const float * outputs,
       const float * inputs,
       const float * temp_space, size_t temp_space_size,
       const float * input_errors,
       float * output_errors,
       Parameters & gradient,
       double example_weight) const
{
    ibprop<float>(outputs, inputs, temp_space, temp_space_size,
                  input_errors, output_errors, gradient, example_weight);
}

void
Auto_Encoder_Stack::
ibprop(const double * outputs,
       const double * inputs,
       const double * temp_space, size_t temp_space_size,
       const double * input_errors,
       double * output_errors,
       Parameters & gradient,
       double example_weight) const
{
    ibprop<double>(outputs, inputs, temp_space, temp_space_size,
                   input_errors, output_errors, gradient, example_weight);
}

template<typename F>
void
Auto_Encoder_Stack::
ibprop(const F * outputs,
       const F * inputs,
       const F * temp_space, size_t temp_space_size,
       const F * input_errors,
       F * output_errors,
       Parameters & gradient,
       double example_weight) const
{
    int ni = this->inputs(), no = this->outputs();

    CHECK_NOT_NAN_N(outputs, no);
    CHECK_NOT_NAN_N(inputs, ni);
    CHECK_NOT_NAN_N(input_errors, ni);
    
    const F * temp_space_start = temp_space;
    const F * temp_space_end = temp_space_start + temp_space_size;
    const F * curr_temp_space = temp_space_end;

    const F * curr_inputs = inputs;

    // Storage for the errors kept between the layers
    F error_storage[layers_.max_internal_width()];

    for (int i = 0;  i < layers_.size(); ++i) {
        int layer_temp_space_size
            = layers_[i].ifprop_temporary_space_required();

        curr_temp_space -= layer_temp_space_size;

        const F * curr_outputs
            = (i == size() - 1
               ? outputs
               : curr_temp_space - layers_[i].outputs());

        const F * curr_input_errors
            = (i == 0 ? input_errors : error_storage);
        
        F * curr_output_errors
            = (i == size() - 1 ? output_errors : error_storage);

        //cerr << "i = " << i << endl;

        layers_[i].ibprop(curr_outputs, curr_inputs, curr_temp_space,
                          layer_temp_space_size, curr_input_errors,
                          curr_output_errors,
                          gradient.subparams(i, layers_[i].name()),
                          example_weight);

        if (curr_temp_space < temp_space_start)
            throw Exception("Layer temp space was out of sync");

        curr_inputs = curr_outputs;
        if (i != layers_.size() - 1) curr_temp_space -= layers_[i].outputs();
    }
    
    if (curr_temp_space != temp_space_start)
        throw Exception("Layer_Stack::bprop(): out of sync");
}

void
Auto_Encoder_Stack::
random_fill(float limit, Thread_Context & context)
{
    layers_.random_fill(limit, context);
}

void
Auto_Encoder_Stack::
zero_fill()
{
    layers_.zero_fill();
}

size_t
Auto_Encoder_Stack::
parameter_count() const
{
    return layers_.parameter_count();
}

std::pair<float, float>
Auto_Encoder_Stack::
targets(float maximum) const
{
    return layers_.targets(maximum);
}

bool
Auto_Encoder_Stack::
supports_missing_inputs() const
{
    return layers_.supports_missing_inputs();
}

Auto_Encoder_Stack *
Auto_Encoder_Stack::
make_copy() const
{
    return new Auto_Encoder_Stack(*this);
}

Auto_Encoder_Stack *
Auto_Encoder_Stack::
deep_copy() const
{
    return new Auto_Encoder_Stack(*this, Deep_Copy_Tag());
}

bool
Auto_Encoder_Stack::
equal_impl(const Layer & other) const
{
    if (typeid(*this) != typeid(other)) return false;
    return layers_.equal_impl(other);
}

bool
Auto_Encoder_Stack::
operator == (const Auto_Encoder_Stack & other) const
{
    return layers_.operator == (other.layers_);
}

namespace {

Register_Factory<Layer, Auto_Encoder_Stack>
AES_REGISTER("Auto_Encoder_Stack");

} // file scope

template class Layer_Stack<Auto_Encoder>;


} // namespace ML
