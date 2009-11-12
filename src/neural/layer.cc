/* layer.cc
   Jeremy Barnes, 20 October 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

*/

#include "layer.h"
#include "db/persistent.h"
#include "arch/demangle.h"
#include "algebra/matrix_ops.h"
#include "arch/simd_vector.h"
#include "boosting/registry.h"
#include "utils/string_functions.h"


using namespace std;
using namespace ML::DB;

namespace ML {


/*****************************************************************************/
/* LAYER                                                                     */
/*****************************************************************************/

/** A basic layer of a neural network.  Other kinds of layers can be built on
    this base.
*/

Layer::
Layer(const std::string & name,
      size_t inputs, size_t outputs)
    : name_(name), inputs_(inputs), outputs_(outputs), parameters_(name)
{
    // Parameters need to be set up in the constructor of the derived class
}

Layer::
Layer(const Layer & other)
    : name_(other.name_), inputs_(other.inputs_), outputs_(other.outputs_),
      parameters_(other.name_)
{
    // We don't copy the parameters; they need to be added in the constructor
    // of the derived class
}

Layer &
Layer::
operator = (const Layer & other)
{
    if (&other == this) return *this;
    name_ = other.name_;
    inputs_ = other.inputs_;
    outputs_ = other.outputs_;

    return *this;
}

void
Layer::
init(const std::string & name, size_t inputs, size_t outputs)
{
    name_ = name;
    inputs_ = inputs;
    outputs_ = outputs;
    parameters_.clear();
    parameters_.set_name(name);
}

void
Layer::
swap(Layer & other)
{
    std::swap(name_, other.name_);
    std::swap(inputs_, other.inputs_);
    std::swap(outputs_, other.outputs_);
    std::swap(parameters_, other.parameters_);
}

bool
Layer::
supports_missing_inputs() const
{
    return false;
}

bool
Layer::
operator == (const Layer & other) const
{
    return name_ == other.name_
        && inputs_ == other.inputs_
        && outputs_ == other.outputs_;
}

bool
Layer::
equal(const Layer & other) const
{
    if (&other == this) return true;
    if (typeid(*this) != typeid(other)) return false;
    
    return equal_impl(other);
}

void
Layer::
update_parameters()
{
    parameters_.clear();
    add_parameters(parameters_);
}

void
Layer::
poly_serialize(ML::DB::Store_Writer & store) const
{
    Registry<Layer>::singleton().serialize(store, this);
}

boost::shared_ptr<Layer>
Layer::
poly_reconstitute(ML::DB::Store_Reader & store)
{
    return Registry<Layer>::singleton().reconstitute(store);
}

distribution<float>
Layer::
apply(const distribution<float> & input) const
{
    if (input.size() != inputs())
        throw Exception("Layer::apply(): invalid number of inputs");
    distribution<float> result(outputs());
    apply(&input[0], &result[0]);
    return result;
}

distribution<double>
Layer::
apply(const distribution<double> & input) const
{
    if (input.size() != inputs())
        throw Exception("Layer::apply(): invalid number of inputs");
    distribution<double> result(outputs());
    apply(&input[0], &result[0]);
    return result;
}
        
void
Layer::
apply(const distribution<float> & input,
      distribution<float> & output) const
{
    if (input.size() != inputs())
        throw Exception("Layer::apply(): invalid number of inputs");
    output.resize(outputs());
    apply(&input[0], &output[0]);
}

void
Layer::
apply(const distribution<double> & input,
      distribution<double> & output) const
{
    if (input.size() != inputs())
        throw Exception("Layer::apply(): invalid number of inputs");
    output.resize(outputs());
    apply(&input[0], &output[0]);
}

#define CHECK_SIZE_OF(element, expected_size) \
    if (element.size() != expected_size) \
        throw Exception(format("%s: Input parameter %s of expected size " \
                               "%s (%d) had real size %d",              \
                               __PRETTY_FUNCTION__, #element, #expected_size, \
                               (int)expected_size, (int)element.size()));

distribution<float>
Layer::
fprop(const distribution<float> & inputs,
      float * temp_space,
      size_t temp_space_size) const
{
    CHECK_SIZE_OF(inputs, this->inputs());
    distribution<float> outputs(this->outputs());
    fprop(&inputs[0], temp_space, temp_space_size, &outputs[0]);
    return outputs;
}

distribution<double>
Layer::
fprop(const distribution<double> & inputs,
      double * temp_space,
      size_t temp_space_size) const
{
    CHECK_SIZE_OF(inputs, this->inputs());
    distribution<double> outputs(this->outputs());
    fprop(&inputs[0], temp_space, temp_space_size, &outputs[0]);
    return outputs;
}

template<typename F>
distribution<F>
Layer::
bprop(const distribution<F> & inputs,
      const distribution<F> & outputs,
      const F * temp_space, size_t temp_space_size,
      const distribution<F> & output_errors,
      Parameters & gradient,
      double example_weight) const
{
    CHECK_SIZE_OF(inputs, this->inputs());
    CHECK_SIZE_OF(outputs, this->outputs());
    CHECK_SIZE_OF(output_errors, this->outputs());
    distribution<F> input_errors(this->inputs());
    bprop(&inputs[0], &outputs[0], temp_space, temp_space_size,
          &output_errors[0], &input_errors[0], gradient, example_weight);
    return input_errors;
}

distribution<float>
Layer::
bprop(const distribution<float> & inputs,
      const distribution<float> & outputs,
      const float * temp_space, size_t temp_space_size,
      const distribution<float> & output_errors,
      Parameters & gradient,
      double example_weight) const
{
    return bprop<float>(inputs, outputs, temp_space, temp_space_size,
                        output_errors, gradient, example_weight);
}

distribution<double>
Layer::
bprop(const distribution<double> & inputs,
      const distribution<double> & outputs,
      const double * temp_space, size_t temp_space_size,
      const distribution<double> & output_errors,
      Parameters & gradient,
      double example_weight) const
{
    return bprop<double>(inputs, outputs, temp_space, temp_space_size,
                         output_errors, gradient, example_weight);
}

void
Layer::
validate() const
{
    if (name_ == "")
        throw Exception("Layer has empty name");
    // Default does none
}

} // namespace ML
