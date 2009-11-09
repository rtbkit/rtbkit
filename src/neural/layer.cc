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

void
Layer::
validate() const
{
    if (name_ == "")
        throw Exception("Layer has empty name");
    // Default does none
}

} // namespace ML
