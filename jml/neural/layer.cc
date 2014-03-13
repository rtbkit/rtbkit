/* layer.cc
   Jeremy Barnes, 20 October 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

*/

#include "layer.h"
#include "jml/db/persistent.h"
#include "jml/arch/demangle.h"
#include "jml/algebra/matrix_ops.h"
#include "jml/arch/simd_vector.h"
#include "jml/boosting/registry.h"
#include "jml/utils/string_functions.h"


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

const Transfer_Function &
Layer::
transfer() const
{
    throw Exception("layer " + type_name(*this)
                    + " doesn't have a transfer function");
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

std::shared_ptr<Layer>
Layer::
poly_reconstitute(ML::DB::Store_Reader & store)
{
    return Registry<Layer>::singleton().reconstitute(store);
}

distribution<float>
Layer::
apply(const distribution<float> & input) const
{
    if (input.size() != inputs()) {
        cerr << "input.size() = " << input.size() << endl;
        cerr << "inputs() = " << inputs() << endl;
        throw Exception("Layer::apply(): invalid number of inputs");
    }
    distribution<float> result(outputs());
    apply(&input[0], &result[0]);
    return result;
}

distribution<double>
Layer::
apply(const distribution<double> & input) const
{
    if (input.size() != inputs()) {
        cerr << "input.size() = " << input.size() << endl;
        cerr << "inputs() = " << inputs() << endl;
        throw Exception("Layer::apply(): invalid number of inputs");
    }
    distribution<double> result(outputs());
    apply(&input[0], &result[0]);
    return result;
}
        
void
Layer::
apply(const distribution<float> & input,
      distribution<float> & output) const
{
    if (input.size() != inputs()) {
        cerr << "input.size() = " << input.size() << endl;
        cerr << "inputs() = " << inputs() << endl;
        throw Exception("Layer::apply(): invalid number of inputs");
    }
    output.resize(outputs());
    apply(&input[0], &output[0]);
}

void
Layer::
apply(const distribution<double> & input,
      distribution<double> & output) const
{
    if (input.size() != inputs()) {
        cerr << "input.size() = " << input.size() << endl;
        cerr << "inputs() = " << inputs() << endl;
        throw Exception("Layer::apply(): invalid number of inputs");
    }
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

namespace {

template<typename F>
F sqr(F val)
{
    return val * val;
}

} // file scope

template<typename F>
void
Layer::
bbprop_jacobian(const F * inputs,
                const F * outputs,
                const F * temp_space, size_t temp_space_size,
                const F * output_errors,
                const F * d2output_errors,
                F * input_errors,
                F * d2input_errors,
                Parameters & gradient,
                Parameters * dgradient,
                double example_weight) const
{
    int ni = this->inputs(), no = this->outputs();

    // 1.  Perform the real bprop to calculate the bprop parameters
    bprop(inputs, outputs, temp_space, temp_space_size, output_errors,
          input_errors, gradient, example_weight);

    if (dgradient == 0 && d2input_errors == 0) return;
    
    // 2.  We need to call bprop() outputs() times, once for each of the
    // parameters, in order to get the derivative of each output with
    // respect to each parameter

    Parameters_Copy<double> gradient_k(*this, 0.0);

    F output_select[no];
    std::fill(output_select, output_select + no,  0.0);

    distribution<F> input_errors_k(ni);
    distribution<double> d2input_errors_accum(ni);

    //cerr << "inputs  = " << distribution<float>(inputs, inputs + no) << endl;
    //cerr << "outputs = " << distribution<float>(outputs, outputs + no) << endl;
    //cerr << "output_errors = " << distribution<float>(output_errors, output_errors + no) << endl;
    //cerr << "d2output_errors = " << distribution<float>(d2output_errors, d2output_errors + no) << endl;

    //cerr << "layer = " << this->print() << endl;

    for (unsigned o = 0;  o < no;  output_select[o] = 0.0, ++o) {

        if (d2output_errors[o] == 0.0) continue;
        
        gradient_k.values.fill(0.0);

        output_select[o] = 1.0;

        bprop(inputs, outputs, temp_space, temp_space_size, output_select,
              &input_errors_k[0], gradient_k, 1.0);
        
        //cerr << "input_errors_k = " << input_errors_k << endl;
        //cerr << "gradient_k = " << gradient_k.values << endl;

        // See LeCun et al
        // d2E/dwi2 ~= sum(k) d2E/do_k2 (do_k/dwi)^2
        
        //cerr << "k = " << d2output_errors[o] * example_weight << endl;
        
        if (dgradient)
            dgradient->update_sqr(gradient_k,
                                  d2output_errors[o] * example_weight);

        Parameters_Copy<double> dgcopy(*dgradient);
        //cerr << "dgradient->values = " << dgcopy.values << endl;
        
        if (d2input_errors)
            d2input_errors_accum += d2output_errors[o] * sqr(input_errors_k);
    }

    if (d2input_errors)
        std::copy(d2input_errors_accum.begin(), d2input_errors_accum.end(),
                  d2input_errors);
}

void
Layer::
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
    return bbprop_jacobian<float>(inputs, outputs, temp_space, temp_space_size,
                                  output_errors, d2output_errors, input_errors,
                                  d2input_errors, gradient, dgradient,
                                  example_weight);
}
 
void
Layer::
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
    return bbprop_jacobian<double>(inputs, outputs, temp_space, temp_space_size,
                                   output_errors, d2output_errors, input_errors,
                                   d2input_errors, gradient, dgradient,
                                   example_weight);
}

void
Layer::
validate() const
{
    //if (name_ == "")
    //    throw Exception("Layer has empty name");
    // Default does none
}

} // namespace ML
