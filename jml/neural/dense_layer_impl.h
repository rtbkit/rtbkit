/* dense_layer_impl.h                                              -*- C++ -*-
   Jeremy Barnes, 5 November 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

   Implementation of dense layer class.
*/

#ifndef __jml__neural__dense_layer_impl_h__
#define __jml__neural__dense_layer_impl_h__

#include "dense_layer.h"
#include "jml/db/persistent.h"
#include "jml/arch/demangle.h"
#include "jml/algebra/matrix_ops.h"
#include "jml/arch/simd_vector.h"
#include "jml/utils/string_functions.h"
#include "jml/boosting/registry.h"
#include "jml/utils/multi_array_utils.h"
#include "jml/stats/distribution_ops.h"
#include "jml/stats/distribution_simd.h"


namespace ML {


/*****************************************************************************/
/* DENSE_LAYER                                                               */
/*****************************************************************************/

/** A simple one way layer with dense connections. */

template<typename Float>
Dense_Layer<Float>::
Dense_Layer()
    : Layer("", 0, 0)
{
}

template<typename Float>
Dense_Layer<Float>::
Dense_Layer(const std::string & name,
            size_t inputs, size_t units,
            Transfer_Function_Type transfer_function,
            Missing_Values missing_values)
    : Layer(name, inputs, units),
      missing_values(missing_values),
      weights(boost::extents[inputs][units]), bias(units)
{
    switch (missing_values) {
    case MV_NONE:
    case MV_ZERO:
        break;
    case MV_INPUT:
        missing_replacements.resize(inputs);
        break;
    case MV_DENSE:
        missing_activations.resize(boost::extents[inputs][units]);
        break;
    default:
        throw Exception("Dense_Layer: invalid missing values");
    }
        
    this->transfer_function = create_transfer_function(transfer_function);

    zero_fill();

    update_parameters();
}

template<typename Float>
Dense_Layer<Float>::
Dense_Layer(const std::string & name,
            size_t inputs, size_t units,
            Transfer_Function_Type transfer_function,
            Missing_Values missing_values,
            Thread_Context & thread_context,
            float limit)
    : Layer(name, inputs, units),
      missing_values(missing_values),
      weights(boost::extents[inputs][units]), bias(units)
{
    switch (missing_values) {
    case MV_NONE:
    case MV_ZERO:
        break;
    case MV_INPUT:
        missing_replacements.resize(inputs);
        break;
    case MV_DENSE:
        missing_activations.resize(boost::extents[inputs][units]);
        break;
    default:
        throw Exception("Dense_Layer: invalid missing values");
    }

    this->transfer_function = create_transfer_function(transfer_function);

    if (limit == -1.0)
        limit = 1.0 / sqrt(inputs);
    random_fill(limit, thread_context);

    update_parameters();
}

template<typename Float>
Dense_Layer<Float>::
Dense_Layer(const Dense_Layer & other)
    : Layer(other),
      transfer_function(other.transfer_function),
      missing_values(other.missing_values),
      weights(other.weights),
      bias(other.bias),
      missing_replacements(other.missing_replacements),
      missing_activations(other.missing_activations)
{
    update_parameters();
}

template<typename Float>
Dense_Layer<Float> &
Dense_Layer<Float>::
operator = (const Dense_Layer & other)
{
    if (&other == this) return *this;
    Dense_Layer new_me(other);
    swap(new_me);
    return *this;
}

template<typename Float>
void
Dense_Layer<Float>::
swap(Dense_Layer & other)
{
    Layer::swap(other);
    transfer_function.swap(other.transfer_function);
    std::swap(missing_values, other.missing_values);
    boost::swap(weights, other.weights);
    bias.swap(other.bias);
    missing_replacements.swap(other.missing_replacements);
    boost::swap(missing_activations, other.missing_activations);
}

template<typename Float>
const Transfer_Function &
Dense_Layer<Float>::
transfer() const
{
    return *transfer_function;
}

template<typename Float>
void
Dense_Layer<Float>::
add_parameters(Parameters & params)
{
    params
        .add(0, "weights", weights)
        .add(1, "bias", bias)
        .add(2, "missing_replacements", missing_replacements)
        .add(3, "missing_activations", missing_activations);
}

template<typename Float>
std::string
Dense_Layer<Float>::
print() const
{
    size_t ni = inputs(), no = outputs();
    std::string result
        = format("{ layer: %zd inputs, %zd neurons, function %s, missing %s\n",
                 inputs(), outputs(), transfer_function->print().c_str(),
                 ML::print(missing_values).c_str());

    result += "  weights: \n";
    for (unsigned i = 0;  i < ni;  ++i) {
        result += "    [ ";
        for (unsigned j = 0;  j < no;  ++j)
            result += format("%8.4f", weights[i][j]);
        result += " ]\n";
    }

    result += "  bias: \n    [ ";
    for (unsigned j = 0;  j < no;  ++j)
        result += format("%8.4f", bias[j]);
    result += " ]\n";
    
    if (missing_values == MV_INPUT) {
        result += "  missing replacements: \n    [ ";
        for (unsigned j = 0;  j < no;  ++j)
            result += format("%8.4f", missing_replacements[j]);
        result += " ]\n";
    }

    if (missing_values == MV_DENSE) {
        result += "  missing activations: \n";
        for (unsigned i = 0;  i < ni;  ++i) {
            result += "    [ ";
            for (unsigned j = 0;  j < no;  ++j)
                result += format("%8.4f", missing_activations[i][j]);
            result += " ]\n";
        }
    }
    
    result += "}\n";
    
    return result;
}

template<typename Float>
std::string
Dense_Layer<Float>::
class_id() const
{
    return "Dense_Layer<" + demangle(typeid(Float).name()) + ">";
}

template<typename Float>
void
Dense_Layer<Float>::
serialize(DB::Store_Writer & store) const
{
    using namespace DB;
    store << compact_size_t(2);
    store << std::string("DENSE LAYER");
    store << this->name_;
    store << compact_size_t(inputs());
    store << compact_size_t(outputs());
    store << missing_values;
    store << weights;
    store << bias;
    store << missing_replacements;
    store << missing_activations;
    transfer_function->poly_serialize(store);
}

template<typename Float>
void
Dense_Layer<Float>::
reconstitute(DB::Store_Reader & store)
{
    using namespace DB;
    using namespace std;

    compact_size_t version(store);
    //cerr << "version = " << version << endl;
    if (version < 1)
        throw Exception("invalid layer version");

    if (version == 1) {

        std::string s;
        store >> s;
        if (s != "PERCEPTRON LAYER")
            throw Exception("invalid layer start " + s);

        compact_size_t inputs_read(store), outputs_read(store);
        
        store >> weights;
        store >> bias;
        store >> missing_replacements;
        Transfer_Function_Type transfer_function_type;
        store >> transfer_function_type;
        
        transfer_function = create_transfer_function(transfer_function_type);

        if (inputs_read != inputs() || outputs_read != outputs())
            throw Exception("inputs read weren't right");
        
        if (weights.shape()[0] != inputs_read)
            throw Exception("weights has wrong shape");
        if (weights.shape()[1] != outputs_read)
            throw Exception("weights has wrong output shape");
        if (bias.size() != outputs_read) {
            cerr << "bias.size() = " << bias.size() << endl;
            cerr << "outputs_read = " << outputs_read << endl;
            throw Exception("bias is wrong size");
        }
        if (missing_replacements.size() != inputs_read)
            throw Exception("missing replacements are wrong size");
    }
    else if (version == 2) {
        std::string s;
        store >> s;
        if (s != "DENSE LAYER")
            throw Exception("Not reconstituting a dense layer");

        store >> name_;

        compact_size_t ni(store), no(store);
        inputs_ = ni;
        outputs_ = no;
        
        store >> missing_values >> weights >> bias >> missing_replacements
              >> missing_activations;

        transfer_function = Transfer_Function::poly_reconstitute(store);

        validate();
        
        update_parameters();
    }
}

template<typename Float>
void
Dense_Layer<Float>::
apply(const float * input,
      float * output) const
{
    int no = outputs();
    float act[no];
    activation(input, act);
    transfer_function->transfer(act, output, no);
}

template<typename Float>
void
Dense_Layer<Float>::
apply(const double * input,
      double * output) const
{
    int no = outputs();
    double act[no];
    activation(input, act);
    transfer_function->transfer(act, output, no);
}

template<typename Float>
template<class F>
void
Dense_Layer<Float>::
activation(const F * inputs, F * activations) const
{
    int ni = this->inputs(), no = this->outputs();
    double accum[no];  // Accumulate in double precision to improve rounding,
                       // even if we're using floats
    std::copy(bias.begin(), bias.end(), accum);

    for (unsigned i = 0;  i < ni;  ++i) {
        const Float * w;
        double input;
        if (!isnan(inputs[i])) {
            input = inputs[i];
            w = &weights[i][0];
        }
        else {
            switch (missing_values) {
            case MV_NONE:
                throw Exception("missing value with MV_NONE");

            case MV_ZERO:
                continue;  // no need to calculate, since weight is zero
                input = 0.0;  w = &weights[i][0];  break;

            case MV_INPUT:
                input = missing_replacements[i];  w = &weights[i][0];  break;

            case MV_DENSE:
                input = 1.0;  w = &missing_activations[i][0];  break;

            default:
                throw Exception("unknown missing values");
            }
        }

        SIMD::vec_add(accum, input, w, accum, no);
    }
    
    std::copy(accum, accum + no, activations);
}

template<typename Float>
void
Dense_Layer<Float>::
activation(const float * inputs,
           float * activations) const
{
    activation<float>(inputs, activations);
}

template<typename Float>
void
Dense_Layer<Float>::
activation(const double * inputs,
           double * activations) const
{
    activation<double>(inputs, activations);
}

template<typename Float>
distribution<float>
Dense_Layer<Float>::
activation(const distribution<float> & inputs) const
{
    int no = outputs();
    distribution<float> output(no);
    activation(&inputs[0], &output[0]);
    return output;
}

template<typename Float>
distribution<double>
Dense_Layer<Float>::
activation(const distribution<double> & inputs) const
{
    int no = outputs();
    distribution<double> output(no);
    activation(&inputs[0], &output[0]);
    return output;
}

template<typename Float>
size_t
Dense_Layer<Float>::
fprop_temporary_space_required() const
{
    // We don't need anything apart from the input and the output
    // TODO: let the transfer function decide...
    return 0;
}

template<typename Float>
void
Dense_Layer<Float>::
fprop(const float * inputs,
      float * temp_space, size_t temp_space_size,
      float * outputs) const
{
    if (temp_space_size != 0)
        throw Exception("Dense_Layer::fprop(): wrong temp space size");

    apply(inputs, outputs);
}

template<typename Float>
void
Dense_Layer<Float>::
fprop(const double * inputs,
      double * temp_space, size_t temp_space_size,
      double * outputs) const
{
    if (temp_space_size != 0)
        throw Exception("Dense_Layer::fprop(): wrong temp space size");

    apply(inputs, outputs);
}

template<typename Float>
template<typename F>
void
Dense_Layer<Float>::
bprop(const F * inputs,
      const F * outputs,
      const F * temp_space, size_t temp_space_size,
      const F * output_errors,
      F * input_errors,
      Parameters & gradient,
      double example_weight) const
{
    bbprop(inputs, outputs, temp_space, temp_space_size,
           output_errors, 0, input_errors, 0, gradient, 0, example_weight);
}

template<typename Float>
void
Dense_Layer<Float>::
bprop(const float * inputs,
      const float * outputs,
      const float * temp_space, size_t temp_space_size,
      const float * output_errors,
      float * input_errors,
      Parameters & gradient,
      double example_weight) const
{
    bprop<float>(inputs, outputs, temp_space, temp_space_size, output_errors,
                 input_errors, gradient, example_weight);
}

template<typename Float>
void
Dense_Layer<Float>::
bprop(const double * inputs,
      const double * outputs,
      const double * temp_space, size_t temp_space_size,
      const double * output_errors,
      double * input_errors,
      Parameters & gradient,
      double example_weight) const
{
    bprop<double>(inputs, outputs, temp_space, temp_space_size, output_errors,
                  input_errors, gradient, example_weight);
}

namespace {

template<typename F>
F sqr(F val)
{
    return val * val;
}

} // file scope

template<typename Float>
template<typename F>
void
Dense_Layer<Float>::
bbprop(const F * inputs,
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

    if (temp_space_size != 0)
        throw Exception("Dense_Layer::bprop(): wrong temp size");
    
    // Differentiate the output function
    F derivs[no];
    transfer_function->derivative(outputs, derivs, no);

    F ddbias[no];
    if (dgradient || d2input_errors) {
        // Approximation to the second derivative of the output errors
        for (unsigned o = 0;  o < no;  ++o)
            ddbias[o] = d2output_errors[o] * sqr(derivs[o]);

#if 1 // improve the approximation using the second derivative
        F ddtransfer[no];
        // Second derivative of the output errors
        transfer_function->second_derivative(outputs, ddtransfer, no);
        for (unsigned o = 0;  o < no;  ++o)
            ddbias[o] += outputs[o] * ddtransfer[o] * d2output_errors[o];
#endif // improve the approximation

        // These are the bias errors...
        dgradient->vector(1, "bias").update(ddbias, example_weight);
    }

    // Bias updates are simply derivs in multiplied by transfer deriv
    F dbias[no];
    SIMD::vec_prod(derivs, &output_errors[0], dbias, no);
    gradient.vector(1, "bias").update(dbias, example_weight);

    Matrix_Parameter & dweights = gradient.matrix(0, "weights");
    Matrix_Parameter * ddweights = 0;
    if (dgradient)
        ddweights = &dgradient->matrix(0, "weights");

    for (unsigned i = 0;  i < ni;  ++i) {
        bool was_missing = isnan(inputs[i]);
        if (input_errors) input_errors[i] = 0.0;

        if (!was_missing) {
            if (inputs[i] == 0.0) continue;

            dweights.update_row(i, dbias, inputs[i] * example_weight);

            if (input_errors)
                input_errors[i]
                    = SIMD::vec_dotprod_dp(&weights[i][0],
                                           &dbias[0], no);
            
            if (ddweights)
                ddweights->update_row(i, ddbias,
                                      inputs[i] * inputs[i] * example_weight);
            
            if (d2input_errors)
                d2input_errors[i]
                    = SIMD::vec_accum_prod3(&weights[i][0],
                                            &weights[i][0],
                                            ddbias,
                                            no);
        }
        else if (missing_values == MV_NONE)
            throw Exception("MV_NONE but missing value");
        else if (missing_values == MV_ZERO) {
            // No update as everything is multiplied by zero
        }
        else if (missing_values == MV_DENSE) {
            gradient.matrix(3, "missing_activations")
                .update_row(i, dbias, example_weight);
            if (dgradient)
                dgradient->matrix(3, "missing_activations")
                    .update_row(i, ddbias, example_weight);
        }
        else if (missing_values == MV_INPUT) {
            // Missing

            // Update the weights
            gradient.matrix(0, "weights")
                .update_row(i, dbias,
                            missing_replacements[i] * example_weight);
            
            gradient.vector(2, "missing_replacements")
                .update_element(i,
                                (example_weight
                                 * SIMD::vec_dotprod_dp(&weights[i][0], dbias,
                                                        no)));

            if (dgradient) {
                dgradient->matrix(0, "weights")
                    .update_row(i, ddbias,
                                sqr(missing_replacements[i]) * example_weight);

                dgradient->vector(2, "missing_replacements")
                    .update_element(i,
                                    (example_weight
                                     * SIMD::vec_accum_prod3(&weights[i][0],
                                                             &weights[i][0],
                                                             ddbias,
                                                             no)));
            }
        }
    }
}

template<typename Float>
void
Dense_Layer<Float>::
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
    return bbprop<float>(inputs, outputs, temp_space, temp_space_size,
                         output_errors, d2output_errors, input_errors,
                         d2input_errors, gradient, dgradient,
                         example_weight);
}
 
template<typename Float>
void
Dense_Layer<Float>::
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
    return bbprop<double>(inputs, outputs, temp_space, temp_space_size,
                          output_errors, d2output_errors, input_errors,
                          d2input_errors, gradient, dgradient,
                          example_weight);
}

namespace {

template<typename Float>
void random_fill_range(Float * start, size_t size, float limit,
                       Thread_Context & context)
{
    for (unsigned i = 0;  i < size;  ++i)
        start[i] = limit * (context.random01() * 2.0f - 1.0f);
}

template<typename Float>
void random_fill_range(std::vector<Float> & vec, float limit,
                       Thread_Context & context)
{
    random_fill_range(&vec[0], vec.size(), limit, context);
}

template<typename Float>
void random_fill_range(boost::multi_array<Float, 2> & arr, float limit,
                       Thread_Context & context)
{
    random_fill_range(arr.data(), arr.num_elements(), limit, context);
}

} // file scope

template<typename Float>
void
Dense_Layer<Float>::
random_fill(float limit, Thread_Context & context)
{
    random_fill_range(weights, limit, context);
    random_fill_range(bias, limit, context);

    switch (missing_values) {
    case MV_NONE: break;
    case MV_ZERO: break;
        
    case MV_INPUT:
        random_fill_range(missing_replacements, limit, context);
        break;

    case MV_DENSE:
        random_fill_range(missing_activations, limit, context);
        break;

    default:
        throw Exception("Dense_Layer::random_fill(): unknown missing values");
    }
}

template<typename Float>
void
Dense_Layer<Float>::
zero_fill()
{
    std::fill(weights.data(),
              weights.data() + weights.num_elements(),
              0.0);
    
    bias.fill(0.0);

    switch (missing_values) {
    case MV_NONE: break;
    case MV_ZERO: break;

    case MV_INPUT:
        missing_replacements.fill(0.0);
        break;

    case MV_DENSE:
        std::fill(missing_activations.data(),
                  missing_activations.data()
                      + missing_activations.num_elements(),
                  0.0);
        break;

    default:
        throw Exception("Dense_Layer::random_fill(): unknown missing values");
    }
}

template<typename Float>
void
Dense_Layer<Float>::
validate() const
{
    Layer::validate();

    if (!transfer_function)
        throw Exception("transfer function not implemented");
    
    if (weights.shape()[1] != bias.size())
        throw Exception("perceptron layer has bad shape");

    int ni = weights.shape()[0], no = weights.shape()[1];
    
    bool has_nonzero = false;

    for (unsigned j = 0;  j < no;  ++j) {
        for (unsigned i = 0;  i < ni;  ++i) {
            if (!finite(weights[i][j]))
                throw Exception("perceptron layer has non-finite weights");
            if (weights[i][j] != 0.0)
                has_nonzero = true;
        }
    }
    
    if (!has_nonzero)
        throw Exception("perceptron layer has all zero weights");

    if (no != bias.size())
        throw Exception("bias sized wrong");

    for (unsigned o = 0;  o < bias.size();  ++o)
        if (!finite(bias[o]))
            throw Exception("perceptron layer has non-finite bias");

    switch (missing_values) {
    case MV_ZERO:
    case MV_NONE:
        if (missing_replacements.size() != 0)
            throw Exception("missing replacements should be empty");
        if (missing_activations.num_elements() != 0)
            throw Exception("missing activations should be empty");
        break;
    case MV_INPUT:
        if (ni != missing_replacements.size())
            throw Exception("missing replacements sized wrong");
        for (unsigned i = 0;  i < missing_replacements.size();  ++i)
            if (!finite(missing_replacements[i]))
                throw Exception("perceptron layer has non-finite missing "
                                "replacement");
        if (missing_activations.num_elements() != 0)
            throw Exception("missing activations should be empty");
        break;

    case MV_DENSE:
        if (missing_replacements.size() != 0)
            throw Exception("missing replacements should be empty");
        if (missing_activations.shape()[0] != ni
            || missing_activations.shape()[1] != no) {
            using namespace std;
            cerr << "ni = " << ni << " ni2 = " << missing_activations.shape()[0]
                 << endl;
            cerr << "no = " << no << " no2 = " << missing_activations.shape()[1]
                 << endl;
            throw Exception("missing activations has wrong size");
        }
        for (const Float * p = missing_activations.data(),
                 * pe = missing_activations.data()
                 + missing_activations.num_elements();
             p != pe;  ++p)
            if (isnan(*p))
                throw Exception("missing_activations is not finite");
        break;
    default:
        throw Exception("unknown missing_values " + ML::print(missing_values));
    }
}

template<typename Float>
size_t
Dense_Layer<Float>::
parameter_count() const
{
    return weights.num_elements() + bias.size() + missing_replacements.size()
        + missing_activations.num_elements();
}

template<typename Float>
bool
Dense_Layer<Float>::
supports_missing_inputs() const
{
    return (missing_values != MV_NONE);
}

template<typename Float>
std::pair<float, float>
Dense_Layer<Float>::
targets(float maximum) const
{
    return transfer_function->targets(maximum);
}

template<typename Float>
bool
Dense_Layer<Float>::
equal_impl(const Layer & other) const
{
    const Dense_Layer<Float> & cast
        = dynamic_cast<const Dense_Layer<Float> &>(other);
    return operator == (cast);
}

template<typename Float>
bool
Dense_Layer<Float>::
operator == (const Dense_Layer & other) const
{
#if 0
    using namespace std;

    if (transfer_function && other.transfer_function)
        if (!transfer_function->equal(*other.transfer_function))
            cerr << "transfer function" << endl;
    if (missing_values != other.missing_values)
        cerr << "missing values" << endl;
    if (inputs() != other.inputs())
        cerr << "inputs" << endl;
    if (outputs() != other.outputs())
        cerr << "outputs" << endl;
    if (weights.shape()[0] != other.weights.shape()[0])
        cerr << "weights shape 0" << endl;
    if (weights.shape()[1] != other.weights.shape()[1])
        cerr << "weights shape 1" << endl;
    if (missing_replacements.size() != other.missing_replacements.size())
        cerr << "missing replacements size" << endl;
    if (bias.size() != other.bias.size())
        cerr << "bias size" << endl;
    if (weights != other.weights)
        cerr << "weights" << endl;
    if ((bias.size() != other.bias.size())
        || !(bias == other.bias).all())
        cerr << "bias" << endl;
    if ((missing_replacements.size() != other.missing_replacements.size())
        || !(missing_replacements == other.missing_replacements).all())
        cerr << "missing replacements" << endl;
    if (missing_activations != other.missing_activations)
        cerr << "missing activations" << endl;
#endif

    return (Layer::operator == (other)
            && ((transfer_function && other.transfer_function
                 && transfer_function->equal(*other.transfer_function))
                || (transfer_function == other.transfer_function))
            && missing_values == other.missing_values
            && weights.shape()[0] == other.weights.shape()[0]
            && weights.shape()[1] == other.weights.shape()[1]
            && missing_replacements.size() == other.missing_replacements.size()
            && bias.size() == other.bias.size()
            && weights == other.weights
            && (bias == other.bias).all()
            && (missing_replacements == other.missing_replacements).all()
            && missing_activations == other.missing_activations);
}

template<typename Float>
struct Dense_Layer<Float>::RegisterMe {
    RegisterMe()
    {
        Register_Factory<Layer, Dense_Layer<Float> >
            STF_REGISTER(Dense_Layer<Float>().class_id());
    }
};

template<typename Float>
typename Dense_Layer<Float>::RegisterMe
Dense_Layer<Float>::register_me;

} // namespace ML



#endif /* __jml__neural__dense_layer_impl_h__ */
