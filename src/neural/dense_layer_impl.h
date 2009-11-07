/* dense_layer_impl.h                                              -*- C++ -*-
   Jeremy Barnes, 5 November 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

   Implementation of dense layer class.
*/

#ifndef __jml__neural__dense_layer_impl_h__
#define __jml__neural__dense_layer_impl_h__

#include "dense_layer.h"
#include "db/persistent.h"
#include "arch/demangle.h"
#include "algebra/matrix_ops.h"
#include "arch/simd_vector.h"
#include "utils/string_functions.h"
#include "boosting/registry.h"

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
      weights(boost::extents[inputs][units]), bias(units),
      missing_replacements(inputs)
{
    this->transfer_function = create_transfer_function(transfer_function);
    zero_fill();
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
      weights(boost::extents[inputs][units]), bias(units),
      missing_replacements(inputs)
{
    this->transfer_function = create_transfer_function(transfer_function);
    if (limit == -1.0)
        limit = 1.0 / sqrt(inputs);
    random_fill(limit, thread_context);
}

template<typename Float>
void
Dense_Layer<Float>::
add_parameters(Parameters & params)
{
    params
        .add(0, "weights", weights)
        .add(1, "bias", bias);

    switch (missing_values) {
    case MV_NONE:
    case MV_ZERO:
        break;
    case MV_INPUT:
        params.add(2, "missing_replacements", missing_replacements);
        break;
    case MV_DENSE:
        params.add(2, "missing_activations", missing_activations);
        break;
    default:
        throw Exception("Dense_Layer::parameters(): none there");
    }
}

template<typename Float>
std::string
Dense_Layer<Float>::
print() const
{
    size_t ni = inputs(), no = outputs();
    std::string result
        = format("{ layer: %zd inputs, %zd neurons, function %s\n",
                 inputs(), outputs(), transfer_function->print().c_str());
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
    
    result += "  missing replacements: \n    [ ";
    for (unsigned j = 0;  j < no;  ++j)
        result += format("%8.4f", missing_replacements[j]);
    result += " ]\n";

    result += "  missing activations: \n";
    for (unsigned i = 0;  i < ni;  ++i) {
        result += "    [ ";
        for (unsigned j = 0;  j < no;  ++j)
            result += format("%8.4f", missing_activations[i][j]);
        result += " ]\n";
    }
    return result;

    
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
        
        store >> missing_values >> weights >> bias >> missing_replacements;

        transfer_function = Transfer_Function::poly_reconstitute(store);

        validate();
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
void
Dense_Layer<Float>::
activation(const float * preprocessed,
           float * activation) const
{
    int ni = inputs(), no = outputs();
    double accum[no];  // Accumulate in double precision to improve rounding
    std::copy(bias.begin(), bias.end(), accum);

    for (unsigned i = 0;  i < ni;  ++i) {
        const Float * w;
        double input;
        if (!isnan(preprocessed[i])) {
            input = preprocessed[i];
            w = &weights[i][0];
        }
        else {
            switch (missing_values) {
            case MV_NONE:
                throw Exception("missing value with MV_NONE");

            case MV_ZERO:
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
    
    std::copy(accum, accum + no, activation);
}

template<typename Float>
void
Dense_Layer<Float>::
activation(const double * preprocessed,
           double * activation) const
{
    int ni = inputs(), no = outputs();
    double accum[no];  // Accumulate in double precision to improve rounding
    std::copy(bias.begin(), bias.end(), accum);

    for (unsigned i = 0;  i < ni;  ++i) {
        const Float * w;
        double input;
        if (!isnan(preprocessed[i])) {
            input = preprocessed[i];
            w = &weights[i][0];
        }
        else {
            switch (missing_values) {
            case MV_NONE:
                throw Exception("missing value with MV_NONE");

            case MV_ZERO:
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
    
    std::copy(accum, accum + no, activation);
}

template<typename Float>
distribution<float>
Dense_Layer<Float>::
activation(const distribution<float> & preprocessed) const
{
    int no = outputs();
    distribution<float> output(no);
    activation(&preprocessed[0], &output[0]);
    return output;
}

template<typename Float>
distribution<double>
Dense_Layer<Float>::
activation(const distribution<double> & preprocessed) const
{
    int no = outputs();
    distribution<double> output(no);
    activation(&preprocessed[0], &output[0]);
    return output;
}

template<typename Float>
size_t
Dense_Layer<Float>::
fprop_temporary_space_required() const
{
    // We just need the outputs...
    return inputs() + outputs();
}

template<typename Float>
distribution<float>
Dense_Layer<Float>::
fprop(const distribution<float> & inputs,
      float * temp_space, size_t temp_space_size) const
{
    int ni = this->inputs(), no = this->outputs();

    if (temp_space_size != ni + no)
        throw Exception("Dense_Layer::fprop(): wrong size");
    distribution<float> result = apply(inputs);
    std::copy(inputs.begin(), inputs.end(), temp_space);
    std::copy(result.begin(), result.end(), temp_space + ni);
    return result;
}

template<typename Float>
distribution<double>
Dense_Layer<Float>::
fprop(const distribution<double> & inputs,
      double * temp_space, size_t temp_space_size) const
{
    int ni = this->inputs(), no = this->outputs();

    if (temp_space_size != ni + no)
        throw Exception("Dense_Layer::fprop(): wrong size");
    distribution<double> result = apply(inputs);
    std::copy(inputs.begin(), inputs.end(), temp_space);
    std::copy(result.begin(), result.end(), temp_space + ni);
    return result;
}

template<typename Float>
void
Dense_Layer<Float>::
bprop(const distribution<float> & output_errors,
      float * temp_space, size_t temp_space_size,
      Parameters & gradient,
      distribution<float> & input_errors,
      double example_weight,
      bool calc_input_errors) const
{
    if (temp_space_size != outputs())
        throw Exception("Dense_Layer::bprop(): wrong temp size");

    int ni = this->inputs(), no = this->outputs();

    const float * inputs = temp_space;
    const float * outputs = temp_space + ni;

    // Differentiate the output function
    float derivs[no];
    transfer_function->derivative(outputs, derivs, no);

    // Bias updates are simply derivs in multiplied by transfer deriv
    float dbias[no];
    SIMD::vec_prod(derivs, &output_errors[0], dbias, no);

    gradient.vector(0, "bias").update(dbias, example_weight);

    if (calc_input_errors) input_errors.resize(ni);

    for (unsigned i = 0;  i < ni;  ++i) {
        bool was_missing = isnan(inputs[i]);
        if (calc_input_errors) input_errors[i] = 0;
        
        if (!was_missing) {
            gradient.matrix(1, "weights")
                .update_row(i, dbias, inputs[i] * example_weight);

            if (calc_input_errors)
                input_errors[i]
                    = SIMD::vec_dotprod_dp(&weights[i][0],
                                           &dbias[0], no);
        }
        else if (missing_values == MV_NONE)
            throw Exception("MV_NONE but missing value");
        else if (missing_values == MV_ZERO) {
            // No update as everything is multiplied by zero
        }
        else if (missing_values == MV_DENSE) {
            gradient.matrix(2, "missing_activations")
                .update_row(i, dbias, example_weight);
        }
        else if (missing_values == MV_INPUT) {
            // Missing

            // Update the weights
            gradient.matrix(1, "weights")
                .update_row(i, dbias,
                            missing_replacements[i] * example_weight);
            
            gradient.vector(2, "missing_replacements")
                .update_element(i,
                                (example_weight
                                 * SIMD::vec_dotprod_dp(&weights[i][0], dbias,
                                                        no)));
        }
    }
}

template<typename Float>
void
Dense_Layer<Float>::
bprop(const distribution<double> & output_errors,
      double * temp_space, size_t temp_space_size,
      Parameters & gradient,
      distribution<double> & input_errors,
      double example_weight,
      bool calculate_input_errors) const
{
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
    if (weights.shape()[1] != bias.size())
        throw Exception("perceptron laye has bad shape");

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

    if (ni != missing_replacements.size())
        throw Exception("missing replacements sized wrong");

    for (unsigned i = 0;  i < missing_replacements.size();  ++i)
        if (!finite(missing_replacements[i]))
            throw Exception("perceptron layer has non-finite missing replacement");
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
std::pair<float, float>
Dense_Layer<Float>::
targets(float maximum) const
{
    return transfer_function->targets(maximum);
}

template<typename Float>
bool
Dense_Layer<Float>::
operator == (const Dense_Layer & other) const
{
    return (transfer_function == other.transfer_function
            && missing_values == other.missing_values
            && inputs() == other.inputs()
            && outputs() == other.outputs()
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
