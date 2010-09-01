/* dense_shared_reverse_layer_impl.h                               -*- C++ -*-
   Jeremy Barnes, 9 November 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

   Implementation of methods for the dense shared reverse layer.
*/

#ifndef __jml__neural__dense_shared_reverse_layer_impl_h__
#define __jml__neural__dense_shared_reverse_layer_impl_h__

#include "dense_shared_reverse_layer.h"
#include "jml/db/persistent.h"
#include "jml/arch/demangle.h"
#include "jml/algebra/matrix_ops.h"
#include "jml/arch/simd_vector.h"
#include "jml/utils/string_functions.h"
#include "jml/boosting/registry.h"
#include "jml/utils/multi_array_utils.h"

namespace ML {


/*****************************************************************************/
/* DENSE_LAYER                                                               */
/*****************************************************************************/

/** A simple one way layer with dense connections. */

template<typename Float>
Dense_Shared_Reverse_Layer<Float>::
Dense_Shared_Reverse_Layer()
    : Layer("", 0, 0), forward(0)
{
}

template<typename Float>
Dense_Shared_Reverse_Layer<Float>::
Dense_Shared_Reverse_Layer(const std::string & name,
                           Dense_Layer<Float> * forward)
    : Layer(name, forward->outputs(), forward->inputs()),
      ibias(forward->inputs()),
      iscales(forward->inputs()),
      hscales(forward->outputs()),
      forward(forward)
{
}

template<typename Float>
Dense_Shared_Reverse_Layer<Float>::
Dense_Shared_Reverse_Layer(const std::string & name,
                           Dense_Layer<Float> * forward,
                           Thread_Context & thread_context,
                           float limit)
    : Layer(name, forward->outputs(), forward->inputs()),
      ibias(forward->inputs()),
      iscales(forward->inputs()),
      hscales(forward->outputs()),
      forward(forward)
{
    if (limit == -1.0)
        limit = 1.0 / sqrt(inputs());
    random_fill(limit, thread_context);
}

template<typename Float>
Dense_Shared_Reverse_Layer<Float> &
Dense_Shared_Reverse_Layer<Float>::
operator = (const Dense_Shared_Reverse_Layer & other)
{
    if (&other == this) return *this;
    Dense_Shared_Reverse_Layer new_me(other);
    swap(new_me);
    return *this;
}

template<typename Float>
void
Dense_Shared_Reverse_Layer<Float>::
swap(Dense_Shared_Reverse_Layer & other)
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
void
Dense_Shared_Reverse_Layer<Float>::
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
        throw Exception("Dense_Shared_Reverse_Layer::parameters(): none there");
    }
}

template<typename Float>
std::string
Dense_Shared_Reverse_Layer<Float>::
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
Dense_Shared_Reverse_Layer<Float>::
class_id() const
{
    return "Dense_Shared_Reverse_Layer<" + demangle(typeid(Float).name()) + ">";
}

template<typename Float>
void
Dense_Shared_Reverse_Layer<Float>::
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
Dense_Shared_Reverse_Layer<Float>::
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
Dense_Shared_Reverse_Layer<Float>::
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
Dense_Shared_Reverse_Layer<Float>::
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
Dense_Shared_Reverse_Layer<Float>::
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
Dense_Shared_Reverse_Layer<Float>::
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
Dense_Shared_Reverse_Layer<Float>::
activation(const distribution<float> & preprocessed) const
{
    int no = outputs();
    distribution<float> output(no);
    activation(&preprocessed[0], &output[0]);
    return output;
}

template<typename Float>
distribution<double>
Dense_Shared_Reverse_Layer<Float>::
activation(const distribution<double> & preprocessed) const
{
    int no = outputs();
    distribution<double> output(no);
    activation(&preprocessed[0], &output[0]);
    return output;
}

template<typename Float>
size_t
Dense_Shared_Reverse_Layer<Float>::
fprop_temporary_space_required() const
{
    // We just need the outputs...
    return inputs() + outputs();
}

template<typename Float>
distribution<float>
Dense_Shared_Reverse_Layer<Float>::
fprop(const distribution<float> & inputs,
      float * temp_space, size_t temp_space_size) const
{
    int ni = this->inputs(), no = this->outputs();

    if (temp_space_size != ni + no)
        throw Exception("Dense_Shared_Reverse_Layer::fprop(): wrong size");
    distribution<float> result = apply(inputs);
    std::copy(inputs.begin(), inputs.end(), temp_space);
    std::copy(result.begin(), result.end(), temp_space + ni);
    return result;
}

template<typename Float>
distribution<double>
Dense_Shared_Reverse_Layer<Float>::
fprop(const distribution<double> & inputs,
      double * temp_space, size_t temp_space_size) const
{
    int ni = this->inputs(), no = this->outputs();

    if (temp_space_size != ni + no)
        throw Exception("Dense_Shared_Reverse_Layer::fprop(): wrong size");
    distribution<double> result = apply(inputs);
    std::copy(inputs.begin(), inputs.end(), temp_space);
    std::copy(result.begin(), result.end(), temp_space + ni);
    return result;
}

template<typename Float>
void
Dense_Shared_Reverse_Layer<Float>::
bprop(const distribution<float> & output_errors,
      float * temp_space, size_t temp_space_size,
      Parameters & gradient,
      distribution<float> & input_errors,
      double example_weight,
      bool calc_input_errors) const
{
    int ni = this->inputs(), no = this->outputs();

    if (temp_space_size != ni + no)
        throw Exception("Dense_Shared_Reverse_Layer::bprop(): wrong temp size");

    const float * inputs = temp_space;
    const float * outputs = temp_space + ni;

    // Differentiate the output function
    float derivs[no];
    transfer_function->derivative(outputs, derivs, no);

    // Bias updates are simply derivs in multiplied by transfer deriv
    float dbias[no];
    SIMD::vec_prod(derivs, &output_errors[0], dbias, no);

    gradient.vector(1, "bias").update(dbias, example_weight);

    if (calc_input_errors) input_errors.resize(ni);

    for (unsigned i = 0;  i < ni;  ++i) {
        bool was_missing = isnan(inputs[i]);
        if (calc_input_errors) input_errors[i] = 0;
        
        if (!was_missing) {
            gradient.matrix(0, "weights")
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
            gradient.matrix(0, "weights")
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
Dense_Shared_Reverse_Layer<Float>::
bprop(const distribution<double> & output_errors,
      double * temp_space, size_t temp_space_size,
      Parameters & gradient,
      distribution<double> & input_errors,
      double example_weight,
      bool calc_input_errors) const
{
    if (temp_space_size != outputs())
        throw Exception("Dense_Shared_Reverse_Layer::bprop(): wrong temp size");

    int ni = this->inputs(), no = this->outputs();

    const double * inputs = temp_space;
    const double * outputs = temp_space + ni;

    // Differentiate the output function
    double derivs[no];
    transfer_function->derivative(outputs, derivs, no);

    // Bias updates are simply derivs in multiplied by transfer deriv
    double dbias[no];
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
Dense_Shared_Reverse_Layer<Float>::
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
        throw Exception("Dense_Shared_Reverse_Layer::random_fill(): unknown missing values");
    }
}

template<typename Float>
void
Dense_Shared_Reverse_Layer<Float>::
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
        throw Exception("Dense_Shared_Reverse_Layer::random_fill(): unknown missing values");
    }
}

template<typename Float>
void
Dense_Shared_Reverse_Layer<Float>::
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
        throw Exception("unknown missing_values");
    }
}

template<typename Float>
size_t
Dense_Shared_Reverse_Layer<Float>::
parameter_count() const
{
    return weights.num_elements() + bias.size() + missing_replacements.size()
        + missing_activations.num_elements();
}

template<typename Float>
std::pair<float, float>
Dense_Shared_Reverse_Layer<Float>::
targets(float maximum) const
{
    return transfer_function->targets(maximum);
}

template<typename Float>
bool
Dense_Shared_Reverse_Layer<Float>::
equal_impl(const Layer & other) const
{
    const Dense_Shared_Reverse_Layer<Float> & cast
        = dynamic_cast<const Dense_Shared_Reverse_Layer<Float> &>(other);
    return operator == (cast);
}

template<typename Float>
bool
Dense_Shared_Reverse_Layer<Float>::
operator == (const Dense_Shared_Reverse_Layer & other) const
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
struct Dense_Shared_Reverse_Layer<Float>::RegisterMe {
    RegisterMe()
    {
        Register_Factory<Layer, Dense_Shared_Reverse_Layer<Float> >
            STF_REGISTER(Dense_Shared_Reverse_Layer<Float>().class_id());
    }
};

template<typename Float>
typename Dense_Shared_Reverse_Layer<Float>::RegisterMe
Dense_Shared_Reverse_Layer<Float>::register_me;

} // namespace ML

#endif /* __jml__neural__dense_shared_reverse_layer_impl_h__ */
