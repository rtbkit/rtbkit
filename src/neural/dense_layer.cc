/* dense_layer.cc
   Jeremy Barnes, 2 November 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

   Dense neural network layer.
*/

#include "dense_layer.h"

#include "layer.h"
#include "db/persistent.h"
#include "arch/demangle.h"
#include "algebra/matrix_ops.h"
#include "arch/simd_vector.h"


using namespace std;
using namespace ML::DB;

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
    this->transfer_function = transfer_function;
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
    this->transfer_function = transfer_function;
    if (limit == -1.0)
        limit = 1.0 / sqrt(inputs);
    random_fill(limit, thread_context);
}

template<typename Float>
boost::shared_ptr<Parameters>
Dense_Layer<Float>::
parameters()
{
    boost::shared_ptr<Parameter_Ref> pparams(new Parameter_Ref());
    Parameter_Ref & params = *pparams;

    params
        .add("weights", weights)
        .add("bias", bias);

    switch (missing_values) {
    case MV_NONE:
    case MV_ZERO:
        break;
    case MV_INPUT:
        params.add("missing_replacements", missing_replacements);
        break;
    case MV_DENSE:
        params.add("missing_activations", missing_activations);
        break;
    default:
        throw Exception("Dense_Layer::parameters(): none there");
    }

    return pparams;
}

template<typename Float>
std::string
Dense_Layer<Float>::
print() const
{
    size_t ni = inputs(), no = outputs();
    string result = format("{ layer: %zd inputs, %zd neurons, function %d\n",
                           inputs(), outputs(), transfer_function);
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
    
    result += "}\n";
    
    return result;
}

template<typename Float>
std::string
Dense_Layer<Float>::
type() const
{
    return "Dense_Layer<" + demangle(typeid(Float).name()) + ">";
}

template<typename Float>
void
Dense_Layer<Float>::
serialize(DB::Store_Writer & store) const
{
    store << compact_size_t(1);
    store << string("PERCEPTRON LAYER");
    store << compact_size_t(inputs());
    store << compact_size_t(outputs());
    store << weights;
    store << bias;
    store << missing_replacements;
    store << transfer_function;
}

template<typename Float>
void
Dense_Layer<Float>::
reconstitute(DB::Store_Reader & store)
{
    compact_size_t version(store);
    //cerr << "version = " << version << endl;
    if (version != 1)
        throw Exception("invalid layer version");

    string s;
    store >> s;
    if (s != "PERCEPTRON LAYER")
        throw Exception("invalid layer start " + s);

    compact_size_t inputs_read(store), outputs_read(store);

    store >> weights;

#if 0
    cerr << "getting bias: offset = " << store.offset() << endl;
    size_t bytes_left = store.try_to_have(256);

    for (unsigned i = 0;  i < bytes_left && i < 256;  i += 16) {
        cerr << format("%04x | ", i + store.offset());
        for (unsigned j = i;  j < i + 16;  ++j) {
            if (j < bytes_left)
                cerr << format("%02x ", (uint8_t)(store[j]));
            else cerr << "   ";
        }
        
        cerr << "| ";
        
        for (unsigned j = i;  j < i + 16;  ++j) {
            if (j < bytes_left) {
                if (store[j] >= ' ' && store[j] < 127)
                    cerr << store[j];
                else cerr << '.';
            }
            else cerr << " ";
        }
        cerr << endl;
    }
#endif

    store >> bias;


    //cerr << "after bias: offset = " << store.offset() << endl;
    store >> missing_replacements;
    store >> transfer_function;

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

template<typename Float>
void
Dense_Layer<Float>::
apply(const float * input,
      float * output) const
{
    int ni = inputs(), no = outputs();
    float act[no];
    activation(input, act);
    transfer(act, output);
}

template<typename Float>
void
Dense_Layer<Float>::
apply(const double * input,
      double * output) const
{
    int ni = inputs(), no = outputs();
    double act[no];
    activation(input, act);
    transfer(act, output);
}


void
Dense_Layer<Float>::
activation(const float * preprocessed,
           float * activation) const
{
    for (unsigned i = 0;  i < ni;  ++i)
        activation[i] = isnan(input[i]) ? missing_replacements[i] : input[i];

    int ni = inputs(), no = outputs();
    double accum[no];  // Accumulate in double precision to improve rounding
    std::copy(bias.begin(), bias.end(), accum);

    for (unsigned i = 0;  i < ni;  ++i) {
        const LFloat * w;
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
    if (!use_dense_missing) {
        Base::activation(preprocessed, activation);
        return;
    }

    int ni = inputs(), no = outputs();
    double accum[no];  // Accumulate in double precision to improve rounding
    std::copy(bias.begin(), bias.end(), accum);

    for (unsigned i = 0;  i < ni;  ++i) {
        const LFloat * w;
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
void
Dense_Layer<Float>::
random_fill(float limit, Thread_Context & context)
{
    int ni = weights.shape()[0], no = weights.shape()[1];
    
    for (unsigned j = 0;  j < no;  ++j)
        for (unsigned i = 0;  i < ni;  ++i)
            weights[i][j] = limit * (context.random01() * 2.0f - 1.0f);
    
    if (no != bias.size())
        throw Exception("bias sized wrong");

    for (unsigned o = 0;  o < bias.size();  ++o)
        bias[o] = limit * (context.random01() * 2.0f - 1.0f);

    for (unsigned i = 0;  i < ni;  ++i)
        missing_replacements[i] = limit * (context.random01() * 2.0f - 1.0f);
}

template<typename Float>
void
Dense_Layer<Float>::
zero_fill()
{
    int ni = weights.shape()[0], no = weights.shape()[1];
    
    for (unsigned j = 0;  j < no;  ++j)
        for (unsigned i = 0;  i < ni;  ++i)
            weights[i][j] = 0.0;
    
    if (no != bias.size())
        throw Exception("bias sized wrong");

    bias.fill(0.0);
    missing_replacements.fill(0.0);
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
    return (inputs() * 1) * (outputs() + 1);
}

template<typename Float>
bool
Dense_Layer<Float>::
operator == (const Dense_Layer & other) const
{
    return (transfer_function == other.transfer_function
            && inputs() == other.inputs()
            && outputs() == other.outputs()
            && weights.shape()[0] == other.weights.shape()[0]
            && weights.shape()[1] == other.weights.shape()[1]
            && missing_replacements.size() == other.missing_replacements.size()
            && bias.size() == other.bias.size()
            && weights == other.weights
            && (bias == other.bias).all()
            && (missing_replacements == other.missing_replacements).all());
}


template<typename Float>
std::pair<float, float>
Dense_Layer<Float>::
targets(float maximum, Transfer_Function_Type transfer_function)
{
    switch (transfer_function) {
    case TF_TANH:
    case TF_IDENTITY: return std::make_pair(-maximum, maximum);
    case TF_LOGSOFTMAX:
    case TF_LOGSIG: return std::make_pair(0.0f, maximum);
    default:
        throw Exception("Layer::targets(): invalid transfer_function");
    }
}



template class Dense_Layer<float>;
template class Dense_Layer<double>;




/*****************************************************************************/
/* DENSE_MISSING_LAYER                                                       */
/*****************************************************************************/


void
Dense_Missing_Layer::
random_fill(float limit, Thread_Context & context)
{
    Base::random_fill(limit, context);

    for (unsigned i = 0;  i < inputs();  ++i) {
        for (unsigned o = 0;  o < outputs();  ++o)
            missing_activations[i][o]
                = limit * (context.random01() * 2.0f - 1.0f);
    }
}

void
Dense_Missing_Layer::
zero_fill()
{
    Base::zero_fill();
    for (unsigned i = 0;  i < inputs();  ++i)
        missing_activations[i].fill(0.0);
}

size_t
Dense_Missing_Layer::
parameter_count() const
{
    size_t result = Base::parameter_count();

    if (use_dense_missing)
        result += inputs() * (outputs() - 1);

    return result;
}

void
Dense_Missing_Layer::
serialize(DB::Store_Writer & store) const
{
    Base::serialize(store);
    store << use_dense_missing;
    if (use_dense_missing)
        store << missing_activations;
}

void
Dense_Missing_Layer::
reconstitute(DB::Store_Reader & store)
{
    Base::reconstitute(store);
    store >> use_dense_missing;
    if (use_dense_missing)
        store >> missing_activations;
}

std::string
Dense_Missing_Layer::
print() const
{
    string result = Base::print();

    size_t ni = inputs(), no = outputs();

    result += "  missing activations: \n";
    for (unsigned i = 0;  i < ni;  ++i) {
        result += "    [ ";
        for (unsigned j = 0;  j < no;  ++j)
            result += format("%8.4f", missing_activations[i][j]);
        result += " ]\n";
    }
    return result;
}

template<typename X>
bool equivalent(const X & x1, const X & x2)
{
    return x1 == x2;
}

template<typename X, class U>
bool equivalent(const distribution<X, U> & x1,
                const distribution<X, U> & x2)
{
    if (x1.size() != x2.size()) return false;
    for (unsigned i = 0;  i < x1.size();  ++i)
        if (!equivalent(x1[i], x2[i])) return false;
    return true;
}

template<typename X>
bool equivalent(const std::vector<X> & x1, const std::vector<X> & x2)
{
    if (x1.size() != x2.size()) return false;
    for (unsigned i = 0;  i < x1.size();  ++i)
        if (!equivalent(x1[i], x2[i])) return false;
    return true;
}

bool
Dense_Missing_Layer::
operator == (const Dense_Missing_Layer & other) const
{
    return (Base::operator == (other)
            && use_dense_missing == other.use_dense_missing
            && missing_activations.size() == other.missing_activations.size()
            && equivalent(missing_activations, other.missing_activations));
}


} // namespace ML

