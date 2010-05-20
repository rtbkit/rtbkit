/* output_encoder.cc
   Jeremy Barnes, 18 May 2010
   Copyright (c) 2010 Jeremy Barnes.  All rights reserved.

*/

#include "output_encoder.h"
#include "jml/db/persistent.h"


using namespace std;


namespace ML {

BYTE_PERSISTENT_ENUM_IMPL(Output_Encoder::Mode);

/*****************************************************************************/
/* OUTPUT_ENCODER                                                            */
/*****************************************************************************/

Output_Encoder::
Output_Encoder()
    : mode(INVALID)
{
}

void
Output_Encoder::
configure(const Feature_Info & label_info,
          const Layer & layer,
          float target_value)
{
    Range range = layer.transfer().range();

    cerr << "range: min = " << range.min
         << " max = " << range.max
         << " neutral = " << range.neutral
         << " min_asymptotic = " << range.min_asymptotic
         << " max_asymptotic = " << range.max_asymptotic
         << " type = " << range.type
         << endl;

    if (target_value == -1.0) {
        if (range.min_asymptotic || range.max_asymptotic)
            target_value = 0.9;
        else target_value = 1.0;
    }


    switch (label_info.type()) {

    case BOOLEAN:
        value_true = target_value * range.max;
        value_false = target_value * range.min;
        num_inputs = layer.outputs();
        num_outputs = layer.outputs();

        if (layer.outputs() == 1)
            mode = BINARY;
        else if (layer.outputs() == 2)
            mode = MULTICLASS;
        else throw Exception("invalid number of outputs");

        break;

    case REAL:
        if (layer.outputs() != 1)
            throw Exception("regression with more than one output");
        value_true = value_false = std::numeric_limits<float>::quiet_NaN();
        mode = REGRESSION;
        num_inputs = num_outputs = 1;
        break;

    case CATEGORICAL:


    default:
        throw Exception("unusable output encoding");
    }

    cerr << "mode = " << mode << " value_true = " << value_true
         << " value_false = " << value_false << " num_inputs = "
         << num_inputs << " num_outputs = " << num_outputs
         << endl;
}

void
Output_Encoder::
configure(const Configuration & config, const Layer & layer)
{
    throw Exception("Output_Encoder::configure(): not done yet");
}

void
Output_Encoder::
swap(Output_Encoder & other)
{
    std::swap(mode, other.mode);
    std::swap(value_true, other.value_true);
    std::swap(value_false, other.value_false);
    std::swap(num_inputs, other.num_inputs);
    std::swap(num_outputs, other.num_outputs);
}

distribution<float>
Output_Encoder::
target(const Label & label) const
{
    distribution<float> result(num_inputs);
    
    switch (mode) {
    case REGRESSION:
        result[0] = label;
        break;
            
    case BINARY:
        if (label)
            result[0] = value_true;
        else result[0] = value_false;
        break;

    case MULTICLASS:
        for (unsigned i = 0;  i < num_inputs;  ++i)
            result[i] = (i == label ? value_true : value_false);
        break;
            
    default:
        throw Exception("invalid output encoder class");
    }

    return result;
}

float
Output_Encoder::
decode_value(float encoded) const
{
    float result
        = std::min(1.0f,
                   std::max(0.0f,
                            (encoded - value_false)
                            / (value_true - value_false)));
    return result;
}

distribution<float>
Output_Encoder::
decode(const distribution<float> & encoded) const
{
    distribution<float> result(num_outputs);

    switch (mode) {
    case REGRESSION:
        result = encoded;
        break;
            
    case BINARY:
        result[0] = decode_value(encoded[0]);
        result[1] = 1.0 - result[0];
        break;

    case MULTICLASS:
        for (unsigned i = 0;  i < num_outputs;  ++i)
            result[i] = decode_value(encoded[i]);
        //cerr << "input " << encoded << " output " << result << endl;
        break;
            
    default:
        throw Exception("invalid output encoder class");
    }

    return result;
}

double
Output_Encoder::
calc_auc(const std::vector<float> & outputs,
                const std::vector<Label> & labels) const
{
    return 0.0;
}

void
Output_Encoder::
serialize(DB::Store_Writer & store) const
{
    store << 1 << mode << value_true << value_false << num_inputs
          << num_outputs;
}

void
Output_Encoder::
reconstitute(DB::Store_Reader & store)
{
    int version;
    store >> version;
    if (version != 1)
        throw Exception("Output_Encoder::reconstitute(): invalid version");

    store >> mode >> value_true >> value_false >> num_inputs >> num_outputs;
}

} // namespace ML

