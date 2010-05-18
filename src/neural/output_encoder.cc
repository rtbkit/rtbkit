/* output_encoder.cc
   Jeremy Barnes, 18 May 2010
   Copyright (c) 2010 Jeremy Barnes.  All rights reserved.

*/

#include "output_encoder.h"
#include "jml/db/persistent.h"

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

Output_Encoder::
Output_Encoder(const Feature_Info & label_info)
{
    init(label_info);
}

void
Output_Encoder::
init(const Feature_Info & label_info)
{
}

void
Output_Encoder::
configure(const Configuration & config, const Layer & layer)
{
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

distribution<float>
Output_Encoder::
decode(const distribution<float> & encoded)
{
    distribution<float> result;
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

