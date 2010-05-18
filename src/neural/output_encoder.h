/* output_encoder.h                                                -*- C++ -*-
   Jeremy Barnes, 18 May 2010
   Copyright (c) 2010 Jeremy Barnes.  All rights reserved.

   Class to take care of encoding the output of a neural network.
*/

#ifndef __jml__neural__output_encoder_h__
#define __jml__neural__output_encoder_h__

#include "layer.h"
#include "jml/boosting/label.h"
#include "jml/boosting/feature_info.h"
#include "jml/db/persistent_fwd.h"

namespace ML {


class Configuration;


/*****************************************************************************/
/* OUTPUT_ENCODER                                                            */
/*****************************************************************************/

struct Output_Encoder {

    Output_Encoder();

    Output_Encoder(const Feature_Info & label_info);

    void init(const Feature_Info & label_info);

    void configure(const Configuration & config,
                   const Layer & layer);

    void swap(Output_Encoder & other);

    enum Mode {
        REGRESSION,
        BINARY,
        MULTICLASS,
        INVALID
    };

    distribution<float> target(const Label & label) const;

    distribution<float> decode(const distribution<float> & encoded);

    double calc_auc(const std::vector<float> & outputs,
                    const std::vector<Label> & labels) const;

    void serialize(DB::Store_Writer & store) const;
    void reconstitute(DB::Store_Reader & store);
    
    Mode mode;
    float value_true, value_false;  // for binary
    int num_inputs, num_outputs;
};

} // namespace ML

#endif /* __jml__neural__output_encoder_h__ */
