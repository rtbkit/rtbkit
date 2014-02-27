/* backprop_cuda.h                                     -*- C++ -*-
   Jeremy Barnes, 25 May 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

   Routines to perform forward and backwards propagation with a CUDA device.
   This is the API; implementation is in the .cu file.
*/

#ifndef __jml__backprop_cuda_h__
#define __jml__backprop_cuda_h__

#include <boost/shared_ptr.hpp>
#include "perceptron_defs.h"

namespace ML {
namespace CUDA {



/*****************************************************************************/
/* BACKPROP                                                                  */
/*****************************************************************************/

struct Backprop {

    struct Plan;     // Implementation is private
    struct Context;  // implementation is private

    /** Plan how to run it on the GPU and transfer all of the data to the
        GPU. */
    std::shared_ptr<Plan>
    plan(int num_layers,
         const int * architecture,
         const float * const * weights,
         const float * const * biases,
         const int * w_strides,
         Activation activation,
         float fire,
         float inhibit,
         float learning_rate,
         bool on_host,
         bool use_textures) const;
        
    /** Execute a planned batch of updates on the GPU.  The set of feature
        vectors to train over and their associated weights, as well as the
        arrays to accumulate the updates in, are provided. */
    std::shared_ptr<Context>
    execute(const Plan & plan,
            const float * feature_vectors,
            int num_feature_vectors,
            const float * example_weights,
            const int * labels,
            float * const * weight_updates,
            float * const * bias_updates,
            float & correct,
            float & total,
            float & rms_error) const;

    /** Wait for the given context to be finished. */
    void synchronize(Context & context) const;
};

} // namespace CUDA
} // namespace ML


#endif /* __jml__backprop_cuda_h__ */

