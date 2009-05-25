/* backprop_cuda.cc                                                -*- C++ -*-
   Jeremy Barnes, 25 May 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

   CUDA-based backprop implementation.
*/

#include "arch/exception.h"
#include "compiler/compiler.h"
#include <cstdio>
#include <iostream>
#include <boost/timer.hpp>
#include <boost/utility.hpp>
#include "arch/cuda/device_data.h"
#include "math/xdiv.h"
#include "perceptron_defs.h"


using namespace std;


__device__ float transform(float input, int activation)
{
    switch (activation) {
    case ML::ACT_TANH: {
        float pos = __expf(input);
        float neg = __expf(-input);
        return __fdividef(pos + neg, pos - neg);
    }
    case ML::ACT_IDENTITY: return input;
    default:
        return 0.0;
    }
}

/* Given an output and an error, what's the delta? */
__device__ delta(float output, float error, int activation)
{
    switch (activation) {
    case ML::ACT_TANH:
        return (1.0f - output * output) * error;
    case ML::ACT_IDENTITY: return output * error; 
    default:
        return 0.0;
    }
}

/** This function will be called with ONE block of threads, with the number
    of threads equal to the widest layer that there is. */
__global__ void
train_example_kernel(const float * input,  // feature vector [ni]
                     int label,            // correct label
                     int num_layers,
                     const float * const * w,  // weights for each layer
                     const float * const * biases, // for each layer
                     const int * architecture,
                     const int * w_strides,
                     float * const * w_updates, // updates for each layer
                     int activation,
                     const int * topology,
                     float fire,   // target value for firing neuron
                     float inhibit // target value for inhibited neuron)
{
    // access thread id
    const unsigned tid = threadIdx.x;

    // access number of threads in this block
    const unsigned num_threads = blockDim.x;

    /* The layer outputs (activation of the neurons).  This is where the
       shared memory goes to.  Note that we store only the activated outputs,
       not the inputs. */
    extern __shared__ float layer_outputs[];

    /* Where we accumulate our errors, layer by layer.  The size is that of
       the largest dimension. */
    extern __shared__ float errors[];

    
    /*************************************************************************/
    /* FPROP                                                                 */
    /*************************************************************************/

    const float * last_layer_outputs = 0;
    float * this_layer_outputs = layer_outputs;
    float * next_layer_outputs;

    for (unsigned l = 0;
         l < num_layers;
         ++l,
             __syncthreads(),
             last_layer_outputs = this_layer_outputs,
             this_layer_outputs = next_layer_outputs) {
        
        // Get information about the layer:
        int ni = architecture[l];
        int no = architecture[l + 1];

        const float * layer_weights = w[l];
        int w_stride = w_strides[l];

        next_layer_outputs = this_layer_outputs + no;

        // Start off with the bias terms
        if (tid < no) this_layer_outputs[tid] = biases[l][tid];

        /* Add in the layer outputs.  We iterate with all threads */
        if (tid < no) {
            float accum = 0;
            for (unsigned i = 0;  i < ni;  ++i) {
                float inval = (l == 0 ? input[i] : last_layer_outputs[i]);

                // Coalesced access; maybe texture would be better
                float weight = layer_weights[i * w_stride + tid];

                accum += weight * inval;
            }

            this_layer_outputs[tid] = transform(accum, activation);
        }
    }

    /*************************************************************************/
    /* BPROP                                                                 */
    /*************************************************************************/

    /* How many output layers? */
    int no = architecture[num_layers];
    
    /* First error calculation pass */
    if (tid < no) {
        bool correct = (label == tid);
        float wanted = (correct ? fire : inhibit);
        errors[tid] = wanted - last_layer_outputs[tid];
    }

    /* Let everything catch up */
    __syncthreads();

    /* Backpropegate. */
    for (int l = nl - 1;  l >= 1;
         --l,
             __syncthreads(),
             last_layer_outputs = this_layer_outputs,
             this_layer_outputs = next_layer_outputs) {
        
        // Get information about the layer:
        int ni = architecture[l];
        int no = architecture[l + 1];

        const float * layer_weights = w[l];
        int w_stride = w_strides[l];

        next_layer_outputs = this_layer_outputs - no;
        
        if (tid >= no) continue;

        float d = delta(last_layer_outputs[tid], errors[tid], activation);

        if (l > 1) {
            /* Calculate new errors (for the next layer). */
            for (unsigned i = 0;  i < ni;  ++i)
                errors[i] = SIMD::vec_dotprod_dp(&delta[0],
                                                 &layer.weights[i][0],
                                                 no);
        }
        

        for (unsigned i = 0;  i < ni;  ++i) {
            float inval = (l == 0 ? input[i] : last_layer_outputs[i]);
            
            // Coalesced access; maybe texture would be better
            float weight = layer_weights[i * w_stride + tid];
            
            accum += weight * inval;
        }

        /* Update the weights. */
        float k = w * learning_rate;
        for (unsigned i = 0;  i < ni;  ++i) {
            float k2 = layer_outputs[l - 1][i] * k;
            SIMD::vec_add(&sub_weight_updates[l][i][0], k2, &delta[0],
                          &sub_weight_updates[l][i][0], no);
        }
        
        /* Update the bias terms.  The previous layer output (input) is
           always 1. */
        SIMD::vec_add(&sub_bias_updates[l][0], k, &delta[0],
                      &sub_bias_updates[l][0], no);
    }
    
#endif

}
