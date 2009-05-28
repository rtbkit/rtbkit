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
#include "arch/cuda/atomic.h"
#include "math/xdiv.h"
#include "perceptron_defs.h"
#include <vector>
#include "backprop_cuda.h"
#include "fixed_point_accum.h"

using namespace std;


/* TODO:
   - Allow it to run with max_width > 512 (maximum thread block width)
   - tanh function that gives bit-for-bit equivalent results as on the
     host
   - Remove learning rate from the update (apply it when updating the weights)
     and use a constant that conditions the numbers to work well within the
     range of the update
   - Process 4 feature vectors together in the same loop
*/


typedef ML::FixedPointAccum32 UpdateFloat;
//typedef float UpdateFloat;

/** Given an activation function and an input, apply that activation
    function */
__device__ float transform(float input, int activation)
{
    switch (activation) {
    case ML::ACT_TANH: {
        float exp2i = __expf(input + input);
        return __fdividef(exp2i - 1.0f, exp2i + 1.0f);
    }
    case ML::ACT_IDENTITY: return input;
    default:
        return 0.0;
    }
}

/** Given an output and an error, what's the delta (derivative * error)? */
__device__ float delta(float output, float error, int activation)
{
    switch (activation) {
    case ML::ACT_TANH:
        return (1.0f - output * output) * error;
    case ML::ACT_IDENTITY: return output * error; 
    default:
        return 0.0;
    }
}

/** Train a fully-connected neural network architecture via backpropagation
    one a single training example.  The work is split over all of the cores
    within a single multiprocessor.  (So, on a Geforce 260 core 216, we have
    27 multiprocessors with 8 cores each, and so we could train on 27 different
    feature vectors in parallel.
*/
__device__ void
train_example(const float * input,
              int label,
              float example_weight,
              int num_layers,
              float * scratch,  // shared memory scratch space
              const float * const * w,  // weights for each layer
              const float * const * biases, // for each layer
              const int * architecture,
              const int * w_strides,
              UpdateFloat * const * w_updates, // wt updates for each layer
              UpdateFloat * const * b_updates, // bias upd for each layer
              int activation,            // activation function
              float fire,   // target value for firing neuron
              float inhibit, // target value for inhibited neuron)
              float learning_rate,
              int num_threads_in_block,
              int total_neurons,
              float * layer_outputs)  // global scratch space[total neurons]
{
    // access thread id
    const unsigned tid = threadIdx.x;

    const unsigned block_num  = blockIdx.x;

#if defined(__DEVICE_EMULATION__) && 0
    if (tid == 0 && block_num == 0) {
        fprintf(stderr, "starting fprop example %d wt %f; label %d\n",
                block_num, example_weight, label);

        int feature_vector_width = architecture[0];

        for (unsigned i = 0;  i < feature_vector_width;  ++i) {
            fprintf(stderr, "input %d: value %f\n",
                    i, input[i]);
        }
    }
#endif


    /*************************************************************************/
    /* FPROP                                                                 */
    /*************************************************************************/

    /* First, copy the inputs into shared memory */
    int ni = architecture[0];
    scratch[tid] = (tid < ni ? input[tid] : 0.0);

    /* Let everything catch up */
    __syncthreads();


    float * last_layer_outputs = 0;
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

#if defined(__DEVICE_EMULATION__) && 1
        if (tid == 0)
            fprintf(stderr, "fprop: tid %d layer %d ni %d no %d last_layer_outputs %p this_layer_outputs %p next_layer_outputs %p\n",
                    tid, l, ni, no, last_layer_outputs, this_layer_outputs,
                    next_layer_outputs);
#endif

        /* Add in the layer outputs.  We iterate with all threads */
        
        // Start off with the bias terms
        double accum = (tid < no ? biases[l][tid] : 0.0);

        if (__any(tid < no)) {

            for (unsigned i = 0;  i < ni;  ++i) {
                // No bank conflicts as all threads are accessing same value
                float inval = scratch[i];
                //float inval = (l == 0 ? input[i] : last_layer_outputs[i]);
                
                // Coalesced access; maybe texture would be better
                float weight
                    = (tid < no ? layer_weights[i * w_stride + tid] : 0.0);
                
                accum += weight * inval;
            }
        }         

        // Let everything catch up so that we can write to scratch
        __syncthreads();
        
        if (__any(tid < no)) {

            if (tid < no)
                this_layer_outputs[tid] = scratch[tid]
                    = transform(accum, activation);
        }

#if defined(__DEVICE_EMULATION__) && 0
        __syncthreads();
        if (tid == 0 && block_num == 0) {
            fprintf(stderr, "completed fprop layer %d example %d; label %d\n",
                    l, block_num, label);
            for (unsigned i = 0;  i < no;  ++i) {
                fprintf(stderr, "output %d: value %f\n",
                        i, this_layer_outputs[i]);
            }
        }
#endif
        
    }


    /*************************************************************************/
    /* BPROP                                                                 */
    /*************************************************************************/

    /* How many output layers? */
    int no = architecture[num_layers];

    this_layer_outputs = last_layer_outputs;
    
    /* First error calculation pass */
    bool correct = (label == tid);
    float wanted = (correct ? fire : inhibit);

    float last_output = scratch[tid];

    __syncthreads();

    scratch[tid] = (tid < no ? wanted - last_output : 0.0);
    
    /* Let everything catch up */
    __syncthreads();


#if defined(__DEVICE_EMULATION__) && 0
    if (tid == 0 && block_num == 0) {
        fprintf(stderr, "completed fprop example %d; label %d\n",
                block_num, label);
        for (unsigned i = 0;  i < no;  ++i) {
            fprintf(stderr, "output %d: value %f error %f correct %d\n",
                    i, this_layer_outputs[i], scratch[i], (label == i));
        }
    }
#endif


    /* Backpropegate. */
    for (int l = num_layers - 1;  l >= 0;
         --l,
             __syncthreads(),
             this_layer_outputs = last_layer_outputs) {
        
        // Get information about the layer:
        int ni = architecture[l];
        int no = architecture[l + 1];

        const float * layer_weights = w[l];
        int w_stride = w_strides[l];

        UpdateFloat * layer_updates = w_updates[l];
        UpdateFloat * layer_bias_updates  = b_updates[l];
        
        last_layer_outputs = this_layer_outputs - ni;

#if defined(__DEVICE_EMULATION__) && 1
        if (tid == 0)
            fprintf(stderr, "bprop: tid %d layer %d ni %d no %d last_layer_outputs %p this_layer_outputs %p layer_outputs %p end %p\n",
                    tid, l, ni, no, last_layer_outputs, this_layer_outputs,
                    layer_outputs, layer_outputs + total_neurons);
#endif

        
        float prev_output = (tid >= no ? 0.0 : this_layer_outputs[tid]);

        float error = scratch[tid];
        
        float d = (tid >= no ? 0.0 : delta(prev_output, error, activation));

        if (l > 0) {
            // Make sure all threads have caught up so that we can modify error
            // without affecting them
            __syncthreads();

            // Broadcast the d values so that we can use them to calculate the
            // errors
            scratch[tid] = d;

            // Make sure everything can get its d value
            __syncthreads();
            
            double total = 0.0;
            if (tid < ni) {
                for (unsigned o = 0;  o < no;  ++o) {
                    float d = scratch[o];  // may be the d from another thread
                    float update = d * layer_weights[tid * w_stride + o];
                    total += update;
                }
            }

            // Wait for everything to finish so that we can overwrite the d
            // values with the new errors
            __syncthreads();
            
            scratch[tid] = total;
        }


#if defined(__DEVICE_EMULATION__) && 0
        __syncthreads();

        if (tid == 0 && block_num == 0) {
            fprintf(stderr, "completed error propagation layer %d\n",
                    l);
            for (unsigned i = 0;  i < ni;  ++i) {
                fprintf(stderr, "input %d: error %f\n",
                        i, scratch[i]);
            }
        }
#endif


        // Again, threads indexed too low just leave
        if (tid >= no) continue;

        /* Update the weights. */
        float k = example_weight * learning_rate;

        /* Now for the updates.  In order to avoid trying to write the same
           memory over and over, we stagger the starting points so that
           each example will start at a different place, thus minimising
           conflicting writes when we have multiple multiprocessors working
           on the same thing. */

        int thread_stride = ni / num_threads_in_block;
        if (thread_stride == 0) thread_stride = 1;

        int start_at = (block_num * thread_stride) % ni;

        for (unsigned i_ = start_at;  i_ < ni + start_at;  ++i_) {

            // Get the real index of i
            unsigned i = i_ - (i_ >= ni) * ni;

            float prev = (l == 0 ? input[i] : last_layer_outputs[i]); 
            float update = prev * k * d;

            atomic_add(layer_updates[i * w_stride + tid], update);
        }
        
        /* Update the bias */
        float update = k * d;

        //layer_bias_updates[tid] += update;
        atomic_add(layer_bias_updates[tid], update);
    }
}

/** Train a fully-connected neural network architecture via backpropagation
    one a single training example.  The work is split over all of the cores
    within a single multiprocessor.  (So, on a Geforce 260 core 216, we have
    27 multiprocessors with 8 cores each, and so we could train on 27 different
    feature vectors in parallel.
*/
__device__ void
train_4_examples(const float * input1,
                 const float * input2,
                 const float * input3,
                 const float * input4,
                 int4 label,
                 float4 example_weight,
                 int num_layers,
                 float * scratch1,
                 float * scratch2,
                 float * scratch3,
                 float * scratch4,
                 const float * const * w,  // weights for each layer
                 const float * const * biases, // for each layer
                 const int * architecture,
                 const int * w_strides,
                 UpdateFloat * const * w_updates, // wt updates for each layer
                 UpdateFloat * const * b_updates, // bias upd for each layer
                 int activation,            // activation function
                 float fire,   // target value for firing neuron
                 float inhibit, // target value for inhibited neuron)
                 float learning_rate,
                 int num_threads_in_block,
                 int total_neurons,
                 float * layer_outputs1,
                 float * layer_outputs2,
                 float * layer_outputs3,
                 float * layer_outputs4)  // global scratch space[total neurons]
{
    // access thread id
    const unsigned tid = threadIdx.x;
    
    const unsigned block_num  = blockIdx.x;
    

    /*************************************************************************/
    /* FPROP                                                                 */
    /*************************************************************************/

    /* First, copy the inputs into shared memory */
    int ni = architecture[0];
    scratch1[tid] = (tid < ni ? input1[tid] : 0.0);
    scratch2[tid] = (tid < ni ? input2[tid] : 0.0);
    scratch3[tid] = (tid < ni ? input3[tid] : 0.0);
    scratch4[tid] = (tid < ni ? input4[tid] : 0.0);

    /* Let everything catch up */
    __syncthreads();


    float * last_layer_outputs1 = 0;
    float * this_layer_outputs1 = layer_outputs1;
    float * next_layer_outputs1;

    float * last_layer_outputs2 = 0;
    float * this_layer_outputs2 = layer_outputs2;
    float * next_layer_outputs2;

    float * last_layer_outputs3 = 0;
    float * this_layer_outputs3 = layer_outputs3;
    float * next_layer_outputs3;

    float * last_layer_outputs4 = 0;
    float * this_layer_outputs4 = layer_outputs4;
    float * next_layer_outputs4;

    for (unsigned l = 0;
         l < num_layers;
         ++l,
             __syncthreads(),
             last_layer_outputs1 = this_layer_outputs1,
             this_layer_outputs1 = next_layer_outputs1,
             last_layer_outputs2 = this_layer_outputs2,
             this_layer_outputs2 = next_layer_outputs2,
             last_layer_outputs3 = this_layer_outputs3,
             this_layer_outputs3 = next_layer_outputs3,
             last_layer_outputs4 = this_layer_outputs4,
             this_layer_outputs4 = next_layer_outputs4) {

        // Get information about the layer:
        int ni = architecture[l];
        int no = architecture[l + 1];

        const float * layer_weights = w[l];
        int w_stride = w_strides[l];

        next_layer_outputs1 = this_layer_outputs1 + no;
        next_layer_outputs2 = this_layer_outputs2 + no;
        next_layer_outputs3 = this_layer_outputs3 + no;
        next_layer_outputs4 = this_layer_outputs4 + no;

        /* Add in the layer outputs.  We iterate with all threads */
        
        // Start off with the bias terms
        double accum1 = (tid < no ? biases[l][tid] : 0.0);
        double accum2 = accum1, accum3 = accum1, accum4 = accum1;

        if (__any(tid < no)) {

            for (unsigned i = 0;  i < ni;  ++i) {
                // No bank conflicts as all threads are accessing same value
                float inval1 = scratch1[i];
                float inval2 = scratch2[i];
                float inval3 = scratch3[i];
                float inval4 = scratch4[i];
                
                // Coalesced access; maybe texture would be better
                float weight
                    = (tid < no ? layer_weights[i * w_stride + tid] : 0.0);
                
                accum1 += weight * inval1;
                accum2 += weight * inval2;
                accum3 += weight * inval3;
                accum4 += weight * inval4;
            }
        }         

        // Let everything catch up so that we can write to scratch
        __syncthreads();
        
        if (__any(tid < no)) {

            if (tid < no) {
                this_layer_outputs1[tid] = scratch1[tid]
                    = transform(accum1, activation);
                this_layer_outputs2[tid] = scratch2[tid]
                    = transform(accum1, activation);
                this_layer_outputs3[tid] = scratch3[tid]
                    = transform(accum1, activation);
                this_layer_outputs4[tid] = scratch4[tid]
                    = transform(accum1, activation);
            }
        }
    }


    /*************************************************************************/
    /* BPROP                                                                 */
    /*************************************************************************/

    /* How many output layers? */
    int no = architecture[num_layers];

    this_layer_outputs1 = last_layer_outputs1;
    this_layer_outputs2 = last_layer_outputs2;
    this_layer_outputs3 = last_layer_outputs3;
    this_layer_outputs4 = last_layer_outputs4;
    
    /* First error calculation pass */
    bool correct1 = (label.x == tid);
    bool correct2 = (label.y == tid);
    bool correct3 = (label.z == tid);
    bool correct4 = (label.w == tid);

    float wanted1 = (correct1 ? fire : inhibit);
    float wanted2 = (correct2 ? fire : inhibit);
    float wanted3 = (correct3 ? fire : inhibit);
    float wanted4 = (correct4 ? fire : inhibit);

    float last_output1 = scratch1[tid];
    float last_output2 = scratch2[tid];
    float last_output3 = scratch3[tid];
    float last_output4 = scratch4[tid];

    __syncthreads();

    scratch1[tid] = (tid < no ? wanted1 - last_output1 : 0.0);
    scratch2[tid] = (tid < no ? wanted2 - last_output2 : 0.0);
    scratch3[tid] = (tid < no ? wanted3 - last_output3 : 0.0);
    scratch4[tid] = (tid < no ? wanted4 - last_output4 : 0.0);
    
    /* Let everything catch up */
    __syncthreads();


    /* Backpropegate. */
    for (int l = num_layers - 1;  l >= 0;
         --l,
             __syncthreads(),
             this_layer_outputs1 = last_layer_outputs1,
             this_layer_outputs2 = last_layer_outputs2,
             this_layer_outputs3 = last_layer_outputs3,
             this_layer_outputs4 = last_layer_outputs4) {
        
        // Get information about the layer:
        int ni = architecture[l];
        int no = architecture[l + 1];

        const float * layer_weights = w[l];
        int w_stride = w_strides[l];

        UpdateFloat * layer_updates = w_updates[l];
        UpdateFloat * layer_bias_updates  = b_updates[l];
        
        last_layer_outputs1 = this_layer_outputs1 - ni;
        last_layer_outputs2 = this_layer_outputs2 - ni;
        last_layer_outputs3 = this_layer_outputs3 - ni;
        last_layer_outputs4 = this_layer_outputs4 - ni;
        
        float prev_output1 = (tid >= no ? 0.0 : this_layer_outputs1[tid]);
        float prev_output2 = (tid >= no ? 0.0 : this_layer_outputs2[tid]);
        float prev_output3 = (tid >= no ? 0.0 : this_layer_outputs3[tid]);
        float prev_output4 = (tid >= no ? 0.0 : this_layer_outputs4[tid]);

        float error1 = scratch1[tid];
        float error2 = scratch2[tid];
        float error3 = scratch3[tid];
        float error4 = scratch4[tid];
        
        float d1 = (tid >= no ? 0.0 : delta(prev_output1, error1, activation));
        float d2 = (tid >= no ? 0.0 : delta(prev_output2, error2, activation));
        float d3 = (tid >= no ? 0.0 : delta(prev_output3, error3, activation));
        float d4 = (tid >= no ? 0.0 : delta(prev_output4, error4, activation));

        if (l > 0) {
            // Make sure all threads have caught up so that we can modify error
            // without affecting them
            __syncthreads();

            // Broadcast the d values so that we can use them to calculate the
            // errors
            scratch1[tid] = d1;
            scratch2[tid] = d2;
            scratch3[tid] = d3;
            scratch4[tid] = d4;

            // Make sure everything can get its d value
            __syncthreads();
            
            double total1 = 0.0, total2 = 0.0, total3 = 0.0, total4 = 0.0;
            if (tid < ni) {
                for (unsigned o = 0;  o < no;  ++o) {
                    float d1 = scratch1[o];
                    float d2 = scratch2[o];
                    float d3 = scratch3[o];
                    float d4 = scratch4[o];
                    
                    float w = layer_weights[tid * w_stride + o];

                    float update1 = d1 * w;
                    float update2 = d2 * w;
                    float update3 = d3 * w;
                    float update4 = d4 * w;

                    total1 += update1;
                    total2 += update2;
                    total3 += update3;
                    total4 += update4;
                }
            }

            // Wait for everything to finish so that we can overwrite the d
            // values with the new errors
            __syncthreads();
            
            scratch1[tid] = total1;
            scratch2[tid] = total2;
            scratch3[tid] = total3;
            scratch4[tid] = total4;
        }

        // Again, threads indexed too low just leave
        if (tid >= no) continue;

        /* Update the weights. */
        float k1 = example_weight.x * learning_rate;
        float k2 = example_weight.y * learning_rate;
        float k3 = example_weight.z * learning_rate;
        float k4 = example_weight.w * learning_rate;

        /* Now for the updates.  In order to avoid trying to write the same
           memory over and over, we stagger the starting points so that
           each example will start at a different place, thus minimising
           conflicting writes when we have multiple multiprocessors working
           on the same thing. */

        int thread_stride = ni / num_threads_in_block;
        if (thread_stride == 0) thread_stride = 1;

        int start_at = (block_num * thread_stride) % ni;

        for (unsigned i_ = start_at;  i_ < ni + start_at;  ++i_) {

            // Get the real index of i
            unsigned i = i_ - (i_ >= ni) * ni;

            float prev1 = (l == 0 ? input1[i] : last_layer_outputs1[i]); 
            float prev2 = (l == 0 ? input2[i] : last_layer_outputs2[i]); 
            float prev3 = (l == 0 ? input3[i] : last_layer_outputs3[i]); 
            float prev4 = (l == 0 ? input4[i] : last_layer_outputs4[i]); 

#if 1
            float update1 = prev1 * k1 * d1;
            float update2 = prev2 * k2 * d2;
            float update3 = prev3 * k3 * d3;
            float update4 = prev4 * k4 * d4;

            atomic_add(layer_updates[i * w_stride + tid], update1);
            atomic_add(layer_updates[i * w_stride + tid], update2);
            atomic_add(layer_updates[i * w_stride + tid], update3);
            atomic_add(layer_updates[i * w_stride + tid], update4);
#else
            float update
                = (k1 * d1 * prev1)
                + (k2 * d2 * prev2)
                + (k3 * d3 * prev3)
                + (k4 * d4 * prev4);

            atomic_add(layer_updates[i * w_stride + tid], update);
#endif
        }
#if 1
        float update1 = k1 * d1;
        float update2 = k2 * d2;
        float update3 = k3 * d3;
        float update4 = k4 * d4;

        atomic_add(layer_bias_updates[tid], update1);
        atomic_add(layer_bias_updates[tid], update2);
        atomic_add(layer_bias_updates[tid], update3);
        atomic_add(layer_bias_updates[tid], update4);
#else
        /* Update the bias */
        float update = ((k1 * d1) + (k2 * d2) + (k3 * d3) + (k4 * d4));

        atomic_add(layer_bias_updates[tid], update);
#endif
    }

#if 0
        for (unsigned i_ = start_at;  i_ < ni + start_at;  ++i_) {

            // Get the real index of i
            unsigned i = i_ - (i_ >= ni) * ni;

            float prev = (l == 0 ? input[i] : last_layer_outputs[i]); 
            float update = prev * k * d;

            atomic_add(layer_updates[i * w_stride + tid], update);
        }
        
        /* Update the bias */
        float update = k * d;

        //layer_bias_updates[tid] += update;
        atomic_add(layer_bias_updates[tid], update);
#endif

}

__global__ void
train_examples_kernel(const float * feature_vectors,  // feature vector [ni]
                      int feature_vector_width,
                      const int * labels,
                      const float * example_weights,
                      int num_layers,
                      const float * const * w,  // weights for each layer
                      const float * const * biases, // for each layer
                      const int * architecture,
                      const int * w_strides,
                      UpdateFloat * const * w_updates, // wt updates for each layer
                      UpdateFloat * const * b_updates, // bias upd for each layer
                      int activation,            // activation function
                      float fire,   // target value for firing neuron
                      float inhibit, // target value for inhibited neuron)
                      float learning_rate,
                      int num_threads_in_block,
                      int total_neurons,
                      float * layer_outputs,  // scratch space[total neurons]
                      int examples_per_block,
                      int total_num_examples,
                      int max_width)
{
    const unsigned block_num  = blockIdx.x;
    
    /* Where we accumulate our errors, layer by layer.  The size is that of
       the largest dimension. */
    extern __shared__ float scratch[];
    
    /* The layer outputs (activation of the neurons).  This is where the
       shared memory goes to.  Note that we store only the activated outputs,
       not the inputs.

       blockDim.x gives us the number of threads, which is also the size of
       the errors array, so that our layer outputs have to start at this
       offset.
    */

    // Get our private scratch memory for this block
    layer_outputs += block_num * total_neurons * 4;
    
    unsigned example_num_base = block_num * examples_per_block;
    unsigned last_example = min(total_num_examples, example_num_base + examples_per_block);

    unsigned example_num = example_num_base;
#if 0
    for (;  example_num < last_example - 3;  example_num += 4) {
        const float * input1 = feature_vectors + example_num * feature_vector_width;
        const float * input2 = input1 + feature_vector_width;
        const float * input3 = input2 + feature_vector_width;
        const float * input4 = input3 + feature_vector_width;

        int4 label;
        label.x = labels[example_num];
        label.y = labels[example_num + 1];
        label.z = labels[example_num + 2];
        label.w = labels[example_num + 3];

        float4 example_weight;
        example_weight.x = example_weights[example_num];
        example_weight.y = example_weights[example_num + 1];
        example_weight.z = example_weights[example_num + 2];
        example_weight.w = example_weights[example_num + 3];

        float * scratch1 = scratch;
        float * scratch2 = scratch1 + max_width;
        float * scratch3 = scratch2 + max_width;
        float * scratch4 = scratch3 + max_width;

        float * layer_outputs1 = layer_outputs;
        float * layer_outputs2 = layer_outputs1 + total_neurons;
        float * layer_outputs3 = layer_outputs2 + total_neurons;
        float * layer_outputs4 = layer_outputs3 + total_neurons;
        
        train_4_examples(input1, input2, input3, input4,
                         label, example_weight,
                         num_layers, scratch1, scratch2, scratch3, scratch4,
                         w, biases, architecture, w_strides,
                         w_updates, b_updates,
                         activation, fire, inhibit, learning_rate,
                         num_threads_in_block, total_neurons,
                         layer_outputs1, layer_outputs2, layer_outputs3,
                         layer_outputs4);

    }
#endif

#if 1
    for (;  example_num < last_example;  example_num += 1) {
        const float * input1 = feature_vectors + example_num * feature_vector_width;
        const float * input2 = input1 + feature_vector_width;
        const float * input3 = input1;
        const float * input4 = input2;

        int4 label;
        label.x = labels[example_num];
        label.y = labels[example_num + 1];
        label.z = labels[example_num];
        label.w = labels[example_num + 1];

        float4 example_weight;
        example_weight.x = 0.0;
        example_weight.y = 0.0;
        example_weight.z = example_weights[example_num];
        example_weight.w = 0.0;//example_weights[example_num + 1];

        float * scratch1 = scratch;
        float * scratch2 = scratch1 + max_width;
        float * scratch3 = scratch2 + max_width;
        float * scratch4 = scratch3 + max_width;
        
        float * layer_outputs1 = layer_outputs;
        float * layer_outputs2 = layer_outputs1 + total_neurons;
        float * layer_outputs3 = layer_outputs2 + total_neurons;
        float * layer_outputs4 = layer_outputs3 + total_neurons;
        
        train_4_examples(input1, input2, input3, input4,
                         label, example_weight,
                         num_layers, scratch1, scratch2, scratch3, scratch4,
                         w, biases, architecture, w_strides,
                         w_updates, b_updates,
                         activation, fire, inhibit, learning_rate,
                         num_threads_in_block, total_neurons,
                         layer_outputs1, layer_outputs2, layer_outputs3,
                         layer_outputs4);

    }
#endif

    for (;  example_num < last_example;  ++example_num) {

        const float * input = feature_vectors + example_num * feature_vector_width;

        int label = labels[example_num];

        float example_weight = example_weights[example_num];

        train_example(input, label, example_weight,
                      num_layers, scratch, w, biases, architecture, w_strides,
                      w_updates, b_updates,
                      activation, fire, inhibit, learning_rate,
                      num_threads_in_block,
                      total_neurons, layer_outputs);
    }
}


namespace ML {
namespace CUDA {

struct Backprop::Plan {
    int num_layers;

    vector<int> architecture;
    DeviceData<int> d_architecture;

    vector<DeviceData<float> > d_weights_storage;
    vector<const float *> weights_vec;
    DeviceData<const float *> d_weights;

    vector<DeviceData<float> > d_biases_storage;
    vector<const float *> biases_vec;
    DeviceData<const float *> d_biases;

    vector<int> w_strides;
    DeviceData<int> d_w_strides;

    Activation activation;
    float fire;
    float inhibit;
    float learning_rate;

    int max_width;
    int total_neurons;

    // We need our grid size to be exactly the maximum width of the output
    dim3 threads;
    
    size_t shared_mem_size;

    Plan(int num_layers,
         const int * architecture,
         const float * const * weights,
         const float * const * biases,
         const int * w_strides,
         Activation activation,
         float fire,
         float inhibit,
         float learning_rate,
         bool on_host)
        : num_layers(num_layers),
          architecture(architecture, architecture + num_layers + 1),
          w_strides(w_strides, w_strides + num_layers),
          activation(activation),
          fire(fire),
          inhibit(inhibit),
          learning_rate(learning_rate)
    {
        //cerr << "plan: num_layers = " << num_layers << endl;

        d_architecture.init(architecture, num_layers + 1);

        d_weights_storage.resize(num_layers);
        weights_vec.resize(num_layers);

        for (unsigned l = 0;  l < num_layers;  ++l) {
            int ni = architecture[l];
            int w_stride = w_strides[l];
            d_weights_storage[l].init(weights[l], ni * w_stride);
            weights_vec[l] = d_weights_storage[l];

            //cerr << "layer " << l << ": no = " << no << " w_stride = "
            //     << w_stride << endl;
        }
    
        d_weights.init(&weights_vec[0], num_layers);

        d_biases_storage.resize(num_layers);
        biases_vec.resize(num_layers);

        for (unsigned l = 0;  l < num_layers;  ++l) {
            int no = architecture[l + 1];
            d_biases_storage[l].init(biases[l], no);
            biases_vec[l] = d_biases_storage[l];
        }
    
        d_biases.init(&biases_vec[0], num_layers);

        d_w_strides.init(w_strides, num_layers);
        
        max_width = 0;
        total_neurons = 0;

        for (unsigned l = 0;  l <= num_layers;  ++l) {
            max_width = max(max_width, architecture[l]);
            total_neurons += architecture[l];
        }

        // We need our grid size to be exactly the maximum width of the output
        threads = dim3(max_width);

        // Since we do 4 examples per loop, we need enough memory for all of
        // the four outputs for a single layer
        shared_mem_size = max_width * sizeof(float)* 4;
    }
};

struct Backprop::Context {

    const Plan & plan;
    
    DeviceData<float> d_feature_vectors;
    DeviceData<float> d_example_weights;
    DeviceData<int> d_labels;
        
    float * const * weight_updates;
    float * const * bias_updates;

    vector<DeviceData<UpdateFloat> > d_weight_updates_storage;
    vector<UpdateFloat *> weight_updates_vec;
    DeviceData<UpdateFloat *> d_weight_updates;
    
    vector<DeviceData<UpdateFloat> > d_bias_updates_storage;
    vector<UpdateFloat *> bias_updates_vec;
    DeviceData<UpdateFloat *> d_bias_updates;

    DeviceData<float> d_layer_outputs;

    dim3 grid;

    int num_feature_vectors;
    int feature_vector_width;
    int num_examples_per_invocation;

    Context(const Plan & plan,
            const float * feature_vectors,
            int num_feature_vectors,
            const float * example_weights,
            const int * labels,
            float * const * weight_updates,
            float * const * bias_updates,
            float & correct,
            float & total,
            float & rms_error)
        : plan(plan), weight_updates(weight_updates),
          bias_updates(bias_updates), num_feature_vectors(num_feature_vectors),
          feature_vector_width(feature_vector_width)
    {
        feature_vector_width = plan.architecture[0];
        
        //cerr << "num_feature_vectors = " << num_feature_vectors << endl;
        //cerr << "feature_vector_width = " << feature_vector_width
        //     << endl;

        d_feature_vectors.init(feature_vectors,
                               num_feature_vectors * feature_vector_width);
        
        d_example_weights.init(example_weights, num_feature_vectors);
        
        d_labels.init(labels, num_feature_vectors);
        
        d_weight_updates_storage.resize(plan.num_layers);
        weight_updates_vec.resize(plan.num_layers);
        
        for (unsigned l = 0;  l < plan.num_layers;  ++l) {
            int ni = plan.architecture[l];
            int w_stride = plan.w_strides[l];
            d_weight_updates_storage[l].init_zeroed(ni * w_stride);
            weight_updates_vec[l] = d_weight_updates_storage[l];
        }

        d_weight_updates.init(&weight_updates_vec[0], plan.num_layers);

        d_bias_updates_storage.resize(plan.num_layers);
        bias_updates_vec.resize(plan.num_layers);

        for (unsigned l = 0;  l < plan.num_layers;  ++l) {
            int no = plan.architecture[l + 1];
            d_bias_updates_storage[l].init_zeroed(no);
            bias_updates_vec[l] = d_bias_updates_storage[l];
        }

        d_bias_updates.init(&bias_updates_vec[0], plan.num_layers);

        num_examples_per_invocation = 4;//16;

        int grid_size = rudiv(num_feature_vectors, num_examples_per_invocation);

        // Get the scratch space
        d_layer_outputs.init(plan.total_neurons * grid_size * 4);
        
        // Our grid size is one per example
        grid = dim3(grid_size);
    }

    void execute()
    {
        train_examples_kernel<<<grid, plan.threads, plan.shared_mem_size>>>
            (d_feature_vectors,
             feature_vector_width,
             d_labels,
             d_example_weights,
             plan.num_layers,
             plan.d_weights,
             plan.d_biases,
             plan.d_architecture,
             plan.d_w_strides,
             d_weight_updates,
             d_bias_updates,
             plan.activation,
             plan.fire,
             plan.inhibit,
             plan.learning_rate,
             grid.x,
             plan.total_neurons,
             d_layer_outputs,
             num_examples_per_invocation,
             num_feature_vectors /* total num examples */,
             plan.max_width);

        //cerr << "launched" << endl;
    }
    
    void synchronize()
    {
        //cerr << "waiting for execution" << endl;
        cudaError_t err = cudaThreadSynchronize();
        
        if (err != cudaSuccess)
            throw Exception(cudaGetErrorString(err));

        //cerr << "copying memory back" << endl;

        


        for (unsigned l = 0;  l < plan.num_layers;  ++l) {
            int ni = plan.architecture[l];
            int w_stride = plan.w_strides[l];
            
            UpdateFloat sync_to[ni * w_stride];

            d_weight_updates_storage[l].sync(sync_to);
            std::copy(sync_to, sync_to + ni * w_stride, weight_updates[l]);
        }

        for (unsigned l = 0;  l < plan.num_layers;  ++l) {
            int no = plan.architecture[l + 1];

            UpdateFloat sync_to[no];
            
            d_bias_updates_storage[l].sync(sync_to);
            std::copy(sync_to, sync_to + no, bias_updates[l]);
        }
    }
};

boost::shared_ptr<Backprop::Plan>
Backprop::
plan(int num_layers,
     const int * architecture,
     const float * const * weights,
     const float * const * biases,
     const int * w_strides,
     Activation activation,
     float fire,
     float inhibit,
     float learning_rate,
     bool on_host) const
{
    boost::shared_ptr<Plan> result
        (new Plan(num_layers, architecture, weights, biases, w_strides,
                  activation, fire, inhibit, learning_rate, on_host));

    return result;
}

boost::shared_ptr<Backprop::Context>
Backprop::
execute(const Plan & plan,
        const float * feature_vectors,
        int num_feature_vectors,
        const float * example_weights,
        const int * labels,
        float * const * weight_updates,
        float * const * bias_updates,
        float & correct,
        float & total,
        float & rms_error) const
{
    boost::shared_ptr<Context> result
        (new Context(plan, feature_vectors, num_feature_vectors,
                     example_weights, labels,
                     weight_updates, bias_updates,
                     correct, total, rms_error));

    result->execute();

    return result;
}

/** Wait for the given context to be finished. */
void
Backprop::
synchronize(Context & context) const
{
    context.synchronize();
}


} // namespace CUDA
} // namespace ML
