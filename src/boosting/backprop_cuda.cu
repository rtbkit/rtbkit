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
   - Try using textures for the W arrays (caching could make a big difference)
*/


typedef ML::FixedPointAccum32 UpdateFloat;
//typedef float UpdateFloat;

/** Given an activation function and an input, apply that activation
    function */
__device__ float transform(float input, int activation)
{
    switch (activation) {
    case ML::ACT_TANH: {
        return tanh(input);
        //float exp2i = __expf(input + input);
        //return __fdividef(exp2i - 1.0f, exp2i + 1.0f);
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

texture<float, 1, cudaReadModeElementType> weights_tex;
texture<float, 1, cudaReadModeElementType> biases_tex;;


#if !defined(__DEVICE_EMULATION__)

template<const texture<float, 1, cudaReadModeElementType> & Tex>
struct WeightsAccess {
    const float * base;  // if zero, then texture access
    int offset;

    __device__ WeightsAccess(const float * base = 0)
        : base(base), offset(0)
    {
    }

    __device__ void init(const float * base)
    {
        this->base = base;
        offset = 0;
    }

    __device__ void operator += (int val)
    {
        offset += val;
    }

    __device__ void operator -= (int val)
    {
        offset -= val;
    }

    __device__ float operator [] (int ofs)
    {
        if (base) return base[offset + ofs];
        else return tex1Dfetch(Tex, offset + ofs);
    }
};

#endif

/** Train a fully-connected neural network architecture via backpropagation
    one a single training example.  The work is split over all of the cores
    within a single multiprocessor.  (So, on a Geforce 260 core 216, we have
    27 multiprocessors with 8 cores each, and so we could train on 27 different
    feature vectors in parallel.
*/

#define N 4
#define train_N_examples train_4_examples
#include "backprop_cuda_train_N_examples.cu"

#define N 1
#define train_N_examples train_1_example
#include "backprop_cuda_train_N_examples.cu"
#undef N
#undef train_N_examples

#if 0
__device__ void
train_example(const float * input,
              int label,
              float example_weight,
              int num_layers,
              float * scratch,  // shared memory scratch space
              const WeightsAccess<weights_tex> & w,
              const WeightsAccess<biases_tex> & biases,
              const int * architecture,
              const int * w_strides,
              UpdateFloat * const * w_updates, // wt updates for each layer
              UpdateFloat * const * b_updates, // bias upd for each layer
              int activation,            // activation function
              float inhibit, // target value for inhibited neuron)
              float fire,   // target value for firing neuron
              float learning_rate,
              int num_threads_in_block,
              int num_threads_on_multiprocessor,
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
    int ni = architecture[0], no, w_stride;
    scratch[tid] = (tid < ni ? input[tid] : 0.0);

    /* Let everything catch up */
    __syncthreads();

    float * this_layer_outputs = layer_outputs;
    WeightsAccess<weights_tex> layer_weights = w;
    WeightsAccess<biases_tex> layer_biases  = biases;

    for (unsigned l = 0;
         l < num_layers;
         ++l,
             __syncthreads(),
             layer_weights += ni * w_stride,
             layer_biases += no,
             this_layer_outputs += no) {

        // Get architecture information about the layer:
        ni = architecture[l];
        no = architecture[l + 1];
        w_stride = w_strides[l];

#if defined(__DEVICE_EMULATION__) && 0
        if (tid == 0)
            fprintf(stderr, "fprop: layer %d ni %d no %d this_layer_outputs %p layer_weights %p layer_biases %p\n",
                    l, ni, no, this_layer_outputs,
                    layer_weights, layer_biases);
#endif

        /* Add in the layer outputs.  We iterate with all threads */
        
#if 0
        // Start off with the bias terms
        double accum = (tid < no ? layer_biases[tid] : 0.0);

        if (__any(tid < no)) {

            for (unsigned i = 0;  i < ni;  ++i) {
                // No bank conflicts as all threads are accessing same value
                float inval = scratch[i];
                
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
                this_layer_outputs[tid]
                    = scratch[tid]
                    = transform(accum, activation);
        }

#else
        /* We want to have each thread working here, even if no is much less
           than the number of threads.  To do so, we assign each thread to
           a certain o value and a certain subset of the i values, and then
           accumulate the updates, broadcasting them at the end.

           For example:
           32 threads
           2 outputs

           So we have 16 threads working on each example

           100 threads
           16 outputs

           So we have 7 threads on the first 4 examples, and 6 threads on
           the rest.
        */

        int nt = num_threads_on_multiprocessor;

        int min_threads = nt / no;
        int left_over   = nt % no;
        int max_threads = min_threads + (left_over > 0);

        int o = tid % no;    // which o value are we working on?
        int idx = tid / no;  // which thread in that block?
        int o_threads = min_threads + (o < left_over);

#if defined(__DEVICE_EMULATION__) && 0
        if (tid == 0)
            fprintf(stderr, "fprop: layer %d ni %d no %d min_threads %d left_over %d max_threads %d nib %d nt %d\n",
                    l, ni, no, min_threads, left_over, max_threads,
                    num_threads_in_block);
        fprintf(stderr, "      tid %d o %d idx %d o_threads %d\n",
                tid, o, idx, o_threads);
#endif

        double accum = 0.0;

        for (unsigned i = idx;  i < ni;  i += o_threads) {
            // warning: bank conflicts...
            float inval = scratch[i];
            float weight = layer_weights[i * w_stride + o];
            
            accum += weight * inval;
        }

        if (max_threads > 1) {

            __syncthreads();

            if (tid < no) scratch[tid] = layer_biases[tid];

            __syncthreads();
            
            /* Now we accumulate them, allowing each thread to increment in its
               turn. */
            for (unsigned i = 0;  i < max_threads;  ++i, __syncthreads())
                if (i == idx) scratch[o] += accum;
            
            if (__any(tid < no)) {
                
                if (tid < no)
                    this_layer_outputs[tid]
                        = scratch[tid]
                        = transform(scratch[tid], activation);
            }
        }
        else {
            accum += layer_biases[o];
            this_layer_outputs[o]
                = scratch[o]
                = transform(accum, activation);
            
        }
#endif


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

    // layer_biases is no longer used

    /*************************************************************************/
    /* BPROP                                                                 */
    /*************************************************************************/

    /* How many output layers? */
    this_layer_outputs -= no;

#if defined(__DEVICE_EMULATION__) && 0
        if (tid == 0)
            fprintf(stderr, "before fixing up layer_weights: %p no: %d w_stride: %d\n",
                    layer_weights, no, w_stride);
#endif

    layer_weights -= ni * w_stride;

#if defined(__DEVICE_EMULATION__) && 0
        if (tid == 0)
            fprintf(stderr, "after fixing up layer_weights: %p no: %d w_stride: %d\n",
                    layer_weights, no, w_stride);
#endif

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
             layer_weights -= (l == -1 ? 0 : architecture[l] * w_strides[l]),
             this_layer_outputs -= architecture[l + 1]) {
        
        // Get information about the layer:
        ni = architecture[l];
        no = architecture[l + 1];
        w_stride = w_strides[l];

        UpdateFloat * layer_updates = w_updates[l];
        UpdateFloat * layer_bias_updates  = b_updates[l];
        
        const float * last_layer_outputs = this_layer_outputs - ni;

#if defined(__DEVICE_EMULATION__) && 0
        if (tid == 0)
            fprintf(stderr, "bprop: tid %d layer %d ni %d no %d last_layer_outputs %p this_layer_outputs %p layer_outputs %p end %p layer_weights %p w_stride %d new layer_weights %p\n",
                    tid, l, ni, no, last_layer_outputs, this_layer_outputs,
                    layer_outputs, layer_outputs + total_neurons,
                    layer_weights, w_stride, layer_weights - (ni * w_stride));
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

#if defined(__DEVICE_EMULATION__) && 0
            //__syncthreads();

            if (tid < 10 && block_num == 0 && l == 2 && i == 0) {
                fprintf(stderr, "update for layer 2 i=0 o=%d = %.15g * %.15g * %.15g = %.15g before update %.15g\n", tid, prev, k, d, update, (float)layer_updates[i * w_stride + tid]);
            }
#endif

            atomic_add(layer_updates[i * w_stride + tid], update);

#if defined(__DEVICE_EMULATION__) && 0
            //__syncthreads();

            if (tid < 10 && block_num == 0 && l == 2 && i == 0) {
                fprintf(stderr, "                          after %.15g\n", (float)layer_updates[i * w_stride + tid]);
            }
#endif
        }
        
        /* Update the bias */
        float update = k * d;

        //layer_bias_updates[tid] += update;
        atomic_add(layer_bias_updates[tid], update);
    }
}
#endif


__global__ void
train_examples_kernel(const float * feature_vectors,  // feature vector [ni]
                      int feature_vector_width,
                      const int * labels,
                      const float * example_weights,
                      int num_layers,
                      const float * w,  // weights for each layer
                      const float * biases, // for each layer
                      const int * architecture,
                      const int * w_strides,
                      UpdateFloat * const * w_updates, // wt updates for each layer
                      UpdateFloat * const * b_updates, // bias upd for each layer
                      int activation,            // activation function
                      float inhibit, // target value for inhibited neuron)
                      float fire,   // target value for firing neuron
                      float learning_rate,
                      int num_threads_in_block,
                      int num_threads_on_multiprocessor,
                      int total_neurons,
                      float * layer_outputs,  // scratch space[total neurons]
                      int examples_per_block,
                      int total_num_examples,
                      int max_width,
                      bool use_textures)
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
    layer_outputs += block_num * total_neurons;
    
    unsigned example_num_base = block_num * examples_per_block;
    unsigned last_example = min(total_num_examples, example_num_base + examples_per_block);

    unsigned example_num = example_num_base;

#if defined(__DEVICE_EMULATION__)
    const float * weights_access = w;
    const float * biases_access = biases;
#else
    WeightsAccess<weights_tex> weights_access;
    WeightsAccess<biases_tex> biases_access;

    if (!use_textures) {
        weights_access.init(w);
        biases_access.init(biases);
    }
#endif

#if 0
    for (;  example_num < last_example;  example_num += N) {

        const float * input = feature_vectors + example_num * feature_vector_width;
        train_4_examples(input,
                         labels + example_num,
                         example_weights + example_num,
                         last_example - example_num,
                         num_layers, scratch,
                         weights_access, biases_access,
                         architecture, w_strides,
                         w_updates, b_updates,
                         activation, inhibit, fire, learning_rate,
                         num_threads_in_block,
                         num_threads_on_multiprocessor,
                         total_neurons, max_width, layer_outputs);
    }
#elif 0
    // Do any others singly
    for (;  example_num < last_example;  ++example_num) {

        const float * input
            = feature_vectors + example_num * feature_vector_width;

        train_4_examples(input,
                         labels + example_num,
                         example_weights + example_num,
                         1 /* num valid examples */,
                         num_layers, scratch,
                         weights_access, biases_access,
                         architecture, w_strides,
                         w_updates, b_updates,
                         activation, inhibit, fire, learning_rate,
                         num_threads_in_block,
                         num_threads_on_multiprocessor,
                         total_neurons, max_width, layer_outputs);
    }
#else
    // Do any others singly
    for (;  example_num < last_example;  ++example_num) {

        const float * input
            = feature_vectors + example_num * feature_vector_width;

        train_1_example(input,
                        labels + example_num,
                        example_weights + example_num,
                        1 /* num valid examples */,
                        num_layers, scratch,
                        weights_access, biases_access,
                        architecture, w_strides,
                        w_updates, b_updates,
                        activation, inhibit, fire, learning_rate,
                        num_threads_in_block,
                        num_threads_on_multiprocessor,
                        total_neurons, max_width, layer_outputs);
    }
#endif
}


namespace ML {
namespace CUDA {

struct Backprop::Plan {
    int num_layers;

    vector<int> architecture;
    DeviceData<int> d_architecture;

    DeviceData<float> d_weights;

    DeviceData<float> d_biases;

    vector<int> w_strides;
    DeviceData<int> d_w_strides;

    Activation activation;
    float inhibit;
    float fire;
    float learning_rate;

    int max_width;
    int total_neurons;

    // We need our grid size to be exactly the maximum width of the output
    dim3 threads;
    
    int shared_mem_stride;
    size_t shared_mem_size;

    bool use_textures;

    Plan(int num_layers,
         const int * architecture,
         const float * const * weights,
         const float * const * biases,
         const int * w_strides,
         Activation activation,
         float inhibit,
         float fire,
         float learning_rate,
         bool on_host,
         bool use_textures)
        : num_layers(num_layers),
          architecture(architecture, architecture + num_layers + 1),
          w_strides(w_strides, w_strides + num_layers),
          activation(activation),
          inhibit(inhibit),
          fire(fire),
          learning_rate(learning_rate),
          use_textures(use_textures)
    {
        //cerr << "plan: num_layers = " << num_layers << endl;

        d_architecture.init(architecture, num_layers + 1);

        size_t total_weights_size = 0;
        size_t total_bias_size = 0;

        for (unsigned l = 0;  l < num_layers;  ++l) {
            int ni = architecture[l];
            int no = architecture[l + 1];
            int w_stride = w_strides[l];
            total_weights_size += ni * w_stride;
            total_bias_size += no;
            // TODO: align?
        }

        d_weights.init(total_weights_size);
        d_biases.init(total_bias_size);
        
        // Now copy them all in

        size_t weights_start_offset = 0;
        size_t bias_start_offset = 0;
        
        for (unsigned l = 0;  l < num_layers;  ++l) {
            int ni = architecture[l];
            int no = architecture[l + 1];
            int w_stride = w_strides[l];
            size_t w_size = ni * w_stride;

            cudaError_t err
                = cudaMemcpy(d_weights + weights_start_offset,
                             weights[l],
                             w_size * sizeof(float),
                             cudaMemcpyHostToDevice);
            
            if (err != cudaSuccess)
                throw Exception(cudaGetErrorString(err));

            err = cudaMemcpy(d_biases + bias_start_offset,
                             biases[l],
                             no * sizeof(float),
                             cudaMemcpyHostToDevice);
            
            if (err != cudaSuccess)
                throw Exception(cudaGetErrorString(err));
            
            weights_start_offset += ni * w_stride;
            bias_start_offset += no;
            // TODO: align?
        }

        d_w_strides.init(w_strides, num_layers);
        
        max_width = 0;
        total_neurons = 0;

        for (unsigned l = 0;  l <= num_layers;  ++l) {
            max_width = max(max_width, architecture[l]);
            total_neurons += architecture[l];
        }

        // We need our grid size to be exactly the maximum width of the output
        threads = dim3(max_width);

        // Storage for max_width
        shared_mem_stride = max_width * sizeof(float);
        
        // Since we do 4 examples per loop, we need enough memory for all of
        // the four outputs for a single layer
        shared_mem_size = shared_mem_stride * 4;

        if (use_textures) {
            cudaError_t err;
            
            err = cudaBindTexture(0, weights_tex, d_weights);
            if (err != cudaSuccess)
                throw Exception(cudaGetErrorString(err));

            err = cudaBindTexture(0, biases_tex, d_biases);
            if (err != cudaSuccess)
                throw Exception(cudaGetErrorString(err));
        }
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

        // Get the scratch space.  This is 4 in flight examples for each
        // of the concurrent threads.
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
             plan.inhibit,
             plan.fire,
             plan.learning_rate,
             grid.x,
             plan.threads.x,
             plan.total_neurons,
             d_layer_outputs,
             num_examples_per_invocation,
             num_feature_vectors /* total num examples */,
             plan.max_width,
             plan.use_textures);
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

#if 0
            cerr << "first 10 weight updates for layer " << l << ": ";
            for (unsigned i = 0;  i < 10;  ++i)
                cerr << sync_to[i] << " ";
            cerr << endl;
#endif
        }

        for (unsigned l = 0;  l < plan.num_layers;  ++l) {
            int no = plan.architecture[l + 1];

            UpdateFloat sync_to[no];
            
            d_bias_updates_storage[l].sync(sync_to);
            std::copy(sync_to, sync_to + no, bias_updates[l]);

#if 0            
            cerr << "first 10 bias updates for layer " << l << ": ";
            for (unsigned i = 0;  i < 10;  ++i)
                cerr << sync_to[i] << " ";
            cerr << endl;
#endif
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
     float inhibit,
     float fire,
     float learning_rate,
     bool on_host,
     bool use_textures) const
{
    boost::shared_ptr<Plan> result
        (new Plan(num_layers, architecture, weights, biases, w_strides,
                  activation, inhibit, fire, learning_rate, on_host,
                  use_textures));

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
