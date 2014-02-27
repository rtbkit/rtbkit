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

extern "c" __sync_lock_test_and_set(...);

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

/** Train a fully-connected neural network architecture via backpropagation
    one a single training example.  The work is split over all of the cores
    within a single multiprocessor.  (So, on a Geforce 260 core 216, we have
    27 multiprocessors with 8 cores each, and so we could train on 27 different
    feature vectors in parallel.
*/

#define N 4
#define train_N_examples train_4_examples
#include "backprop_cuda_train_N_examples.cu"
#undef N
#undef train_N_examples

#define N 1
#define train_N_examples train_1_example
#include "backprop_cuda_train_N_examples.cu"
#undef N
#undef train_N_examples

#include "backprop_cuda_one_example.cu"

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
                      UpdateFloat * const * w_updates2, // wt updates for each layer
                      UpdateFloat * const * b_updates2, // bias upd for each layer
                      int activation,            // activation function
                      float inhibit, // target value for inhibited neuron)
                      float fire,   // target value for firing neuron
                      float learning_rate,
                      int num_threads_in_block,
                      int num_threads_on_multiprocessor,
                      int total_neurons,
                      float * layer_outputs,  // scratch space[total neurons]
                      float * layer_outputs2,  // scratch space[total neurons]
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

    WeightsAccess<weights_tex> weights_access;
    WeightsAccess<biases_tex> biases_access;

    if (!use_textures) {
        weights_access.init(w);
        biases_access.init(biases);
    }

#if 1
    for (;  example_num < last_example;  example_num += 4) {

        const float * input = feature_vectors + example_num * feature_vector_width;
        train_4_examples(input,
                         labels + example_num,
                         example_weights + example_num,
                         min(4, last_example - example_num),
                         num_layers, scratch,
                         weights_access, biases_access,
                         architecture, w_strides,
                         w_updates, b_updates,
                         activation, inhibit, fire, learning_rate,
                         num_threads_in_block,
                         num_threads_on_multiprocessor,
                         total_neurons, max_width, layer_outputs);

        train_4_examples(input,
                         labels + example_num,
                         example_weights + example_num,
                         min(4, last_example - example_num),
                         num_layers, scratch,
                         weights_access, biases_access,
                         architecture, w_strides,
                         w_updates2, b_updates2,
                         activation, inhibit, fire, learning_rate,
                         num_threads_in_block,
                         num_threads_on_multiprocessor,
                         total_neurons, max_width, layer_outputs2);
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
#endif
    // Do any others singly
    for (;  example_num < last_example;  ++example_num) {

        const float * input
            = feature_vectors + example_num * feature_vector_width;

#if 1
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

        train_1_example(input,
                        labels + example_num,
                        example_weights + example_num,
                        1 /* num valid examples */,
                        num_layers, scratch,
                        weights_access, biases_access,
                        architecture, w_strides,
                        w_updates2, b_updates2,
                        activation, inhibit, fire, learning_rate,
                        num_threads_in_block,
                        num_threads_on_multiprocessor,
                        total_neurons, max_width, layer_outputs);
#else
        train_example(input,
                      labels[example_num],
                      example_weights[example_num],
                      num_layers, scratch,
                      weights_access, biases_access,
                      architecture, w_strides,
                      w_updates, b_updates,
                      activation, inhibit, fire, learning_rate,
                      num_threads_in_block,
                      num_threads_on_multiprocessor,
                      total_neurons, layer_outputs);
#endif
    }
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

    vector<DeviceData<UpdateFloat> > d_weight_updates2_storage;
    vector<UpdateFloat *> weight_updates2_vec;
    DeviceData<UpdateFloat *> d_weight_updates2;
    
    vector<DeviceData<UpdateFloat> > d_bias_updates2_storage;
    vector<UpdateFloat *> bias_updates2_vec;
    DeviceData<UpdateFloat *> d_bias_updates2;

    DeviceData<float> d_layer_outputs;
    DeviceData<float> d_layer_outputs2;

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


        d_weight_updates2_storage.resize(plan.num_layers);
        weight_updates2_vec.resize(plan.num_layers);
        
        for (unsigned l = 0;  l < plan.num_layers;  ++l) {
            int ni = plan.architecture[l];
            int w_stride = plan.w_strides[l];
            d_weight_updates2_storage[l].init_zeroed(ni * w_stride);
            weight_updates2_vec[l] = d_weight_updates2_storage[l];
        }

        d_weight_updates2.init(&weight_updates2_vec[0], plan.num_layers);

        d_bias_updates2_storage.resize(plan.num_layers);
        bias_updates2_vec.resize(plan.num_layers);

        for (unsigned l = 0;  l < plan.num_layers;  ++l) {
            int no = plan.architecture[l + 1];
            d_bias_updates2_storage[l].init_zeroed(no);
            bias_updates2_vec[l] = d_bias_updates2_storage[l];
        }

        d_bias_updates2.init(&bias_updates2_vec[0], plan.num_layers);



        num_examples_per_invocation = 4;//16;

        int grid_size = rudiv(num_feature_vectors, num_examples_per_invocation);

        // Get the scratch space.  This is 4 in flight examples for each
        // of the concurrent threads.
        d_layer_outputs.init(plan.total_neurons * grid_size * 4);

        d_layer_outputs2.init(plan.total_neurons * grid_size * 4);
        
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
             d_weight_updates2,
             d_bias_updates2,
             plan.activation,
             plan.inhibit,
             plan.fire,
             plan.learning_rate,
             grid.x,
             plan.threads.x,
             plan.total_neurons,
             d_layer_outputs,
             d_layer_outputs2,
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

        /* Copy back the layer outputs */
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
