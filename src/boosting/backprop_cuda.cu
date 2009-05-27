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


using namespace std;


/** Given an activation function and an input, apply that activation
    function */
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
    28 multiprocessors with 8 cores each, and so we could compute 28 different
    samples at once).
    
    This kernel will be called with ONE block of threads, with the number
    of threads equal to the widest layer that there is.
*/
__global__ void
train_example_kernel(const float * feature_vectors,  // feature vector [ni]
                     int feature_vector_width,
                     const int * labels,
                     const float * example_weights,
                     int num_layers,
                     const float * const * w,  // weights for each layer
                     const float * const * biases, // for each layer
                     const int * architecture,
                     const int * w_strides,
                     float * const * w_updates, // wt updates for each layer
                     float * const * b_updates, // bias upd for each layer
                     int activation,            // activation function
                     float fire,   // target value for firing neuron
                     float inhibit, // target value for inhibited neuron)
                     float learning_rate)
{
    // access thread id
    const unsigned tid = threadIdx.x;

    // 
    const unsigned example_num  = blockIdx.x;

    /* The layer outputs (activation of the neurons).  This is where the
       shared memory goes to.  Note that we store only the activated outputs,
       not the inputs. */
    extern __shared__ float layer_outputs[];

    /* Where we accumulate our errors, layer by layer.  The size is that of
       the largest dimension. */
    extern __shared__ float errors[];

    const float * input = feature_vectors + example_num * feature_vector_width;

    int label = labels[example_num];

    float example_weight = example_weights[example_num];

    
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
    for (int l = num_layers - 1;  l >= 1;
         --l,
             __syncthreads(),
             last_layer_outputs = this_layer_outputs,
             this_layer_outputs = next_layer_outputs) {
        
        // Get information about the layer:
        int ni = architecture[l];
        int no = architecture[l + 1];

        const float * layer_weights = w[l];
        int w_stride = w_strides[l];

        float * layer_updates = w_updates[l];
        float * layer_bias_updates  = b_updates[l];
        
        next_layer_outputs = this_layer_outputs - no;
        
        if (tid >= no) continue;
        
        float d = delta(last_layer_outputs[tid], errors[tid], activation);

        /* Calculate the new error terms for the next layer */
        // TODO: atomic... and then find a way to avoid data dependencies...
        if (l > 1)
            for (unsigned i = 0;  i < ni;  ++i) {
                float update = d * layer_weights[i * w_stride + tid];
                //errors[i] += update;
                atomic_add(errors[i], update); 
            }


        /* Update the weights. */
        float k = example_weight * learning_rate;
        for (unsigned i = 0;  i < ni;  ++i) {
            // No bank conflicts here as all threads are reading with the same
            // i value
            float update = last_layer_outputs[i] * k * d;

            //layer_updates[i * w_stride + tid] += update;
            atomic_add(layer_updates[i * w_stride + tid], update);
        }
        
        /* Update the bias */
        //layer_bias_updates[tid] += k * d;
        atomic_add(layer_bias_updates[tid], k * d);
    }
}

namespace ML {
namespace CUDA {

struct Backprop::Plan {
    int num_layers;

    vector<int> architecture;
    DeviceData<int> d_architecture;

    vector<DeviceData<float> > d_weights_storage;
    vector<const float *> d_weights;

    vector<DeviceData<float> > d_biases_storage;
    vector<const float *> d_biases;

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
        d_architecture.init(architecture, num_layers + 1);

        d_weights_storage.resize(num_layers);
        d_weights.resize(num_layers);

        for (unsigned l = 0;  l < num_layers;  ++l) {
            int no = architecture[l + 1];
            int w_stride = w_strides[l];
            d_weights_storage[l].init(weights[l], no * w_stride);
            d_weights[l] = d_weights_storage[l];
        }
    
        d_biases_storage.resize(num_layers);
        d_biases.resize(num_layers);

        for (unsigned l = 0;  l < num_layers;  ++l) {
            int no = architecture[l + 1];
            d_biases_storage[l].init(biases[l], no);
            d_biases[l] = d_biases_storage[l];
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

        shared_mem_size = (max_width + total_neurons) * sizeof(float);
    }
};

struct Backprop::Context {

    const Plan & plan;
    
    DeviceData<float> d_feature_vectors;
    DeviceData<float> d_example_weights;
    DeviceData<int> d_labels;
        
    float * const * weight_updates;
    vector<DeviceData<float> > d_weight_updates_storage;
    float * const * bias_updates;
    vector<float *> d_weight_updates;
    
    vector<DeviceData<float> > d_bias_updates_storage;
    vector<float *> d_bias_updates;

    dim3 grid;

    int feature_vector_width;

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
        : plan(plan), weight_updates(weight_updates), bias_updates(bias_updates)
    {
        feature_vector_width = plan.architecture[0];
        
        d_feature_vectors.init(feature_vectors,
                               num_feature_vectors * feature_vector_width);
        
        d_example_weights.init(example_weights, num_feature_vectors);
        
        d_labels.init(labels, num_feature_vectors);
        
        d_weight_updates_storage.resize(plan.num_layers);
        
        for (unsigned l = 0;  l < plan.num_layers;  ++l) {
            int no = plan.architecture[l + 1];
            int w_stride = plan.w_strides[l];
            d_weight_updates_storage[l].init(weight_updates[l],
                                             no * w_stride);
            d_weight_updates[l] = d_weight_updates_storage[l];
        }

        d_bias_updates_storage.resize(plan.num_layers);

        for (unsigned l = 0;  l < plan.num_layers;  ++l) {
            int no = plan.architecture[l + 1];
            d_bias_updates_storage[l].init(bias_updates[l], no);
            d_bias_updates[l] = d_bias_updates_storage[l];
        }

        // Our grid size is one per example
        grid = dim3(num_feature_vectors);
    }

    void execute()
    {
        train_example_kernel<<<grid, plan.threads, plan.shared_mem_size>>>
            (d_feature_vectors,
             feature_vector_width,
             d_labels,
             d_example_weights,
             plan.num_layers,
             &plan.d_weights[0],
             &plan.d_biases[0],
             plan.d_architecture,
             plan.d_w_strides,
             &d_weight_updates[0],
             &d_bias_updates[0],
             plan.activation,
             plan.fire,
             plan.inhibit,
             plan.learning_rate);
    }
    
    void synchronize()
    {
        cudaError_t err = cudaThreadSynchronize();
        
        if (err != cudaSuccess)
            throw Exception(cudaGetErrorString(err));
        
        for (unsigned l = 0;  l < plan.num_layers;  ++l)
            d_weight_updates_storage[l].sync(weight_updates[l]);
        for (unsigned l = 0;  l < plan.num_layers;  ++l)
            d_bias_updates_storage[l].sync(bias_updates[l]);
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
