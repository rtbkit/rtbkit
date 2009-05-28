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
    28 multiprocessors with 8 cores each, and so we could compute 28 different
    samples at once).
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
                     UpdateFloat * const * w_updates, // wt updates for each layer
                     UpdateFloat * const * b_updates, // bias upd for each layer
                     int activation,            // activation function
                     float fire,   // target value for firing neuron
                     float inhibit, // target value for inhibited neuron)
                     float learning_rate,
                     int num_threads_in_block)
{
    // access thread id
    const unsigned tid = threadIdx.x;

    const unsigned example_num  = blockIdx.x;

#ifdef __DEVICE_EMULATION__
    //fprintf(stderr, "tid = %d example_num = %d\n",
    //        tid, example_num);
#endif

    /* Where we accumulate our errors, layer by layer.  The size is that of
       the largest dimension. */
    extern __shared__ float errors[];

    /* The layer outputs (activation of the neurons).  This is where the
       shared memory goes to.  Note that we store only the activated outputs,
       not the inputs.

       blockDim.x gives us the number of threads, which is also the size of
       the errors array, so that our layer outputs have to start at this
       offset.
    */
    float * layer_outputs = errors + blockDim.x;

    const float * input = feature_vectors + example_num * feature_vector_width;

    int label = labels[example_num];

    float example_weight = example_weights[example_num];

#ifdef __DEVICE_EMULATION__
    if (tid == 0 && example_num == 0) {
        fprintf(stderr, "starting fprop example %d wt %f; label %d\n",
                example_num, example_weight, label);

        for (unsigned i = 0;  i < feature_vector_width;  ++i) {
            fprintf(stderr, "input %d: value %f\n",
                    i, input[i]);
        }
    }
#endif


    /*************************************************************************/
    /* FPROP                                                                 */
    /*************************************************************************/

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

#if defined(__DEVICE_EMULATION__) && 0
        if (tid == 0)
            fprintf(stderr, "fprop: tid %d layer %d ni %d no %d last_layer_outputs %p this_layer_outputs %p\n",
                    tid, l, ni, no, last_layer_outputs, this_layer_outputs);
#endif

#if 0
    std::copy(bias.begin(), bias.end(), output.begin());
    for (unsigned i = 0;  i < input.size();  ++i)
        SIMD::vec_add(&output[0], input[i], &weights[i][0], &output[0],
                      outputs());
        //for (unsigned o = 0;  o < output.size();  ++o)
        //    output[o] += input[i] * weights[i][o];
    transform(output);
#endif

        /* Add in the layer outputs.  We iterate with all threads */
        if (tid < no) {
            // Start off with the bias terms
            double accum = biases[l][tid];

            for (unsigned i = 0;  i < ni;  ++i) {
                float inval = (l == 0 ? input[i] : last_layer_outputs[i]);

                // Coalesced access; maybe texture would be better
                float weight = layer_weights[i * w_stride + tid];
                
                accum += weight * inval;
            }
            
            this_layer_outputs[tid] = transform(accum, activation);
        }

#if defined(__DEVICE_EMULATION__)
        __syncthreads();
        if (tid == 0 && example_num == 0) {
            fprintf(stderr, "completed fprop layer %d example %d; label %d\n",
                    l, example_num, label);
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
    errors[tid] = (tid < no ? wanted - this_layer_outputs[tid] : 0.0);
    
    /* Let everything catch up */
    __syncthreads();


#if defined(__DEVICE_EMULATION__)
    if (tid == 0 && example_num == 0) {
        fprintf(stderr, "completed fprop example %d; label %d\n",
                example_num, label);
        for (unsigned i = 0;  i < no;  ++i) {
            fprintf(stderr, "output %d: value %f error %f correct %d\n",
                    i, this_layer_outputs[i], errors[i], (label == i));
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
        
        float prev_output = (tid > no ? 0.0 : last_layer_outputs[tid]);

        if (prev_output > 1.0) prev_output *= 1.0000001;

        // 
        float error = errors[tid];
        
        float d = (tid > no ? 0.0 : delta(prev_output, error, activation));
        float d2 = 0.0;

        if (l > 0) {
            // Make sure all threads have caught up so that we can modify error
            // without affecting them
            __syncthreads();

            // Broadcast the d values so that we can use them to calculate the
            // errors
            errors[tid] = d;

            // Make sure everything can get its d value
            __syncthreads();

            // Get the d value for the next thread, for the updates
            if (tid < no - 1) d2 = errors[tid + 1];
            
            double total = 0.0;
            if (tid < ni) {
                for (unsigned o = 0;  o < no;  ++o) {
                    float d = errors[o];  // may be the d from another thread
                    float update = d * layer_weights[tid * w_stride + o];
                    total += update;
                }
            }

            // Wait for everything to finish so that we can overwrite the d
            // values with the new errors
            __syncthreads();
            
            errors[tid] = total;
        }


#if defined(__DEVICE_EMULATION__)
        __syncthreads();

        if (tid == 0 && example_num == 0) {
            fprintf(stderr, "completed error propagation layer %d\n",
                    l);
            for (unsigned i = 0;  i < ni;  ++i) {
                fprintf(stderr, "input %d: error %f\n",
                        i, errors[i]);
            }
        }
#endif


        // Again, threads indexed too low just leave
        if (tid >= no) continue;

        /* Update the weights. */
        float k = example_weight * learning_rate;

#if defined(__DEVICE_EMULATION__) && 0
        if (tid == 0)
            fprintf(stderr, "bprop: tid %d layer %d ni %d no %d last_layer_outputs %p this_layer_outputs %p\n",
                    tid, l, ni, no, last_layer_outputs, this_layer_outputs);
#endif

        /* Now for the updates.  In order to avoid trying to write the same
           memory over and over, we stagger the starting points so that
           each example will start at a different place, thus minimising
           conflicting writes when we have multiple multiprocessors working
           on the same thing. */

        int thread_stride = ni / num_threads_in_block;
        if (thread_stride == 0) thread_stride = 1;

        int start_at = (example_num * thread_stride) % ni;

        for (unsigned i_ = start_at;  i_ < ni + start_at;  ++i_) {
            if (tid % 2 == 1) continue;

            // Get the real index of i
            unsigned i = i_ - (i_ >= ni) * ni;
            
            float prev = (l == 0 ? input[i] : last_layer_outputs[i]); 
            float update = prev * k * d;
            float update2 = prev * k * d2;

            ML::FixedPointAccum32 upd1(update), upd2(update2);

            unsigned long long two_updates
                = (((unsigned long long)upd2.rep) << 32)
                | upd1.rep;
            
            atomicAdd((unsigned long long *)(layer_updates
                                             + i * w_stride + tid),
                      two_updates);
            
            //atomic_add(layer_updates[i * w_stride + tid], update);
        }
        
        /* Update the bias */
        float update = k * d;

        //layer_bias_updates[tid] += update;
        atomic_add(layer_bias_updates[tid], update);
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

        shared_mem_size = (max_width + total_neurons) * sizeof(float);
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

    dim3 grid;

    int num_feature_vectors;
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
            d_weight_updates_storage[l].init(ni * w_stride);
            weight_updates_vec[l] = d_weight_updates_storage[l];
        }

        d_weight_updates.init(&weight_updates_vec[0], plan.num_layers);

        d_bias_updates_storage.resize(plan.num_layers);
        bias_updates_vec.resize(plan.num_layers);

        for (unsigned l = 0;  l < plan.num_layers;  ++l) {
            int no = plan.architecture[l + 1];
            d_bias_updates_storage[l].init(no);
            bias_updates_vec[l] = d_bias_updates_storage[l];
        }

        d_bias_updates.init(&bias_updates_vec[0], plan.num_layers);

        int total_inputs = 0;
        for (unsigned l = 0;  l < plan.num_layers;  ++l) {
            int ni = plan.architecture[l];
            total_inputs += ni;
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
             num_feature_vectors);

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
