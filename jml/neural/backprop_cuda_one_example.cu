/* -*- c++ -*- */

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

        /* Add in the layer outputs.  We iterate with all threads */
        
#if 1
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

    }

    // layer_biases is no longer used

    /*************************************************************************/
    /* BPROP                                                                 */
    /*************************************************************************/

    /* How many output layers? */
    this_layer_outputs -= no;

    layer_weights -= ni * w_stride;

    /* First error calculation pass */
    bool correct = (label == tid);
    float wanted = (correct ? fire : inhibit);

    float last_output = scratch[tid];

    __syncthreads();

    scratch[tid] = (tid < no ? wanted - last_output : 0.0);
    
    /* Let everything catch up */
    __syncthreads();

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
