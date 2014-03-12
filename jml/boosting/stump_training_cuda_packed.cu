/* stump_training_cuda.cu                                          -*- C++ -*-
   CUDA version of stump training code.
 */                                                                

#include "arch/exception.h"
#include "compiler/compiler.h"
#include <cstdio>
#include <iostream>
#include <boost/timer.hpp>
#include <boost/utility.hpp>
#include <boost/scoped_array.hpp>
#include "stump_training_cuda.h"
#include "fixed_point_accum.h"
#include "arch/cuda/device_data.h"

using namespace std;

typedef ML::CUDA::Test_Buckets_Binsym::Float Float;
typedef ML::CUDA::Test_Buckets_Binsym::TwoBuckets TwoBuckets;

/** Execution kernel

    Parameters:
    - example_data: structure (packed into memory: 4-12 bytes per entry, size
      is number of feature occurrences)
      uint16_t bucket
      uint16_t label
      uint32_t example num (if not exactly one per example)
      float    divisor (if not exactly one per label)

    - buckets: structure, in shared memory, size is number of buckets
      - double true_corr: total count for true/correct bucket
      - double true_incorr: total count for true/incorrect bucket
      - Will eventually have more to minimise amount of contention on the
        shared buckets

    - Splitting up:
      - 512 threads per block
      - 64 entries per thread
      - 32,768 split points accumulated per block

    - Algorithm
      - Read and extract bucket, label, example, divisor (4 at once, for 16-48 bytes to be read)
      - Read 4 weights (16 bytes)
      - Accumulate in shared memory
      - Continue until block is finished
      
    - To void variability
      - Weights are accumulated in 64 bit integers
      - Use two floats (one to hold the first 23 bits of mantissa, the other
        for the other 23 bits)

    - To avoid using double precision
      - 
*/

/* Divide, but round up */
template<class X, class Y>
__device__ __host__
X rudiv(X val, Y by)
{
    X result = (val / by);
    X missing = val - (result * by);
    result += (missing > 0);
    return result;
}

// Texture.  Please note that this is global, so we can't be working on more
// than one thing at once.  Test only.
texture<float, 1, cudaReadModeElementType> weights_tex;
texture<float, 1, cudaReadModeElementType> ex_weights_tex;
texture<float4, 1, cudaReadModeElementType> weights_tex4;
texture<float4, 1, cudaReadModeElementType> ex_weights_tex4;

__global__ void
testKernel(const uint16_t * buckets,
           const uint32_t * examples, // or 0 if example num == i
           const int32_t * labels,
           const float * divisors,  // or 0 if always equal to 1
           uint32_t size,
           const float * weights,
           const float * ex_weights,
           TwoBuckets * buckets_global,
           TwoBuckets * w_label_global_,
           int num_buckets,
           int bucket_expansion,
           int num_todo,
           bool use_texture)
{
    // access thread id
    const unsigned tid = threadIdx.x;

    // access number of threads in this block
    const unsigned num_threads = blockDim.x;

    // Access where the block starts
    const unsigned block_num = blockIdx.x;

    unsigned offset = (block_num * (num_threads * num_todo));

    // shared memory.  Our buckets are accumulated in this.
    extern  __shared__  TwoBuckets shared_data[];
    TwoBuckets * buckets_shared = shared_data + 1;

    TwoBuckets * w_label_shared = shared_data;

    TwoBuckets * w_label_global = w_label_global_;

    bucket_expansion = 1;

    int buckets_allocated = num_buckets * bucket_expansion;

    int expansion_offset = tid % bucket_expansion;

    // Initialization of shared (across threads)

    for (unsigned i = tid;  i < buckets_allocated;  i += num_threads)
        buckets_shared[i][0] = buckets_shared[i][1] = 0.0f;

    if (tid == 0) w_label_shared[0][0] = w_label_shared[0][1] = 0.0f;
    
    // Wait for the initialization to be finished
    __syncthreads();


    unsigned start_at = offset + tid;
    
    Float w_label_true(0.0f), w_label_false(0.0f);

    // Optimize for where examples == 0, which means we access everything
    // with unit stride

    if (examples == 0) {

        // We modify so that 4 values are done by tid 0, then 4 by tid 1,
        // and so on so that out memory accesses really read 4 values.
        start_at = offset + (tid * 4);
        const float4 uniform_divisors = make_float4(1.0f, 1.0f, 1.0f, 1.0f);

        for (unsigned i = 0;  i < num_todo;  i += 4) {
            int example = start_at + i * num_threads;

            if (example >= size) break;
            
            float4 weight, ex_weight;
            
            if (use_texture) {
                ex_weight.x = tex1Dfetch(ex_weights_tex, example);
                ex_weight.y = tex1Dfetch(ex_weights_tex, example + 1);
                ex_weight.z = tex1Dfetch(ex_weights_tex, example + 2);
                ex_weight.w = tex1Dfetch(ex_weights_tex, example + 3);

                //ex_weight = tex1Dfetch(ex_weights_tex4, example);

                weight.x = tex1Dfetch(weights_tex, example);
                weight.y = tex1Dfetch(weights_tex, example + 1);
                weight.z = tex1Dfetch(weights_tex, example + 2);
                weight.w = tex1Dfetch(weights_tex, example + 3);

                //weight = tex1Dfetch(weights_tex4, example);
            }
            else if (use_texture  && false /* doesn't work; most get zero */) {
                ex_weight = tex1Dfetch(ex_weights_tex4, example);
                weight = tex1Dfetch(weights_tex4, example);
            }
            else {
                ex_weight = *(const float4 *)(&ex_weights[example]);
                weight = *(const float4 *)(&weights[example]);
            }

            weight.x *= ex_weight.x;
            weight.y *= ex_weight.y;
            weight.z *= ex_weight.z;
            weight.w *= ex_weight.w;
            
            const int4 label = *(const int4 *)(&labels[example]);
            
            const float4 divisor = (divisors == 0 ? uniform_divisors
                                    : *(const float4 *)(divisors + example));
            
            const short4 real_bucket = *(const short4 *)(buckets + example);

            float to_add;
            int bucket;

            // First update (x)
            to_add = weight.x * divisor.x;
            bucket = real_bucket.x * bucket_expansion + expansion_offset;
            atomic_add_shared(buckets_shared[bucket][label.x], to_add);
            if (label.x) w_label_true += to_add;
            else w_label_false += to_add;
            if (example + 1 == size) break;
            
            // Second update (y)
            to_add = weight.y * divisor.y;
            bucket = real_bucket.y * bucket_expansion + expansion_offset;
            atomic_add_shared(buckets_shared[bucket][label.y], to_add);
            if (label.y) w_label_true += to_add;
            else w_label_false += to_add;
            if (example + 2 == size) break;
            
            // Third update (z)
            to_add = weight.z * divisor.z;
            bucket = real_bucket.z * bucket_expansion + expansion_offset;
            atomic_add_shared(buckets_shared[bucket][label.z], to_add);
            if (label.z) w_label_true += to_add;
            else w_label_false += to_add;
            if (example + 3 == size) break;
            
            // Fourth update (w)
            to_add = weight.w * divisor.w;
            bucket = real_bucket.w * bucket_expansion + expansion_offset;
            atomic_add_shared(buckets_shared[bucket][label.w], to_add);
            if (label.w) w_label_true += to_add;
            else w_label_false += to_add;
        }
    }
    else {
        for (unsigned i = 0;  i < num_todo;  ++i) {
            int index = start_at + i * num_threads;
            
            if (index >= size) break;
            
            int example = (examples == 0 ? index : examples[index]);
            
            float weight;
            
            if (use_texture) {
                weight = tex1Dfetch(ex_weights_tex, example);
                if (weight == 0.0) continue;
                weight *= tex1Dfetch(weights_tex, example);
            }
            else {
                weight = ex_weights[example];
                if (weight == 0.0) continue;
                weight *= weights[example];
            }
            if (weight == 0.0) continue;
            
            const int label = labels[index];
            
            const float divisor = (divisors == 0 ? 1.0f : divisors[index]);
            
            const int real_bucket = buckets[index];
            const int bucket = real_bucket * bucket_expansion + expansion_offset;
            
            const float to_add = weight * divisor;
            const float to_add_true  = (label ? to_add : 0.0f);
            const float to_add_false = (label ? 0.0f : to_add);
            
            atomic_add_shared(buckets_shared[bucket][label], to_add);
            
            w_label_true  += to_add_true;
            w_label_false += to_add_false;
        }
    }
        
    /* Accumulate the total removed field */
    atomic_add_shared(w_label_shared[0][0], w_label_false);
    atomic_add_shared(w_label_shared[0][1], w_label_true);
    
    /* Wait until all shared is done */
    __syncthreads();

    /* Update the global results using atomic additions (one thread only) */
    for (unsigned i = tid;  i < buckets_allocated;  i += num_threads) {
        int real_bucket = i / bucket_expansion;
        atomic_add(buckets_global[real_bucket][0], buckets_shared[i][0]);
        atomic_add(buckets_global[real_bucket][1], buckets_shared[i][1]);
    }
    
    if (tid == 0) {
        atomic_add(w_label_global[0][0], w_label_shared[0][0]);
        atomic_add(w_label_global[0][1], w_label_shared[0][1]);
    }
}

namespace ML {
namespace CUDA {


/*****************************************************************************/
/* TEST_BUCKETS_BINSYM                                                       */
/*****************************************************************************/

struct Test_Buckets_Binsym::Context {
    const Plan * plan;
    
    TwoBuckets * accum;
    TwoBuckets * w_label;
    DeviceData<TwoBuckets>  d_accum;
    DeviceData<TwoBuckets>  d_w_label;
    bool on_device;

    void synchronize()
    {
        if (on_device) {
            
            cudaError_t err = cudaThreadSynchronize();
            
            if (err != cudaSuccess)
                throw Exception(cudaGetErrorString(err));
            
            
            d_accum.sync(accum);
            d_w_label.sync(w_label);
        }

#if 1
        cerr << "final results: " << endl;
        for (unsigned i = 0;  i < 2 /*num_buckets*/;  ++i)
            cerr << "bucket " << i << ": 0: " << accum[i][0]
                 << "  1: " << accum[i][1] << endl;
#endif
        cerr << "w_label: 0: " << w_label[0][0] << " 1: " << w_label[0][1]
             << endl;
    }
};

struct Test_Buckets_Binsym::Plan {

    Plan(const uint16_t * buckets,
         const uint32_t * examples, // or 0 if example num == i
         const int32_t * labels,
         const float * divisors,
         uint32_t size,
         const float * weights,
         const float * ex_weights,
         int num_buckets,
         bool on_device)
        : buckets(buckets), examples(examples), labels(labels),
          divisors(divisors), size(size), weights(weights),
          ex_weights(ex_weights), num_buckets(num_buckets),
          on_device(on_device)
          
    {
        if (!buckets)
            throw Exception("no buckets");
        
        if (!on_device) return;  // nothing to set up if running on host

        // How many concurrent threads are launched at the same time to work
        // together?  If this number is too high, then we will have contention
        // on the shared memory (the probability that we update multiple buckets
        // at once increases).  If the number is too low, then we won't be able
        // to launch many concurrent blocks.
        threads  = dim3( 128, 1, 1);
        
        // How many does each thread block do?
        num_todo = 32;
        
        // How many of these thread blocks?
        grid = dim3( rudiv(size, threads.x * num_todo));
        
        cerr << "num_todo = " << num_todo << endl;
        cerr << "grid: x = " << grid.x << endl;
        
        // If there aren't enough buckets, then create some more and merge
        // them together at the end.
        // This helps to avoid bank conflicts.
        buckets_to_allocate = num_buckets;
        bucket_expansion = 1;

        if (num_buckets < 8) {
            bucket_expansion = (16 / num_buckets) + 1;
            buckets_to_allocate = num_buckets * bucket_expansion;
        }
        
        cerr << "num_buckets = " << num_buckets << " bucket_expansion = "
             << bucket_expansion << " buckets_to_allocate = "
             << buckets_to_allocate << endl;

        // How much shared memory?
        // We need:
        // - 2 doubles for each bucket;
        // - 2 doubles for the total
        // Note that this effectively limits us to 1023 buckets, as we only
        // have 16kb of shared memory available, and this will severely limit
        // parallelism.
        shared_mem_size = sizeof(Float) * 2 * (buckets_to_allocate + 1);
        
        cerr << "shared_mem_size = " << shared_mem_size << endl;
        
        d_buckets.init(buckets, size);
        d_examples.init(examples, size);
        d_labels.init(labels, size);
        d_divisors.init(divisors, size);
        d_weights.init(weights, size);
        d_ex_weights.init(ex_weights, size);

        // set texture parameters
        cudaError_t err;
            
        err = cudaBindTexture(0, weights_tex, d_weights /*, d_weights.num_bytes()*/);
        if (err != cudaSuccess)
            throw Exception(cudaGetErrorString(err));
                
        err = cudaBindTexture(0, ex_weights_tex, d_ex_weights /* , d_ex_weights.num_bytes()*/);
        if (err != cudaSuccess)
            throw Exception(cudaGetErrorString(err));

        err = cudaBindTexture(0, weights_tex4, d_weights /*, d_weights.num_bytes()*/);
        if (err != cudaSuccess)
            throw Exception(cudaGetErrorString(err));
                
        err = cudaBindTexture(0, ex_weights_tex4, d_ex_weights /*, d_ex_weights.num_bytes()*/);
        if (err != cudaSuccess)
            throw Exception(cudaGetErrorString(err));

        use_texture = true;
    }

    const uint16_t * buckets;
    const uint32_t * examples;
    const int32_t * labels;
    const float * divisors;
    uint32_t size;
    const float * weights;
    const float * ex_weights;
    int num_buckets;
    bool on_device;

    dim3 threads;
    int num_todo;
    dim3  grid;

    int buckets_to_allocate;
    int bucket_expansion;
    int shared_mem_size;

    DeviceData<uint16_t> d_buckets;
    DeviceData<uint32_t> d_examples;
    DeviceData<int32_t>  d_labels;
    DeviceData<float>    d_divisors;
    DeviceData<float>    d_weights;
    DeviceData<float>    d_ex_weights;

    bool use_texture;

    boost::shared_ptr<Context>
    executeHost(TwoBuckets * accum,
                TwoBuckets & w_label) const
    {
        boost::shared_ptr<Context> result(new Context());
        //result->plan = this;

        // Get the data structures
        result->on_device = false;
        result->accum = accum;
        result->w_label = &w_label;

        for (unsigned i = 0;  i < size;  ++i) {
            int index = i;
            int example = (examples == 0 ? index : examples[index]);
            float weight = ex_weights[example];
            
            if (weight == 0.0) continue;
            weight *= weights[example];
            if (weight == 0.0) continue;
            
            const int label = labels[index];
            
            const float divisor = (divisors == 0 ? 1.0f : divisors[index]);
            
            const int bucket = buckets[index];
            
            const float to_add = weight * divisor;

            accum[bucket][label] += to_add;
            w_label[label] += to_add;
        }

        return result;
    }

    boost::shared_ptr<Context>
    executeDevice(TwoBuckets * accum,
                  TwoBuckets & w_label) const
    {
        boost::shared_ptr<Context> result(new Context());
        //result->plan = this;

        // Get the data structures
        result->d_accum.init(accum, num_buckets);
        result->d_w_label.init(&w_label, 1);
        result->on_device = true;
        result->accum = accum;
        result->w_label = &w_label;

        // execute the kernel
        testKernel<<< grid, threads, shared_mem_size >>>
            ( d_buckets, d_examples, d_labels, d_divisors,
              size, d_weights, d_ex_weights,
              result->d_accum, result->d_w_label,
              num_buckets, bucket_expansion, num_todo,
              use_texture);
        
        cudaError_t err = cudaGetLastError();
        
        if (err != cudaSuccess)
            throw Exception(cudaGetErrorString(err));

        return result;
    }

    boost::shared_ptr<Context>
    execute(TwoBuckets * accum,
            TwoBuckets & w_label) const
    {
        // Clear result to start with
        for (unsigned i = 0;  i < num_buckets;  ++i)
            accum[i][0] = accum[i][1] = 0.0;
        w_label[0] = w_label[1] = 0.0;
    
        if (on_device)
            return executeDevice(accum, w_label);
        else return executeHost(accum, w_label);
    }
};

boost::shared_ptr<Test_Buckets_Binsym::Plan>
Test_Buckets_Binsym::
plan(const uint16_t * buckets,
     const uint32_t * examples, // or 0 if example num == i
     const int32_t * labels,
     const float * divisors,
     uint32_t size,
     const float * weights,
     const float * ex_weights,
     int num_buckets,
     bool on_device) const
{
    return boost::shared_ptr<Test_Buckets_Binsym::Plan>
        (new Plan(buckets, examples, labels, divisors, size, weights,
                  ex_weights, num_buckets, on_device));
}

boost::shared_ptr<Test_Buckets_Binsym::Context>
Test_Buckets_Binsym::
execute(const Plan & plan,
        TwoBuckets * accum,
        TwoBuckets & w_label) const
{
    return plan.execute(accum, w_label);
}

void
Test_Buckets_Binsym::
synchronize(Context & context) const
{
    context.synchronize();
}


} // namespace CUDA
} // namespace ML
