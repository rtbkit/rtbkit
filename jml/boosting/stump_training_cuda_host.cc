/* stump_training_cuda_host.cc
   Jeremy Barnes, 7 April 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

*/

#include "stump_training_cuda.h"
#include "jml/arch/bit_range_ops.h"
#include "jml/arch/tick_counter.h"

typedef ML::CUDA::Test_Buckets_Binsym::Float Float;
typedef ML::CUDA::Test_Buckets_Binsym::TwoBuckets TwoBuckets;

namespace ML {
namespace CUDA {

void executeHostCompressed(TwoBuckets * accum,
                           TwoBuckets & w_label,
                           const float * weights,
                           const float * ex_weights,
                           const uint64_t * data,
                           int bucket_bits,
                           int example_bits,
                           int label_bits,
                           int divisor_bits,
                           size_t size)
{
    ML::Bit_Extractor<uint64_t> index(data);
    
    for (unsigned i = 0;  i < size;  ++i) {
        int bucket
            = index.extract<unsigned>(bucket_bits);
        
        int example = index.extract<unsigned>(example_bits);
        if (example_bits == 0) example = i;

        int label = index.extract<unsigned>(label_bits);
        
        float divisor = index.extract<float>(divisor_bits);

        if (divisor_bits == 0) divisor = 1.0;
        
        float weight = ex_weights[example];
        
        if (weight == 0.0) continue;
        weight *= weights[example];
        if (weight == 0.0) continue;
        
        const float to_add = weight * divisor;
        
        accum[bucket][label] += to_add;
        w_label[label] += to_add;
    }
}

void executeHost(TwoBuckets * accum,
                 TwoBuckets & w_label,
                 const float * weights,
                 const float * ex_weights,
                 const uint16_t * buckets,
                 const uint32_t * examples,
                 const int32_t * labels,
                 const float * divisors,
                 size_t size)
{
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
}

} // namespace CUDA
} // namespace ML
