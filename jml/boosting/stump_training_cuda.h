/* stump_training_cuda.h                                           -*- C++ -*-
   Jeremy Barnes, 29 March 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

   Routines to train stumps for CUDA.  Exports an object that can be used
   to setup and run the CUDA split engines.  This is the API; implementation
   is in the .cu file.
*/

#ifndef __jml__stump_training_cuda_h__
#define __jml__stump_training_cuda_h__

#include <boost/shared_ptr.hpp>
#include "fixed_point_accum.h"

namespace ML {
namespace CUDA {


/*****************************************************************************/
/* TEST_BUCKETS_BINSYM                                                       */
/*****************************************************************************/

struct Test_Buckets_Binsym {

    typedef FixedPointAccum64 Float;
    typedef Float TwoBuckets[2];

    struct Plan;     // Implementation is private
    struct Context;  // implementation is private

    std::shared_ptr<Plan>
    plan(const uint16_t * buckets,
         const uint32_t * examples, // or 0 if example num == i
         const int32_t * labels,
         const float * divisors,
         uint32_t size,
         const float * weights,
         const float * ex_weights,
         int num_buckets,
         bool on_device,
         bool compressed) const;

    std::shared_ptr<Context>
    execute(const Plan & plan,
            TwoBuckets * accum,
            TwoBuckets & w_label) const;

    /** Wait for the given context to be finished. */
    void synchronize(Context & context) const;
};

} // namespace CUDA
} // namespace ML


#endif /* __jml__stump_training_cuda_h__ */

