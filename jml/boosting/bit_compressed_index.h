/* bit_compressed_index.h                                          -*- C++ -*-
   Jeremy Barnes, 7 April 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

   A training index, but compressed so that memory bandwidth is reduced to
   a minimum.
*/

#ifndef __boosting__bit_compressed_index_h__
#define __boosting__bit_compressed_index_h__

#include <boost/shared_array.hpp>
#include <stdint.h>

namespace ML {

/*****************************************************************************/
/* BIT_COMPRESSED_INDEX                                                      */
/*****************************************************************************/

struct Bit_Compressed_Index {
    Bit_Compressed_Index();

    Bit_Compressed_Index(const uint16_t * buckets,
                         const uint32_t * examples,
                         const int32_t * labels,
                         const float * divisors,
                         uint32_t size);

    void init(const uint16_t * buckets,
              const uint32_t * examples,
              const int32_t * labels,
              const float * divisors,
              uint32_t size);

    boost::shared_array<uint64_t> data;
    int bucket_bits;
    int example_bits;
    int label_bits;
    int divisor_bits;
    int total_bits;
    size_t size;
    size_t num_words;
};

} // namespace ML

#endif /* __boosting__bit_compressed_index_h__ */
