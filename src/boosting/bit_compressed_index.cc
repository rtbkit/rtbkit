/* bit_compressed_index.cc                                         -*- C++ -*-
   Jeremy Barnes, 7 April 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

   Index, compressed bit by bit.
*/

#include "bit_compressed_index.h"
#include "jml/arch/bitops.h"
#include "jml/arch/bit_range_ops.h"
#include "jml/math/xdiv.h"
#include <iostream>


using namespace std;


namespace ML {

/*****************************************************************************/
/* BIT_COMPRESSED_INDEX                                                      */
/*****************************************************************************/

Bit_Compressed_Index::
Bit_Compressed_Index()
    : bucket_bits(0), example_bits(0), label_bits(0),
      divisor_bits(0), total_bits(0), size(0), num_words(0)
{
}

Bit_Compressed_Index::
Bit_Compressed_Index(const uint16_t * buckets,
                     const uint32_t * examples,
                     const int32_t * labels,
                     const float * divisors,
                     uint32_t size)
{
    init(buckets, examples, labels, divisors, size);
}

void
Bit_Compressed_Index::
init(const uint16_t * buckets,
     const uint32_t * examples,
     const int32_t * labels,
     const float * divisors,
     uint32_t size)
{
    data.reset();
    bucket_bits = example_bits = label_bits = divisor_bits = total_bits
        = 0;
    this->size = size;
    this->num_words = 0;
    
    if (size == 0) return;
    
    uint16_t highest_bucket = *std::max_element(buckets, buckets + size);
    
    uint32_t highest_example = 0;
    if (examples)
        highest_example = *std::max_element(examples, examples + size);
    
    uint32_t highest_label = 0;
    if (labels)
        highest_label = *std::max_element(labels, labels + size);
    
    bucket_bits = highest_bit(highest_bucket) + 1;
    example_bits = highest_bit(highest_example) + 1;
    label_bits  = highest_bit(highest_label) + 1;
    divisor_bits = (divisors ? 32 : 0);
    
    total_bits = bucket_bits + example_bits + label_bits + divisor_bits;
        
    /* Get the memory size in 128 bit words, then multiple by 2 to get
       the number of 64 bit words, then add 2 in order to avoid
       requiring any special logic to not read off the end */
    num_words = rudiv(total_bits * size, 128) * 2 + 2;
    data.reset(new uint64_t[num_words]);
    std::fill(data.get(), data.get() + num_words, 0);

    Bit_Writer<uint64_t> writer(data.get());
        
    /* Now go through and construct the index */
    for (unsigned i = 0;  i < size;  ++i) {
        writer.write(buckets[i], bucket_bits);
        if (example_bits) writer.write(examples[i], example_bits);
        writer.write(labels[i], label_bits);
        if (divisor_bits)
            writer.write(divisors[i], divisor_bits);
    }

    size_t bytes_before
        = 2 * size
        + 4 * size
        + (examples != 0) * 4 * size
        + (divisors != 0) * 4 * size;
        
    size_t bytes_after = num_words * 8;

    cerr << bytes_before / 1024.0 << "kb before, "
         << bytes_after / 1024.0 << "kb after, "
         << 1.0 * bytes_after / bytes_before << " compression"
         << endl;
    cerr << "total_bits = " << total_bits << endl;
}

} // namespace ML
