/* buckets.h                                                       -*- C++ -*-
   Jeremy Barnes, 12 September 2011
   Copyright (c) 2011 Jeremy Barnes and Datacratic.  All rights reserved.

   Bucketing algorithms, extracted.
*/

#ifndef __jml__buckets_h__
#define __jml__buckets_h__

#include "jml/stats/sparse_distribution.h"
#include "jml/utils/sorted_vector.h"
#include <stdint.h>

namespace ML {

inline bool bit_equal(float f1, float f2)
{
    return (f1 < f1 && f2 < f2)
        || f1 == f2;
}

/** The type of the frequency distribution. */
typedef sparse_distribution<float, float,
                            sorted_vector<float, float> > BucketFreqs;

/** Structure describing the set of buckets for one value. */
struct Bucket_Info {
    bool initialized;
    std::vector<uint16_t> buckets; ///< Bucket numbers, sorted by example
    std::vector<float> splits;     ///< Split points between the buckets
};
    
/** Create a distribution of bucket split points where we have few enough
    values that each element can go in its own bucket.  This usually occurs
    with categorical and boolean features.
*/
void bucket_dist_full(std::vector<float> & result,
                      const BucketFreqs & freqs);

/** Create a distribution of bucket split points where we don't have
    enough values, and thus we need to put elements together in the
    buckets.  This primarily occurs with real valued features.
*/
void bucket_dist_reduced(std::vector<float> & result,
                         const BucketFreqs & freqs,
                         size_t num_buckets);

/** Create a distribution of buckets for the given set of values, targeting
    the given number of buckets.
*/
void bucket_dist(std::vector<float> & result,
                 const BucketFreqs & freqs,
                 size_t num_buckets);

/** Purely functional version of the same. */
inline std::vector<float>
bucket_dist(const BucketFreqs & freqs, size_t num_buckets)
{
    std::vector<float> result;
    bucket_dist(result, freqs, num_buckets);
    return result;
}

void get_freqs(BucketFreqs & result, std::vector<float> values);

/** Create a set of buckets for the given set of values. */
Bucket_Info create_buckets(const std::vector<float> & values,
                           size_t num_buckets);


} // namespace ML

#endif /* __jml__buckets_h__ */


