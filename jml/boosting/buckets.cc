/* buckets.cc
   Jeremy Barnes, 12 September 2011
   Copyright (c) 2011 Jeremy Barnes.  All rights reserved.

   Refactored out buckets code.
*/

#include "buckets.h"
#include <iostream>
#include <boost/utility.hpp>
#include "jml/arch/format.h"
#include "jml/utils/floating_point.h"
#include "jml/utils/pair_utils.h"
#include "jml/arch/exception.h"

using namespace std;


namespace ML {

void
bucket_dist_full(vector<float> & result, const BucketFreqs & freqs)
{
    bool debug = false;
    
    result.clear();
    result.reserve(std::max<unsigned>(freqs.size() - 1, 0));
    
    /* We want the split points, which are half way in between the
       current value and the next one. */
    for (int i = 0;  i < (int)freqs.size() - 1;  ++i)
        result.push_back(((freqs.begin() + i)->first
                          + (freqs.begin() + i + 1)->first) * 0.5);

    if (debug) {
        for (BucketFreqs::const_iterator it = freqs.begin();  it != freqs.end();
             ++it)
            cerr << "freqs[" << it->first << "] = " << it->second << endl;

        for (unsigned i = 0;  i < result.size();  ++i)
            cerr << "  bucket " << i << " has value " << result[i]
                 << ", \t" << (freqs.begin() + i)->second << " values" << endl;
    }
}

void 
bucket_dist_reduced(vector<float> & result,
                    const BucketFreqs & freqs,
                    size_t num_buckets)
{
    bool debug = false;
    //debug = (feature.type() == 22);

    /* NOTE: this algorithm tends to overcluster well-distributed data.
       It should probably be re-done to produce a number of buckets
       closer to the required number.
    */

    /* Find good split points. */
    double total = freqs.total();

    vector<float> bucket_sizes;
    result.clear();
    result.reserve(num_buckets);

    double num = 0;
    float per_bucket = total / num_buckets;

    /* Go through and look for the really big buckets.  We take these
       out of the bucket size calculation as otherwise we end up with
       too few buckets.  Note that this calculation is not exact
       (since the bucket size becomes smaller after), but it should
       fix up the most problematic cases.
    */
    int num_big_buckets = 0;
    double num_in_big_buckets = 0.0;
    for (BucketFreqs::const_iterator it = freqs.begin();
         it != freqs.end();  ++it) {
        float n = it->second;
        if (n >= per_bucket) {
            num_in_big_buckets += n;
            num_big_buckets += 1;
        }
    }

    if (num_buckets != num_big_buckets)
        per_bucket = (total - num_in_big_buckets) /
            (num_buckets - num_big_buckets);
    
    if (debug) {
        cerr << "  per_bucket before = "
             << total / num_buckets << endl;
        cerr << "  per_bucket = " << per_bucket << endl;
        cerr << "  num_buckets = " << num_buckets << endl;
        cerr << "  num_big_buckets = " << num_big_buckets << endl;
        cerr << "  num_in_big_buckets = " << num_in_big_buckets << endl;
    }

    //cerr << "freqs.size() = " << freqs.size() << endl;
    int i = 0;
    for (BucketFreqs::const_iterator it = freqs.begin();
         it != freqs.end();  ++it, ++i) {
        float n = it->second;
        float v = it->first;
        num += n;

        if (debug && false)
            cerr << "i = " << i << " n = " << n << " v = " << v
                 << " num = " << num << endl;

        /* We split if we have enough in our bucket, or if the next one
           would have enough by itself. */
        if (n >= per_bucket && num > n) {
            /* Extra split, since this one had enough by itself. */
            float val = (v + boost::prior(it)->first) / 2.0;
            result.push_back(val);
            bucket_sizes.push_back(num - n);
            if (debug) 
                cerr << "i = " << i << ": split [1] at "
                     << format("val: %16.9f 0x%08x ", val,
                               reinterpret_as_int(val))
                     << format("v: %16.9f 0x%08x ", v,
                               reinterpret_as_int(v))
                     << format("prior: %16.9f 0x%08x ", boost::prior(it)->first,
                               reinterpret_as_int(boost::prior(it)->first))
                     << " with " << num - n
                     << " examples." << endl;
            num -= n;
        }
        if (i < (int)freqs.size() - 1
            && (num >= per_bucket || boost::next(it)->second >= per_bucket)) {
            float val = (v + boost::next(it)->first) * 0.5f;

            // If the two values are one ulp apart, then we need to make sure
            // that it gets rounded up otherwise we will have two buckets
            // with the same split value
            if (val == v)
                val = boost::next(it)->first;

            result.push_back(val);
            bucket_sizes.push_back(num);
            if (debug)
                cerr << "i = " << i << ": split [2] at "
                     << format("val: %16.9f 0x%08x ", val,
                               reinterpret_as_int(val))
                     << format("v: %16.9f 0x%08x ", v,
                               reinterpret_as_int(v))
                     << format("next: %16.9f 0x%08x ", boost::next(it)->first,
                               reinterpret_as_int(boost::next(it)->first))
                     << " with " << num
                     << " examples." << endl;
            num = 0;
        }
    }

    for (unsigned i = 1;  i < result.size();  ++i)
        if (result[i] == result[i - 1])
            throw Exception("two buckets with the same split point");
    
    if (debug) {
        for (unsigned i = 0;  i < result.size();  ++i)
            cerr << format("  bucket %5d has value %16.9f 0x%08x, %8.1f values",
                           i, result[i], reinterpret_as_int(result[i]),
                           bucket_sizes[i]) << endl;
    }
}

#if 0
const Dataset_Index::Index_Entry::Bucket_Info &
Dataset_Index::Index_Entry::
create_buckets(size_t num_buckets)
{
    check_used();
    Guard guard(lock);

    //cerr << "create_buckets(" << num_buckets << ")" << endl;

}
#endif

void get_freqs(BucketFreqs & result, std::vector<float> values)
{
    std::sort(values.begin(), values.end(), ML::safe_less<float>());

    vector<pair<float, float> > freqs2;
    freqs2.reserve(values.size());
    
    float last = -INFINITY;
    int count = 0;
            
    for (unsigned i = 0;  i < values.size();  ++i) {
        if (isnan(values[i]))
            throw Exception("NaN in values");
            
        if (i == 0) {
            last = values[i];
            count += 1;
        }
        else if (bit_equal(last, values[i]))
            count += 1;
        else {
            freqs2.push_back(make_pair(last, count));
            count = 1;
            last = values[i];
        }
    }
    freqs2.push_back(make_pair(last, count));

    std::sort(freqs2.begin(), freqs2.end());
    
    result = BucketFreqs(freqs2.begin(), freqs2.end());
} // namespace ML

void bucket_dist(std::vector<float> & result,
                 const BucketFreqs & freqs,
                 size_t num_buckets)
{
    if (num_buckets >= freqs.size())
        bucket_dist_full(result, freqs);
    else
        bucket_dist_reduced(result, freqs, num_buckets);
}

Bucket_Info create_buckets(const std::vector<float> & values,
                           size_t num_buckets)
{
    BucketFreqs freqs;
    get_freqs(freqs, values);

    Bucket_Info result;
    bucket_dist(result.splits, freqs, num_buckets);

    vector<int> bucket_count(result.splits.size() + 1);
        
    for (unsigned i = 0;  i < values.size();  ++i) {
        float value = values[i];
        int bucket = std::upper_bound(result.splits.begin(),
                                      result.splits.end(), value)
            - result.splits.begin();

        bucket_count[bucket] += 1;

        result.buckets.push_back(bucket);
    }

    return result;
}

} // namespace ML

