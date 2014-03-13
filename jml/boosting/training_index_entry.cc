/* training_index_entry.cc
   Jeremy Barnes, 23 February 2005
   Copyright (c) 2005 Jeremy Barnes.  All rights reserved.
   $Source$

   Implementation of index entries for the Training_Data.
*/

#include "training_index_entry.h"
#include "jml/utils/sgi_numeric.h"
#include "jml/utils/vector_utils.h"
#include "feature_info.h"
#include "feature_space.h"
#include "jml/utils/floating_point.h"
#include "jml/utils/pair_utils.h"
#include <boost/timer.hpp>
#include "jml/utils/exc_assert.h"

using namespace std;

namespace ML {


/*****************************************************************************/
/* DATASET_INDEX::INDEX_ENTRY                                                */
/*****************************************************************************/

Dataset_Index::Index_Entry::
Index_Entry()
    : used(false), initialized(false),
      example_count(0), seen(0), found_in(0), missing_from(0), found_twice(0),
      zeros(0), ones(0),
      non_integral(0), max_value(-INFINITY), min_value(INFINITY),
      last_example((unsigned)-1), in_this_ex(0),
      has_examples_sorted(false), has_values_sorted(false),
      has_counts(false), has_counts_sorted(false),
      has_divisors(false), has_divisors_sorted(false),
      has_labels(false), has_labels_sorted(false),
      has_freqs(false), has_category_freqs(false)
{
}

string Dataset_Index::Index_Entry::
print_info() const
{
    check_used();
    return format("ex_cnt: %d  seen: %d  found: %d  missing: %d  >1: %d  0: %d "
                  "1: %d  non-int: %d  min: %f  max: %f dense: %d exactly_one: %d",
                  example_count, seen, found_in, missing_from, found_twice,
                  zeros, ones, non_integral, min_value, max_value,
                  dense(), exactly_one());
}

void
Dataset_Index::Index_Entry::
check_used() const
{
    if (!used)
        throw Exception("attempt to access unused feature %s",
                        feature_space->print(feature).c_str());
}
   
void Dataset_Index::Index_Entry::
insert(float value, unsigned example, unsigned example_count, bool sparse,
       const Feature_Set & fs)
{
    check_used();
    if (isnanf(value)) return;

    if (!sparse && values.empty())
        values.reserve(example_count);
    
#if 0
    if (feature_space->print(feature) == "documentName")
        cerr << "inserting " << feature_space->print(feature) 
             << " val " << value << " ("
             << feature_space->print(feature, value)
             << ") example " << example << endl;
#endif
    
    bool was_exactly_one = (values.size() && examples.empty());

    seen += 1;
    if (last_example != example) {
        found_in += 1;
        in_this_ex = 1;
    }
    else in_this_ex += 1;

    if (in_this_ex == 2) {
#if 0 // debugging sort problem
        cerr << "feature " + feature_space->print(feature) + " was found twice"
             << " on example " << example << " last_example " << last_example
             << " of " << example_count << " with value "
             << value << endl;
        cerr << "fs = " << feature_space->print(fs) << endl;

        for (Feature_Set::const_iterator it = fs.begin(), end = fs.end();  it != end;
             ++it) {
            cerr << it.feature() << " " << feature_space->print(it.feature()) << " -> "
                 << feature_space->print(it.feature(), it.value()) << endl;
        }
        throw Exception("feature " + feature_space->print(feature) + " was found twice");
#endif // debugging sort problem
        found_twice += 1;
    }
    if (value == 0.0) zeros += 1;
    if (value == 1.0) ones += 1;
    if (round(value) != value) non_integral += 1;
    if (value < min_value) min_value = value;
    if (value > max_value) max_value = value;

    /* Update the examples index by adding this to the back of it.  As an
       optimisation, we don't create the index when it just counts up.
    */
    bool is_exactly_one = (found_in - 1 == example && seen == found_in);

    if (examples.empty() && !is_exactly_one && was_exactly_one) {
        /* Up until last_example, it was exactly_one.  Recreate this. */
        examples.resize(last_example + 1);
        std::iota(examples.begin(), examples.end(), 0);
    }

    if (!is_exactly_one) {
        if (examples.empty() && !sparse)
            examples.reserve(example_count);
        examples.push_back(example);
    }

    values.push_back(value);
    
    if (examples.size() && examples.size() != values.size()) {
        cerr << "value = " << value << " example = " << example
             << " was_exactly_one = " << was_exactly_one
             << " is_exactly_one = " << is_exactly_one
             << " last_example = " << last_example
             << endl;
        throw Exception
            (format("Index_Entry::insert(): internal error: "
                    "examples and values index don't have the same "
                    "size (%zd vs %zd); info is %s",
                    examples.size(), values.size(),
                    print_info().c_str()));
    }

    last_example = example;
}

void Dataset_Index::Index_Entry::
finalize(unsigned example_count, const Feature & feature,
         std::shared_ptr<const Feature_Space> feature_space)
{
    check_used();
    //boost::timer t;

    this->feature = feature;
    this->feature_space = feature_space;
    this->example_count = example_count;
    missing_from = example_count - found_in;
    if (values.capacity() != values.size()) values = vector<float>(values);

    if (dense() && exactly_one()) {
        ExcAssert(examples.empty());
    }
    else {
        /* If the feature occurred densely at the start but then stopped, we
           never had a chance to fill in the missing values. */
        if (examples.empty() && values.size()) {
            examples.resize(values.size());
            std::iota(examples.begin(), examples.end(), 0);
        }

        if (examples.size() != values.size())
            throw Exception
                (format("Index_Entry::finalize(): internal error: "
                        "examples and values index don't have the same "
                        "size (%zd vs %zd); info is %s",
                        examples.size(), values.size(),
                        print_info().c_str()));

#if 0 // invalid; feature that never occurs will throw
        if (examples.empty()) {
            cerr << "finalizing feature " << feature << " ("
                 << feature_space->print(feature) << ") with info "
                 << print_info() << ": examples is empty" << endl;
            throw Exception("examples is empty");
        }
#endif // invalid

        if (examples.capacity() != examples.size())
            examples = vector<unsigned>(examples);
    }

#if 0
    if (found_twice == 0) return;

    cerr << "finalized feature " << feature_space->print(feature) << endl;
    cerr << "  example_count = " << example_count << endl;
    cerr << "  seen          = " << seen << endl;
    cerr << "  found_in      = " << found_in << endl;
    cerr << "  missing_from  = " << missing_from << endl;
    cerr << "  found_twice   = " << found_twice << endl;
    cerr << "  zeros         = " << zeros << endl;
    cerr << "  ones          = " << ones << endl;
    cerr << "  non_integral  = " << non_integral << endl;
    cerr << "  min_value     = " << min_value << endl;
    cerr << "  max_value     = " << max_value << endl;
    cerr << "  last_example  = " << last_example << endl;
    cerr << "  in_this_ex    = " << in_this_ex << endl;
    cerr << "  info          = " << print_info() << endl;
    cerr << "  dense         = " << dense() << endl;
    cerr << "  only_one      = " << only_one() << endl;
    cerr << "  exactly_one   = " << exactly_one() << endl;
    cerr << "  density       = " << density() << endl;
    cerr << endl;
#endif

    //cerr << "finalizing feature " << feature_space->print(feature)
    //     << " took " << t.elapsed() << "s" << endl;
}

const vector<float> &
Dataset_Index::Index_Entry::
get_values(Sort_By sort_by)
{
    check_used();
    if (sort_by == BY_EXAMPLE) return values;
    else if (sort_by == BY_VALUE) {
        if (has_values_sorted) return values_sorted;
        vector<float> new_values_sorted = values;
        std::sort(new_values_sorted.begin(), new_values_sorted.end());
        //safe_less<float>());
        
        Guard guard(lock);
        if (has_values_sorted) return values_sorted;
        values_sorted.swap(new_values_sorted);
        has_values_sorted = true;
        return values_sorted;
    }
    else throw Exception("invalid sort_by");
}

const vector<unsigned> &
Dataset_Index::Index_Entry::
get_examples(Sort_By sort_by)
{
    check_used();
    if (sort_by == BY_EXAMPLE) return examples;

    else if (sort_by == BY_VALUE) {
        if (has_examples_sorted) return examples_sorted;
        
        vector<unsigned> examples2;
        if (examples.empty()) {
            /* Construct a vector counting from 0 */
            examples2 = vector<unsigned>(values.size());
            std::iota(examples2.begin(), examples2.end(), 0);
        }
        const vector<unsigned> & examples3
            = (examples.size() ? examples : examples2);

        vector<pair<float, unsigned> > pairs
            (pair_merger(values.begin(), examples3.begin()),
             pair_merger(values.end(), examples3.end()));
        sort_on_first_ascending(pairs);
        
        if (values_sorted.size()) {
            /* Should be the same whether pre-calculated or not. */
            ExcAssert(std::equal(values_sorted.begin(),
                              values_sorted.end(),
                              first_extractor(pairs.begin())));
        }
        else {
            values_sorted = vector<float>(first_extractor(pairs.begin()),
                                          first_extractor(pairs.end()));
        }

        vector<unsigned> new_examples_sorted(second_extractor(pairs.begin()),
                                             second_extractor(pairs.end()));
        Guard guard(lock);
        if (has_examples_sorted) return examples_sorted;
        examples_sorted.swap(new_examples_sorted);
        has_examples_sorted = true;
        return examples_sorted;
    }
    else throw Exception("invalid sort_by");
}

const vector<unsigned> &
Dataset_Index::Index_Entry::
get_counts(Sort_By sort_by)
{
    check_used();
    if (only_one()) return counts;  // one per example; no problem

    bool debug = false;
    //debug = (feature_space->print(feature) == "lemma-try");

    if (debug) {
        cerr << "get_counts(" << sort_by << ")" << endl;
        cerr << "  get_examples = " << get_examples(sort_by) << endl;
    }

    if (sort_by == BY_EXAMPLE) {
        if (has_counts) return counts;

        if (debug) {
            cerr << "constructing from examples" << endl;
            cerr << "examples = " << examples << endl;
        }

        vector<unsigned> new_counts;
        new_counts.reserve(examples.size());
        
        unsigned last = 0;
        unsigned count = 0;
        
        for (unsigned i = 0;  i < examples.size();  ++i) {
            if (i == 0 || examples[i] == last) {
                count += 1;
                last = examples[i];
            }
            else {
                for (unsigned j = 0;  j < count;  ++j)
                    new_counts.push_back(count);
                last = examples[i];
                count = 1;
            }
        }
        
        for (unsigned j = 0;  j < count;  ++j)
            new_counts.push_back(count);
        
        ExcAssert(new_counts.size() == examples.size());
        
        if (debug)
            cerr << "counts = " << counts << endl;

        Guard guard(lock);
        if (has_counts) return counts;
        counts.swap(new_counts);
        has_counts = true;
        return counts;
    }
    else if (sort_by == BY_VALUE) {
        if (has_counts_sorted) return counts_sorted;

        /* Use three other arrays to construct ours. */
        if (debug) cerr << "constructing from 3 arrays" << endl;
        hash_map<unsigned, unsigned> examp_counts;
        const vector<unsigned> & ex_counts = get_counts(BY_EXAMPLE);
        const vector<unsigned> & ex_examples = get_examples(BY_EXAMPLE);
        
        if (ex_examples.empty())
            for (unsigned i = 0;  i < ex_counts.size();  ++i)
                examp_counts[i] = ex_counts[i];
        else
            for (unsigned i = 0;  i < examples.size();  ++i)
                examp_counts[ex_examples[i]] = ex_counts[i];
        
        const vector<unsigned> & val_examples = get_examples(BY_VALUE);
        
        vector<unsigned> new_counts_sorted;
        new_counts_sorted.reserve(val_examples.size());
        for (unsigned i = 0;  i < val_examples.size();  ++i)
            new_counts_sorted.push_back(examp_counts[val_examples[i]]);

        Guard guard(lock);
        if (has_counts_sorted) return counts_sorted;
        counts_sorted.swap(new_counts_sorted);
        has_counts_sorted = true;
        return counts_sorted;

#if 0 // old code; optimization
        vector<int> example_counts;
        map<int, int> sparse_example_counts;

        if (size() * 10 < example_count) {

            // Calculate the example counts (sparse version)
            sparse_example_counts.clear();
            for (unsigned i = 0;  i < size();  ++i)
                sparse_example_counts[(*this)[i].example()] += 1;
        
            for (unsigned i = 0;  i < size();  ++i) {
                int count = sparse_example_counts[(*this)[i].example()];
                if (count == 0)
                    throw Exception("Training_Data::finish(): count = 0");
                (*this)[i].example_counts() = count;
                if (count != 1) all_one = false;
                if (count == 0) missing = true;
            }
        }
        else {
            example_counts.resize(example_count);

            // Calculate the example counts (dense version)
            std::fill(example_counts.begin(), example_counts.end(), 0);

            for (unsigned i = 0;  i < size();  ++i) {
                example_counts[(*this)[i].example()] += 1;
            }

            for (unsigned i = 0;  i < size();  ++i) {
                int count = example_counts[(*this)[i].example()];
                if (count == 0)
                    throw Exception("Training_Data::finish(): count = 0");
                (*this)[i].example_counts() = count;
                if (count != 1) all_one = false;
                if (count == 0) missing = true;
            }
        }
#endif // old code; optimization

    }
    else throw Exception("invalid sort_by");
}

const vector<float> &
Dataset_Index::Index_Entry::
get_divisors(Sort_By sort_by)
{
    check_used();
    /* We calculate these directly from the counts. */
    if (sort_by == BY_EXAMPLE) {
        if (has_divisors) return divisors;

        const vector<unsigned> & counts = get_counts(sort_by);

        if (!counts.empty()) {
            std::vector<float> new_divisors(counts.size());
            for (unsigned i = 0;  i < counts.size();  ++i)
                new_divisors[i] = 1.0 / counts[i];

            Guard guard(lock);
            if (has_divisors) return divisors;
            divisors.swap(new_divisors);
            has_divisors = true;
        }
        has_divisors = true;
        return divisors;
    }
    else if (sort_by == BY_VALUE) {
        if (has_divisors_sorted) return divisors_sorted;

        const vector<unsigned> & counts = get_counts(sort_by);

        if (!counts.empty()) {
            std::vector<float> new_divisors(counts.size());
            for (unsigned i = 0;  i < counts.size();  ++i)
                new_divisors[i] = 1.0 / counts[i];

            Guard guard(lock);
            if (has_divisors_sorted) return divisors_sorted;
            divisors_sorted.swap(new_divisors);
            has_divisors_sorted = true;
        }

        has_divisors_sorted = true;
        return divisors_sorted;
    }
    else throw Exception("invalid sort_by");
}

namespace {

struct Safe_Less_Pair {

    bool operator () (const std::pair<float, float> & p1,
                      const std::pair<float, float> & p2) const
    {
        return sl(p1.first, p2.first)
            || (bit_equal(p1.first, p2.first) && sl(p1.second, p2.second));
    }

    safe_less<float> sl;
};

} // file scope

const Dataset_Index::Freqs &
Dataset_Index::Index_Entry::
get_freqs()
{
    check_used();
    if (has_freqs) return freqs;

    vector<pair<float, float> > freqs2;
    freqs2.reserve(seen);

    if (only_one()) {
        /* Accumulate them; example_count is always one since only_one is
           true. */
        const vector<float> & vals = get_values(BY_VALUE);

        float last = -INFINITY;
        int count = 0;
            
        for (unsigned i = 0;  i < vals.size();  ++i) {
            if (isnan(vals[i]))
                throw Exception("NaN in vals");
            
            if (i == 0) {
                last = vals[i];
                count += 1;
            }
            else if (bit_equal(last, vals[i]))
                count += 1;
            else {
                freqs2.push_back(make_pair(last, count));
                count = 1;
                last = vals[i];
            }
        }
        freqs2.push_back(make_pair(last, count));
    }
    else {
        /* Accumulate them; example count is not always one; we need to
           get hold of the counts. */
        const vector<unsigned> & counts = get_counts(BY_VALUE);
        const vector<float> & vals = get_values(BY_VALUE);
            
        float last = -INFINITY;
        double count = 0.0;
            
        for (unsigned i = 0;  i < vals.size();  ++i) {
            if (isnan(vals[i]))
                throw Exception("NaN in vals");

            if (i == 0) {
                count += 1.0 / counts[i];
                last = vals[i];
            }
            else if (bit_equal(last, vals[i]))
                count += 1.0 / counts[i];
            else {
                freqs2.push_back(make_pair(last, count));
                count = 1.0 / counts[i];
                last = vals[i];
            }
        }
        freqs2.push_back(make_pair(last, count));
    }

#if 0
    bool changed;
    do {
        changed = false;
        for (unsigned i = 1;  i < freqs2.size();  ++i) {
            if (Safe_Less_Pair()(freqs2[i], freqs2[i-1])) {
                std::swap(freqs2[i-1], freqs2[i]);
                changed = true;
            }
        }
    } while (changed);
    //std::sort(freqs2.begin(), freqs2.end(), Safe_Less_Pair());
#endif
    std::sort(freqs2.begin(), freqs2.end());
    
    Guard guard(lock);
    if (has_freqs) return freqs;
    freqs = Freqs(freqs2.begin(), freqs2.end());
    has_freqs = true;
    return freqs;
}

const Dataset_Index::Category_Freqs &
Dataset_Index::Index_Entry::
get_category_freqs(size_t num_categories)
{
    check_used();
    if (has_category_freqs) {
        if (num_categories != category_freqs.size())
            throw Exception("get_category_freqs(): feature has changed number of "
                            "categories");
        return category_freqs;
    }

    const Freqs & freqs = get_freqs();

    Category_Freqs new_category_freqs(num_categories);

    for (Freqs::const_iterator it = freqs.begin();  it != freqs.end();  ++it) {
        if (round(it->first) != it->first
            || it->first < 0 || it->first >= num_categories)
            throw Exception("get_category_freqs: value not categorical");
        new_category_freqs[(int)it->first] = it->second;
    }

    Guard guard(lock);
    if (has_category_freqs) {
        if (num_categories != category_freqs.size())
            throw Exception("get_category_freqs(): feature has changed number of "
                            "categories");
        return category_freqs;
    }
    category_freqs.swap(new_category_freqs);
    has_category_freqs = true;
    return category_freqs;
}

const vector<Label> &
Dataset_Index::Index_Entry::
get_labels()
{
    check_used();
    if (!feature_space && example_count != 0)
        throw Exception("get_labels(): no feature space");
    
    //cerr << "get_labels(" << feature << "): labels.size() = "
    //     << labels.size() << endl;
 
    if (has_labels || example_count == 0) return labels;
    /* No labels?  create them */

    if (!exactly_one()) {
        cerr << print_info() << endl;
        throw Exception("get_labels(): currently label must have exactly "
                        "one feature per example for feature "
                        + feature_space->print(feature) + ": "
                        + print_info());
    }
    
    Feature_Info info = feature_space->info(feature);
        
    ExcAssert(values.size() == example_count);

    //cerr << "values = " << values << endl;

    vector<Label> new_labels;
    new_labels.reserve(example_count);

    switch (info.type()) {

    case REAL:
        for (unsigned i = 0;  i < values.size();  ++i)
            new_labels.push_back(Label(values[i]));
        break;
        
    default:
        for (unsigned i = 0;  i < values.size();  ++i)
            new_labels.push_back(Label((int)values[i]));
    }
    
    Guard guard(lock);
    if (has_labels) return labels;
    new_labels.swap(labels);
    has_labels = true;
    return labels;
}

const vector<Label> &
Dataset_Index::Index_Entry::
get_mapped_labels(const vector<Label> & labels, const Feature & target,
                  Sort_By sort_by)
{
    check_used();
    //cerr << "examples.size() = " << examples.size() << endl;
    //cerr << "labels.size() = " << labels.size() << endl;
    //cerr << "example_count = " << example_count << endl;
    //cerr << endl;

    if (labels.size() != example_count) {
        cerr << "target: " << target << endl;
        cerr << "sort_by: " << sort_by << endl;

        throw Exception
            (format("get_mapped_labels(): size(%zd) != example_count(%d)",
                    labels.size(), example_count));
    }

    if (exactly_one() && sort_by == BY_EXAMPLE)
        return labels;  // don't need to update them...

    Guard guard(lock);
    vector<Label> & result = (sort_by == BY_VALUE ? mapped_labels[target]
                              : mapped_labels_sorted[target]);
    
    if (result.size()) return result;

    /* Get the examples to map. */
    const vector<unsigned> & examples = get_examples(sort_by);
    
    ExcAssert(!examples.empty());

    result = vector<Label>(examples.size());

    for (unsigned x = 0;  x < examples.size();  ++x)
        result[x] = labels[examples[x]];
        
    return result;
}

const Bucket_Info &
Dataset_Index::Index_Entry::
create_buckets(size_t num_buckets)
{
    check_used();
    Guard guard(lock);

    //cerr << "create_buckets(" << num_buckets << ")" << endl;

    const Freqs & freqs = get_freqs();

    Bucket_Info & result = bucket_info[num_buckets];

    if (num_buckets >= freqs.size())
        bucket_dist_full(result.splits, freqs);
    else
        bucket_dist_reduced(result.splits, freqs, num_buckets);

    //cerr << "  produced " << result.splits.size() + 1
    //     << " buckets" << endl;

    result.buckets.clear();

    const vector<float> & values = get_values(BY_EXAMPLE);

    vector<int> bucket_count(result.splits.size() + 1);
        
    for (unsigned i = 0;  i < values.size();  ++i) {
        float value = values[i];
        int bucket = std::upper_bound(result.splits.begin(),
                                      result.splits.end(), value)
            - result.splits.begin();

        //cerr << "i = " << i << "  value = " << value << "  bucket = "
        //     << bucket << endl;

        bucket_count[bucket] += 1;

        result.buckets.push_back(bucket);
    }

    //if (feature.type() == 22) {
    //    cerr << "bucket_count = " << bucket_count << endl;
    //}

    return result;
}

const Bucket_Info &
Dataset_Index::Index_Entry::
buckets(size_t num_buckets)
{
    check_used();
    /* If there are more buckets than distinct values, then we use the
       number of distinct values as the number of buckets.
    */

    //cerr << "buckets(" << num_buckets << ")" << endl;

    const Freqs & freqs = get_freqs();
    if (freqs.size() < num_buckets)
        num_buckets = freqs.size();
    
    Guard guard(lock);
    if (bucket_info.count(num_buckets))
        return bucket_info[num_buckets];
    
    return create_buckets(num_buckets);
}

#if 0 // TODO: potential bug here; this throws sometimes
if (bucket_splits[bucket] != value) {
    cerr << "buckets.size() = " << buckets.size()
         << "  num_buckets = " << num_buckets
         << "  value = " << value << "  bucket = "
         << bucket << "  bucket_vals[bucket] = "
         << bucket_vals[bucket] << endl;
    cerr << "sorted.size() = " << sorted.size() << endl;
    cerr << "bucket_vals: " << endl;
    for (unsigned i = 0;  i < bucket_vals.size();  ++i) {
        cerr << "bucket " << i << " val "
             << bucket_vals[i] << " sorted " << sorted[i].first
             << endl;
    }
    throw Exception("boosting internal error: bad bucketizing");
}
#endif // potential bug


} // namespace ML
