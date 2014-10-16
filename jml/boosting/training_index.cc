/* training_index.cc
   Jeremy Barnes, 18 February 2005
   Copyright (c) 2005 Jeremy Barnes.  All rights reserved.
   $Source$

   Implementation of training data indexes.
*/

#include "training_index.h"
#include "training_index_entry.h"
#include "feature_map.h"
#include "feature_space.h"
#include "training_data.h"
#include "jml/utils/sgi_numeric.h"
#include "jml/utils/vector_utils.h"
#include <boost/timer.hpp>
#include "jml/utils/string_functions.h"
#include "jml/arch/demangle.h"
#include <set>


using namespace std;


namespace ML {


/*****************************************************************************/
/* DATASET_INDEX                                                             */
/*****************************************************************************/


struct Dataset_Index::Itl {
    typedef Feature_Map<Index_Entry> index_type;
    index_type index;
    std::shared_ptr<const Feature_Space> feature_space;
    std::vector<Feature> all_features;
};

void
Dataset_Index::
init(const Training_Data & data,
     const std::vector<Feature> & features_)
{
    //boost::timer t;

    itl.reset(new Itl());
    itl->feature_space = data.feature_space();
    
    size_t nx = data.example_count();
    bool sparse = (data.feature_space()->type() == SPARSE);

    //cerr << "sparse = " << sparse << endl;
    
    /* Iterate through the data and index it one by one. */

    vector<Index_Entry *> entries;
    vector<Feature> features;

    std::set<Feature> keep_features(features_.begin(), features_.end());
    
    for (unsigned x = 0;  x < nx;  ++x) {
        //cerr << "x = " << x << " of " << nx << endl;
        const Feature_Set & fs = data[x];
        if (x == 0) {
            for (Feature_Set::const_iterator it = fs.begin();
                 it != fs.end();  ++it) {
                const Feature & feat = it.feature();
                
                itl->index[feat].used = keep_features.empty() || keep_features.count(feat);
                itl->index[feat].feature = feat;
                itl->index[feat].feature_space = itl->feature_space;
                itl->index[feat].initialized = true;
                features.push_back(feat);
                entries.push_back(&itl->index[feat]);
            }
        }
        
        //std::set<Feature> doneFeatures;

        int i = 0;
        for (Feature_Set::const_iterator it = fs.begin();
             it != fs.end();  ++it, ++i) {
            const Feature & feat = it.feature();
            float val = it.value();

#if 0 // debugging sort() problem               
            if (doneFeatures.count(feat)) {
                //sleep(1);

                std::set<Feature> doneFeatures2;
                for (Feature_Set::const_iterator it = fs.begin();
                     it != fs.end();  ++it) {
                    const Feature & feat = it.feature();
                    if (doneFeatures2.count(feat))
                        throw Exception("doubled up twice on feature " + itl->feature_space->print(feat)
                                        + " on " + type_name(fs));  // debug
                    doneFeatures2.insert(feat);
                }

                throw Exception("doubled up temporarily feature " + itl->feature_space->print(feat)
                                + " on " + type_name(fs));  // debug
            }
            doneFeatures.insert(feat);
#endif

            /* Save a map lookup for the common case of always the same
               features or always the same ones at the start. */
            if (i < features.size() && features[i] == feat) {
                if (entries[i]->used)
                    entries[i]->insert(val, x, nx, sparse, fs);
            }
            else {
                Index_Entry & entry = itl->index[feat];
                if (!entry.initialized) {
                    entry.initialized = true;
                    entry.used = (keep_features.empty() || keep_features.count(feat));
                    entry.feature = feat;
                    entry.feature_space = itl->feature_space;
                }
                if (entry.used)
                    entry.insert(val, x, nx, sparse, fs);
            }
        }
    }
    
    itl->all_features.clear();
    itl->all_features.reserve(itl->index.size());

    /* Finalize the data structures. */
    for (Itl::index_type::iterator it = itl->index.begin();
         it != itl->index.end();  ++it) {
        Feature feature = it.key();
        itl->all_features.push_back(feature);
        if (!it->used) continue;

        //cerr << "finalizing feature " << itl->feature_space->print(feature)
        //     << endl;

        it->finalize(data.example_count(), feature, itl->feature_space);

        //cerr << "  " << it->print_info() << endl;
        //cerr << "  examples = " << it->examples.size() << endl;
        //cerr << "  values   = " << it->values.size() << endl;
    }

    std::sort(itl->all_features.begin(), itl->all_features.end());

    //cerr << "finalize features: " << t.elapsed() << "s" << endl;

    //cerr << "Dataset_Index::init(): " << format("%6.2fs", t.elapsed()) << endl;
}


void
Dataset_Index::
init(const Training_Data & data,
     const Feature & label,
     const std::vector<Feature> & features)
{
    vector<Feature> features2 = features;
    if (std::find(features.begin(), features.end(), label) == features.end())
        features2.push_back(label);
    init(data, features2);

    // Generate labels
    itl->index[label].get_labels();
}

const std::vector<Feature> & Dataset_Index::all_features() const
{
    return itl->all_features;
}

const Dataset_Index::Freqs &
Dataset_Index::freqs(const Feature & feature) const
{
    return itl->index[feature].get_freqs();
}

const Dataset_Index::Category_Freqs &
Dataset_Index::category_freqs(const Feature & feature) const
{
    size_t num_categories
        = itl->feature_space->info(feature).value_count();
    return itl->index[feature].get_category_freqs(num_categories);
}

const std::vector<Label> &
Dataset_Index::labels(const Feature & feature) const
{
    return itl->index[feature].get_labels();
}

const std::vector<float> &
Dataset_Index::values(const Feature & feature) const
{
    return itl->index[feature].get_values(BY_EXAMPLE);
}

Joint_Index
Dataset_Index::
joint(const Feature & target, const Feature & independent,
      Sort_By sort_by, unsigned contents, size_t num_buckets) const
{
    /* Normally if you have done this, you are trying to predict a label
       using itself, which is an error.
    */
    if (target == independent)
        throw Exception("Dataset_Index::joint(): distribution between "
                        "a feature and itself requested; use dist() if "
                        "this is really what you mean");
    
    bool want_buckets = (num_buckets > 0);
    bool want_labels = true;
    bool want_counts = true;
    bool want_examples = true;
    bool want_divisors = true;
    bool want_values = true;

    /* Labels are joint between two distributions. */
    const Label * labels = 0;
    const uint16_t * buckets = 0;
    const vector<float> * bucket_splits = 0;
    const unsigned * counts = 0;
    const unsigned * examples = 0;
    const float * divisors = 0;
    const float * values = 0;

    if (itl->index[independent].seen == 0) // unknown feature...
        return Joint_Index(values, buckets, labels, examples, counts, divisors,
                           0, bucket_splits);

    if (want_labels) {
        const vector<Label> & example_labels
            = itl->index[target].get_labels();
        //cerr << "example_labels = " << example_labels << endl;
        const vector<Label> & mapped_labels
            = itl->index[independent].get_mapped_labels(example_labels, target,
                                                        sort_by);
        //cerr << "mapped_labels = " << mapped_labels << endl;
        labels = &mapped_labels[0];
    }

    if (want_buckets) {
        const Bucket_Info & bucket_info
            = itl->index[independent].buckets(num_buckets);
        buckets = &bucket_info.buckets[0];
        bucket_splits = &bucket_info.splits;
    }

    if (want_counts) {
        const vector<unsigned> & counts_vector
            = itl->index[independent].get_counts(sort_by);
        if (!counts_vector.empty())
            counts = &counts_vector[0];
    }

    if (want_divisors) {
        const vector<float> & divisors_vector
            = itl->index[independent].get_divisors(sort_by);
        if (!divisors_vector.empty())
            divisors = &divisors_vector[0];
    }

    if (want_examples) {
        const vector<unsigned> & examples_vector
            = itl->index[independent].get_examples(sort_by);
        if (!examples_vector.empty())
            examples = &examples_vector[0];
    }

    if (want_values) {
        const vector<float> & values_vector
            = itl->index[independent].get_values(sort_by);
        values = &values_vector[0];
    }

    Joint_Index result(values, buckets, labels, examples, counts, divisors,
                       itl->index[independent].seen, bucket_splits);

#if 0
    if (itl->index[independent].found_twice > 0
        && itl->feature_space->print(independent) == "lemma-try") {
        cerr << "got joint distribution for feature "
             << itl->feature_space->print(independent) << endl;
        Index_Iterator it(&result, 0), end(&result, result.size());

        int i = 0;
        cerr << "  num    value m bckt labl exmp exct o div\n";
        while (it != end) {
            cerr << format("%5d %8g %1d %4d %4d %4d %4d %1d %4.2f\n",
                           i++, it->value(), it->missing(),
                           0/*it->bucket()*/,
                           it->label().operator int(),
                           it->example(), it->example_counts(),
                           it->one_example(), it->divisor());
            ++it;
        }
        cerr << endl;
    }
#endif

    return result;
}

Joint_Index
Dataset_Index::
dist(const Feature & feature, Sort_By sort_by, unsigned content,
     size_t num_buckets) const
{
    bool want_buckets = (num_buckets > 0);
    bool want_counts = true;
    bool want_divisors = true;
    bool want_examples = true;
    bool want_values = true;

    /* Labels are joint between two distributions. */
    const Label * labels = 0;
    const uint16_t * buckets = 0;
    const vector<float> * bucket_splits = 0;
    const unsigned * counts = 0;
    const float * divisors = 0;
    const unsigned * examples = 0;
    const float * values = 0;

    if (itl->index[feature].seen == 0) // unknown feature...
        return Joint_Index(values, buckets, labels, examples, counts, divisors,
                           0, bucket_splits);
    if (want_buckets) {
        const Bucket_Info & bucket_info
            = itl->index[feature].buckets(num_buckets);
        buckets = &bucket_info.buckets[0];
        bucket_splits = &bucket_info.splits;
    }

    if (want_counts) {
        const vector<unsigned> & counts_vector
            = itl->index[feature].get_counts(sort_by);
        if (!counts_vector.empty())
            counts = &counts_vector[0];
    }

    if (want_divisors) {
        const vector<float> & divisors_vector
            = itl->index[feature].get_divisors(sort_by);
        if (!divisors_vector.empty())
            divisors = &divisors_vector[0];
    }

    if (want_examples) {
        const vector<unsigned> & examples_vector
            = itl->index[feature].get_examples(sort_by);
        if (!examples_vector.empty())
            examples = &examples_vector[0];
    }

    if (want_values) {
        const vector<float> & values_vector
            = itl->index[feature].get_values(sort_by);
        values = &values_vector[0];
    }

    return Joint_Index(values, buckets, labels, examples, counts, divisors,
                       itl->index[feature].seen, bucket_splits);
}

double Dataset_Index::density(const Feature & feat) const
{
    return itl->index[feat].density();
}

bool Dataset_Index::exactly_one(const Feature & feat) const
{
    return itl->index[feat].exactly_one();
}

bool Dataset_Index::dense(const Feature & feat) const
{
    return itl->index[feat].dense();
}

bool Dataset_Index::only_one(const Feature & feat) const
{
    return itl->index[feat].only_one();
}

size_t Dataset_Index::count(const Feature & feat) const
{
    return itl->index[feat].seen;
}

std::pair<float, float> Dataset_Index::range(const Feature & feature) const
{
    return make_pair(itl->index[feature].min_value,
                     itl->index[feature].max_value);
}

bool Dataset_Index::constant(const Feature & feat) const
{
    std::pair<float, float> r = range(feat);
    return r.first == r.second;
}

bool Dataset_Index::integral(const Feature & feature) const
{
    return itl->index[feature].non_integral == 0;
}

Feature_Info Dataset_Index::
guess_info(const Feature & feat) const
{
    /* Algorithm: if there is just 1 and 0 values, we say it's boolean.
       If there is just a 1 value or missing, we say it's a presence
       feature.  If none are missing and all values are the same, we
       say it's inutile.  Otherwise, we say that it's real. */
    //return Feature_Info(PRESENCE);

    const Index_Entry & entry = itl->index[feat];
    
    string name = itl->feature_space->print(feat);
    bool debug = false;//(name == "AVG_FILLED_IN_CONF");

    if (debug) cerr << "guess_info for " << name << endl;
    if (debug) cerr << "entry: " << entry.print_info() << endl;

    Mutable_Feature_Info result;
    if (entry.seen == 0)
        result.set_type(INUTILE);
    else {
        bool dense = entry.dense();
        bool boolean = (entry.ones + entry.zeros == entry.seen);
        bool uniform_one = (entry.ones == entry.seen);
        bool one_example = (entry.exactly_one());
        bool one_value = (entry.max_value == entry.min_value);
        //bool all_integral = (entry.non_integral == 0);

        if (debug) {
            cerr << "feature " << name << ": "
                 << dense << " " << boolean << " " << uniform_one
                 << " " << one_example << " " << one_value << endl;
        }

        /* Uniformly the same value and none missing? */
        if (dense && one_example && one_value)
            result.set_type(INUTILE);
        else if (one_value && uniform_one) result.set_type(PRESENCE);
        else if (boolean) result.set_type(BOOLEAN);
        else if (itl->feature_space->info(feat).type()
                     == CATEGORICAL
                 || itl->feature_space->info(feat).type()
                     == STRING)
            result = itl->feature_space->info(feat);
#if 0
        else if (all_integral
                 && entry.min_value >= 0 && entry.max_value <= 255) {
            result.type = CATEGORICAL;
            result.categorical.reset
                (new Fixed_Categorical_Info((int)entry.max_value + 1));
            // TODO: detect categorical (?)
        }
#endif
        else result.set_type(REAL);
    }
    
    if (debug)
        cerr << "result is " << result << endl;

    if (debug) cerr << "result now " << result << endl;

    return result;
}

Feature_Info Dataset_Index::
guess_info_categorical(const Feature & feat) const
{
    const Index_Entry & entry = itl->index[feat];

    Mutable_Feature_Info result;
    if (entry.seen == 0)
        result.set_type(INUTILE);
    else {
        bool dense = entry.dense();
        bool boolean = (entry.ones + entry.zeros == entry.seen);
        bool uniform_one = (entry.ones == entry.seen);
        bool one_example = (entry.exactly_one());
        bool one_value = (entry.max_value == entry.min_value);
        bool all_integral = (entry.non_integral == 0);
        
        /* Uniformly the same value and none missing? */
        if (dense && one_example && one_value)
            result.set_type(INUTILE);
        else if (one_value && uniform_one)
            result.set_type(PRESENCE);
        else if (boolean) result.set_type(BOOLEAN);
        else if (itl->feature_space->info(feat).type()
                     == CATEGORICAL
                 || itl->feature_space->info(feat).type()
                     == STRING)
            result = itl->feature_space->info(feat);
        else if (all_integral && entry.min_value >= 0
                 && entry.max_value <= 255) {
            result.set_type(CATEGORICAL);
            result.set_categorical
                (new Mutable_Categorical_Info((int)entry.max_value + 1));
        }
        else result.set_type(REAL);
    }

    return result;
}


} // namespace ML

