/* naive_bayes.cc
   Jeremy Barnes, 13 March 2004
   Copyright (c) 2004 Jeremy Barnes.  All rights reserved.
   $Source$

   Implementation of the Naive Bayes classifier.
*/

#include "naive_bayes.h"
#include "classifier_persist_impl.h"
#include "jml/arch/simd_vector.h"
#include "jml/stats/distribution_ops.h"
#include "stump_training.h"
#include "jml/utils/vector_utils.h"
#include "config_impl.h"
#include "jml/algebra/multi_array_utils.h"
#include "jml/utils/pair_utils.h"


using namespace std;
using namespace DB;


/*****************************************************************************/
/* NAIVE_BAYES                                                               */
/*****************************************************************************/


namespace ML {

Naive_Bayes::Naive_Bayes()
{
}

Naive_Bayes::
Naive_Bayes(std::shared_ptr<const Feature_Space> feature_space,
            const Feature & feature)
    : Classifier_Impl(feature_space, feature)
{
}

Naive_Bayes::~Naive_Bayes()
{
}

void Naive_Bayes::swap(Naive_Bayes & other)
{
    Classifier_Impl::swap(other);
    features.swap(other.features);
    swap_multi_arrays(probs, other.probs);
    label_priors.swap(other.label_priors);
}

namespace {

} // file scope

distribution<float>
Naive_Bayes::
predict(const Feature_Set & feature_set,
        PredictionContext * context) const
{
    /* How we do the prediction depends upon the relative size of our rules
       array versus the feature set.  If the feature set is much bigger, then
       we look for each rule feature in turn in the feature set.  If the
       number of rules is much bigger than the feature set, then we assume
       that they are all missing to start with, and search for the non-missing
       rules one at a time.  If the two arrays are roughly the same size, then
       we iterate through them together (since both are sorted), and thus
       avoid lookup costs.
    */
    distribution<double> result(label_count());

    //cerr << "features.size() = " << features.size()
    //     << " feature_set.size() = " << feature_set.size() << endl;

    if (features.size() * 10 < feature_set.size()) {
        /* Way more features in the feature set.  We just search for them. */

        /* Go through each feature. */
        for (unsigned f = 0;  f < features.size();  ++f) {
            const Feature & feature = features[f].feature;
            Feature_Set::const_iterator first, last;
            boost::tie(first, last) = feature_set.find(feature);
            
            float weight_true = 0.0, weight_false = 0.0, weight_missing = 0.0;
            
            if (first == last) weight_missing = 1.0;
            else {
                float scale = 1.0 / (last - first);
                while (first != last) {
                    if (!finite((*first).second))
                        weight_missing += scale;
                    else if ((*first).second >= features[f].arg)
                        weight_true += scale;
                    else weight_false += scale;
                    ++first;
                }
            }

            //cerr << "feature " << feature_space()->print(feature)
            //     << " true " << weight_true << " false " << weight_false
            //     << " missing " << weight_missing << endl;
            //distribution<double> result_before = result;

            for (unsigned l = 0;  l < label_count();  ++l) {
                result[l] += (probs[f][false][l]     * weight_false
                              + probs[f][true][l]    * weight_true
                              + probs[f][MISSING][l] * weight_missing);
            }
            //cerr << "added " << result - result_before << endl;

            //cerr << "results now " << result << endl;
        }
    }
    else if (features.size() > 10 * feature_set.size()) {
        /* Way more features in out feature list.  We search for them. */

        /* Initialise assuming that all are missing. */
        copy(missing_total.begin(), missing_total.end(), result.begin());

        /* Go through each feature. */
        for (unsigned f = 0;  f < feature_set.size(); /*no inc*/) {
            const Feature & feature = feature_set[f].first;

            /* Find the number of times this feature occurred. */
            int fe = f;
            while (fe < feature_set.size() && feature_set[fe].first == feature)
                ++fe;

            int nf = fe - f;  // number of times we saw this feature
            float scale = 1.0 / nf;

            //cerr << "f = " << f << "  fe = " << fe << endl;

            /* Find where in our rule array this feature occurred.  Relies
               on the fact that it is sorted. */
            vector<Bayes_Feature>::const_iterator first, last, it;
            first = std::lower_bound(features.begin(), features.end(),
                                     Bayes_Feature(feature, -INFINITY));
            last = std::upper_bound(features.begin(), features.end(),
                                    Bayes_Feature(feature, INFINITY));

            if (first == last) {
                f = fe;
                continue;  // already have the missing
            }

            /* Go through each feature. */
            while (f < fe) {
                float val = feature_set[f].second;
                ++f;
                if (isnan(val)) continue;  // missing
                it = first;
            
                /* Go through each rule. */
                while (it != last) {
                    int rule = it - features.begin();

                    if (val >= it->arg) {
                        for (unsigned l = 0;  l < label_count();  ++l)
                            result[l]
                                += (probs[rule][true][l]
                                    - probs[rule][MISSING][l]) * scale;
                    }
                    else {
                        for (unsigned l = 0;  l < label_count();  ++l)
                            result[l]
                                += (probs[rule][false][l]
                                    - probs[rule][MISSING][l]) * scale;
                    }

                    /* Update.  We have to remove the missing weight as well. */
                    ++it;
                }
            }
            
            f = fe;
            //cerr << "results now " << result << endl;
        }
    }
    else {  // roughly the same size; scan them simultaneously
        distribution<float> accum(label_count(), 0.0);
        int done = 0;

        /* Scan through the two together. */
        Feature_Set::const_iterator fsit = feature_set.begin();
        vector<Bayes_Feature>::const_iterator fit = features.begin();

        while (fsit != feature_set.end() && fit != features.end()) {
            const Feature & feature = fit->feature;

            /* Skip any we don't care about. */
            if ((*fsit).first < feature) { ++fsit;  continue; }

            /* Find how many examples match. */
            Feature_Set::const_iterator first = fsit, last = fsit;
            while (last != feature_set.end()
                   && (*last).first == feature) ++last;
            
            int example_count = last - first;
            unsigned f = fit - features.begin();

            if (example_count == 0)
                SIMD::vec_add(&accum[0], &probs[f][MISSING][0], &accum[0],
                              label_count());
            else if (example_count == 1) {
                if (!finite((*first).second))
                    SIMD::vec_add(&accum[0], &probs[f][MISSING][0], &accum[0],
                                  label_count());
                else
                    SIMD::vec_add(&accum[0],
                                  &probs[f][(*first).second >= fit->arg][0],
                                  &accum[0], label_count());
            }
            else {
                float weight_true = 0.0, weight_false = 0.0,
                    weight_missing = 0.0;
                float scale = 1.0 / example_count;
                while (first != last) {
                    if (!finite((*first).second))
                        weight_missing += scale;
                    else if ((*first).second >= fit->arg)
                        weight_true += scale;
                    else weight_false += scale;
                    ++first;
                }
                
                if (weight_false > 0.0)
                    for (unsigned l = 0;  l < label_count();  ++l)
                        accum[l] += probs[f][false][l] * weight_false;
                
                if (weight_true > 0.0)
                    for (unsigned l = 0;  l < label_count();  ++l)
                        accum[l] += probs[f][true][l] * weight_true;
                
                if (weight_missing > 0.0)
                    for (unsigned l = 0;  l < label_count();  ++l)
                        accum[l] += probs[f][MISSING][l] * weight_missing;
            }
            //cerr << "feature " << f << " accum = " << accum << endl;

            ++done;

            //__builtin_prefetch(&probs[f + 1][0][0] + 48);

            if (done >= 50) {
                for (unsigned l = 0;  l < label_count();  ++l) {
                    result[l] += accum[l];
                    accum[l] = 0.0;
                    done = 0;
                }
            }

            fsit = last;
            ++fit;
        }

        for (unsigned l = 0;  l < label_count();  ++l) {
            result[l] += accum[l];
            accum[l] = 0.0;
        }

        while (fit != features.end()) {
            /* Any left over are missing. */
            unsigned f = fit - features.begin();
            for (unsigned l = 0;  l < label_count();  ++l)
                result[l] += probs[f][MISSING][l];
            ++fit;
        }
    }

    //cerr << "result (before convert) = " << result << endl;
    
    //cerr << "with priors = " << result + log(label_priors) << endl;

    result -= result.max();

    //cerr << "result (before exp) = " << result << endl;

    for (unsigned l = 0;  l < label_count();  ++l)
        result[l] = exp(result[l]) * label_priors[l];

    //cerr << "result (before norm) = " << result << endl;
    
    result.normalize();

    //cerr << "result = " << result << endl;

    return distribution<float>(result.begin(), result.end());
}

float Naive_Bayes::
predict(int label, const Feature_Set & features,
        PredictionContext * context) const
{
    return predict(features)[label];
}

std::string Naive_Bayes::print() const
{
    return "Naive Bayes classifier";
}

std::vector<ML::Feature> Naive_Bayes::all_features() const
{
    std::vector<ML::Feature> result;
    for (std::vector<Bayes_Feature>::const_iterator it = features.begin();
         it != features.end();  ++it)
        result.push_back(it->feature);
    make_vector_set(result);
    return result;
}


Output_Encoding
Naive_Bayes::
output_encoding() const
{
    return OE_PROB;
}
    
std::string Naive_Bayes::class_id() const
{
    return "NAIVE_BAYES";
}

Naive_Bayes * Naive_Bayes::make_copy() const
{
    return new Naive_Bayes(*this);
}

namespace {

static const std::string NAIVE_BAYES_MAGIC = "NAIVE_BAYES";
static const compact_size_t NAIVE_BAYES_VERSION = 2;

} // file scope

void Naive_Bayes::serialize(DB::Store_Writer & store) const
{
    store << NAIVE_BAYES_MAGIC << NAIVE_BAYES_VERSION
          << compact_size_t(label_count());
    feature_space()->serialize(store, predicted_);
    store << compact_size_t(features.size());

    for (unsigned i = 0;  i < features.size();  ++i) {
        feature_space()->serialize(store, features[i].feature);
        store << features[i].arg;
    }

    store << label_priors;

    for (unsigned f = 0;  f < features.size();  ++f)
        for (unsigned i = 0;  i < 3;  ++i)
            for (unsigned l = 0;  l < label_count();  ++l)
                store << probs[f][i][l];

#if 0
    ofstream dump("nb-serial.txt");
    dump << features.size() << " " << label_count() << endl;
    for (unsigned f = 0;  f < features.size();  ++f) {
        dump << "f = " << f << endl;
        for (unsigned i = 0;  i < 3;  ++i) {
            for (unsigned l = 0;  l < label_count();  ++l)
                dump << " " << probs[f][i][l];
            dump << endl;
        }
    }
#endif
}

Naive_Bayes::
Naive_Bayes(DB::Store_Reader & store,
            const std::shared_ptr<const Feature_Space> & feature_space)
{
    string magic;
    compact_size_t version;
    store >> magic >> version;
    if (magic != NAIVE_BAYES_MAGIC)
        throw Exception("Attempt to reconstitute \"" + magic
                        + "\" with naive_bayes reconstitutor");
    if (version > NAIVE_BAYES_VERSION)
        throw Exception(format("Attemp to reconstitute naive_bayes version "
                               "%zd, only <= %zd supported", version.size_,
                               NAIVE_BAYES_VERSION.size_));
    
    compact_size_t label_count(store);

    predicted_ = MISSING_FEATURE;
    if (version >= 2)
        feature_space->reconstitute(store, predicted_);

    compact_size_t num_features(store);

    Classifier_Impl::init(feature_space, predicted_);

    vector<pair<Naive_Bayes::Bayes_Feature, int> > sorted(num_features);

    for (unsigned i = 0;  i < num_features;  ++i) {
        feature_space->reconstitute(store, sorted[i].first.feature);
        store >> sorted[i].first.arg;
        sorted[i].second = i;
    }
    
    //sort_on_first_ascending(sorted);

    store >> label_priors;

    boost::multi_array<float, 3> probs(boost::extents[num_features][3][label_count]);
    for (unsigned f = 0;  f < num_features;  ++f) {
        int f2 = sorted[f].second;
        //cerr << "f2 = " << f2 << endl;
        for (unsigned i = 0;  i < 3;  ++i)
            for (unsigned l = 0;  l < label_count;  ++l)
                store >> probs[f2][i][l];
    }

#if 0
    ofstream dump("nb-reconst.txt");
    dump << num_features.size_ << " " << label_count << endl;
    for (unsigned f = 0;  f < num_features;  ++f) {
        dump << "f = " << f << endl;
        for (unsigned i = 0;  i < 3;  ++i) {
            for (unsigned l = 0;  l < label_count;  ++l)
                dump << " " << probs[f][i][l];
            dump << endl;
        }
    }
#endif

    this->probs = probs;
    //probs.swap(this->probs);
    this->features = vector<Bayes_Feature>
        (first_extractor(sorted.begin()), first_extractor(sorted.end()));

    calc_missing_total();
}

void Naive_Bayes::
reconstitute(DB::Store_Reader & store,
             const std::shared_ptr<const Feature_Space> & features)
{
    /* Implement the strong exception guarantee. */
    Naive_Bayes new_me(store, features);
    swap(new_me);
}

void Naive_Bayes::calc_missing_total()
{
    int nl = label_count();
    
    distribution<double> missing(nl, 0.0);
    
    for (unsigned f = 0;  f < features.size();  ++f)
        for (unsigned l = 0;  l < nl;  ++l)
            missing[l] += probs[f][MISSING][l];

    missing_total = distribution<float>(missing.begin(), missing.end());
}



/*****************************************************************************/
/* REGISTRATION                                                              */
/*****************************************************************************/

namespace {

Register_Factory<Classifier_Impl, Naive_Bayes>
    NAIVE_BAYES_REGISTER("NAIVE_BAYES");

} // file scope

} // namespace ML

