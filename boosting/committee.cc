/* committee.cc
   Jeremy Barnes, 15 March 2006
   Copyright (c) 2006 Jeremy Barnes.  All rights reserved.
   $Source$

   A committee of classifiers.
*/

#include "committee.h"
#include <memory>
#include "jml/utils/string_functions.h"
#include "jml/db/persistent.h"
#include "config_impl.h"
#include "classifier_persist_impl.h"
#include <set>


using namespace std;
using namespace DB;


namespace ML {


/*****************************************************************************/
/* COMMITTEE                                                                 */
/*****************************************************************************/

Committee::
Committee()
    : optimized_(false)
{
}

Committee::
Committee(DB::Store_Reader & store,
          const std::shared_ptr<const Feature_Space> & feature_space)
    : optimized_(false)
{
    reconstitute(store, feature_space);
}

Committee::
Committee(const std::shared_ptr<const Feature_Space>
             & feature_space,
          const Feature & predicted)
    : Classifier_Impl(feature_space, predicted),
      optimized_(false)
{
}

float
Committee::
predict(int label, const Feature_Set & features,
        PredictionContext * context) const
{
    if (label >= bias.size())
        throw Exception("Committee::predict(): invalid label");

    float result = bias[label];

    for (unsigned i = 0;  i < classifiers.size();  ++i) {
        if (weights[i] == 0.0) continue;
        result += weights[i] * classifiers[i]->predict(label, features, context);
    }

    return result;
}

Label_Dist
Committee::
predict(const Feature_Set & features,
        PredictionContext * context) const
{
    Label_Dist result = bias;

    for (unsigned i = 0;  i < classifiers.size();  ++i) {
        if (weights[i] == 0.0) continue;

        Label_Dist sub_result = classifiers[i]->predict(features, context);

        result += weights[i] * sub_result;
    }

    return result;
}

bool
Committee::
optimization_supported() const
{
    return true;
}

bool
Committee::
predict_is_optimized() const
{
    return optimized_;
}

bool
Committee::
optimize_impl(Optimization_Info & info)
{
    bool any_succeeded = false;

    for (unsigned i = 0;  i < classifiers.size();  ++i) {
        bool succeeded = classifiers[i]->optimize_impl(info);
        if (succeeded) any_succeeded = true;
    }

    return optimized_ = any_succeeded;
}

Label_Dist
Committee::
optimized_predict_impl(const float * features,
                       const Optimization_Info & info,
                       PredictionContext * context) const
{
    int nl = bias.size();

    double accum[nl];
    std::copy(&bias[0], &bias[0] + nl, accum);

    for (unsigned i = 0;  i < classifiers.size();  ++i) {
        if (weights[i] == 0.0) continue;

        classifiers[i]
            ->optimized_predict_impl(features, info, accum, weights[i], context);
    }

    return Label_Dist(accum, accum + nl);
}

void
Committee::
optimized_predict_impl(const float * features,
                       const Optimization_Info & info,
                       double * accum,
                       double weight,
                       PredictionContext * context) const
{
    int nl = bias.size();

    for (unsigned i = 0;  i < nl;  ++i)
        accum[i] += weight * bias[i];

    for (unsigned i = 0;  i < classifiers.size();  ++i) {
        if (weights[i] == 0.0) continue;
        classifiers[i]
            ->optimized_predict_impl(features, info, accum,
                                     weight * weights[i], context);
    }
}

float
Committee::
optimized_predict_impl(int label,
                       const float * features,
                       const Optimization_Info & info,
                       PredictionContext * context) const
{
    if (label >= bias.size())
        throw Exception("Committee::predict(): invalid label");

    float result = bias[label];

    for (unsigned i = 0;  i < classifiers.size();  ++i) {
        if (weights[i] == 0.0) continue;
        result = result
            + weights[i]
            * classifiers[i]->optimized_predict_impl(label, features, info,
                                                     context);
    }

    return result;
}

Explanation
Committee::
explain(const Feature_Set & feature_set,
        int label,
        double weight,
        PredictionContext * context) const
{
    Explanation result(feature_space(), weight);

    for (unsigned i = 0;  i < classifiers.size();  ++i) {
        if (weights[i] == 0.0) continue;
        result.add(classifiers[i]->explain(feature_set, label, 1.0, context),
                   weight * weights[i]);
    }

    return result;
}

void
Committee::
add(std::shared_ptr<Classifier_Impl> classifier, float weight)
{
    if (!classifier)
        throw Exception("Committee::add(): attempt to add null classifier");
    if (classifier->feature_space() != feature_space())
        throw Exception("Committee::add(): attempt to add classifier with "
                        "a different feature space");
    if (classifier->predicted() != predicted())
        throw Exception("Committee::add(): attempt to add classifier that "
                        "predicts the wrong feature");

    if (classifiers.empty()) {
        bias.clear();
        bias.resize(classifier->label_count(), 0.0);
    }

    classifiers.push_back(classifier);
    weights.push_back(weight);
    optimized_ = false;
}

std::string
Committee::
print() const
{
    string result
        = format("Committee of %zd classifiers", classifiers.size());
    return result;
}

std::vector<Feature>
Committee::
all_features() const
{
    set<Feature> result_set;

    for (unsigned i = 0;  i < classifiers.size();  ++i) {
        if (weights[i] == 0.0) continue;
        vector<Feature> child_features = classifiers[i]->all_features();
        result_set.insert(child_features.begin(), child_features.end());
    }

    return vector<Feature>(result_set.begin(), result_set.end());
}

Output_Encoding
Committee::
output_encoding() const
{
    return encoding;
}

namespace {

static const std::string COMMITTEE_MAGIC = "COMMITTEE";
static const compact_size_t COMMITTEE_VERSION = 0;

} // file scope

void
Committee::
serialize(DB::Store_Writer & store) const
{
    store << COMMITTEE_MAGIC << COMMITTEE_VERSION;
    feature_space()->serialize(store, predicted_);
    store << bias << weights;
    for (unsigned i = 0;  i < classifiers.size();  ++i)
        classifiers[i]->poly_serialize(store, false /* write_fs */);
    store << string("END COMMITTEE");
}

void
Committee::
reconstitute(DB::Store_Reader & store,
             const std::shared_ptr<const Feature_Space> & features)
{
    /* Implement the strong exception guarantee, except for the store. */
    string magic;
    compact_size_t version;
    store >> magic >> version;
    if (magic != COMMITTEE_MAGIC)
        throw Exception("Attempt to reconstitute \"" + magic
                        + "\" with committee reconstitutor");
    if (version > COMMITTEE_VERSION)
        throw Exception(format("Attemp to reconstitute committee "
                               "version %zd, only <= %zd supported",
                               version.size_,
                               COMMITTEE_VERSION.size_));
    
    features->reconstitute(store, predicted_);
    Committee new_me(features, predicted_);
    store >> new_me.bias >> new_me.weights;

    //cerr << "weights = " << new_me.weights << endl;

    new_me.classifiers.resize(new_me.weights.size());
    for (unsigned i = 0;  i < new_me.weights.size();  ++i) {
        //cerr << "reconstituting classifier " << i << endl;
        new_me.classifiers[i]
            = Classifier_Impl::poly_reconstitute(store, features);
        //cerr << "got classifier of type "
        //     << demangle(typeid(*classifiers[i]).name()) << endl;
    }
 
    string canary;
    store >> canary;

    if (canary != "END COMMITTEE")
        throw Exception("Committee::reconstitute(): invalid canary \""
                        + canary + "\"");
   
    new_me.optimized_ = false;

    swap(new_me);
}

Committee *
Committee::
make_copy() const
{
    auto_ptr<Committee> result(new Committee(*this));
    for (unsigned i = 0;  i < classifiers.size();  ++i)
        result->classifiers[i].reset(result->classifiers[i]->make_copy());
    return result.release();
}

/*****************************************************************************/
/* REGISTRATION                                                              */
/*****************************************************************************/

namespace {

Register_Factory<Classifier_Impl, Committee>
    STUMP_REGISTER("COMMITTEE");

} // file scope

} // namespace ML
