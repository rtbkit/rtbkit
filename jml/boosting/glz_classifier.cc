/* glz_classifier.cc
   Jeremy Barnes, 6 August 2003
   Copyright (c) 2003 Jeremy Barnes.  All rights reserved.
   $Source$

   Implementation of the GLZ classifier.
*/

#include "glz_classifier.h"
#include "classifier_persist_impl.h"
#include "null_feature_space.h"
#include "dense_features.h"
#include "jml/algebra/irls.h"
#include <boost/timer.hpp>
#include "training_index.h"
#include "config_impl.h"
#include <limits>
#include "jml/utils/vector_utils.h"
#include "jml/compiler/compiler.h"

using namespace std;
using namespace ML::DB;


namespace ML {


/*****************************************************************************/
/* GLZ_CLASSIFIER                                                            */
/*****************************************************************************/

GLZ_Classifier::GLZ_Classifier()
    : add_bias(true), link(LOGIT), optimized_(false)
{
}

GLZ_Classifier::
GLZ_Classifier(const std::shared_ptr<const Feature_Space> & fs,
               const Feature & predicted)
    : Classifier_Impl(fs, predicted), add_bias(true), link(LOGIT),
      optimized_(false)
{
}

GLZ_Classifier::
GLZ_Classifier(DB::Store_Reader & store,
               const std::shared_ptr<const Feature_Space> & fs)
{
    reconstitute(store, fs);
}

GLZ_Classifier::
GLZ_Classifier(DB::Store_Reader & store)
{
    reconstitute(store);
    set_feature_space(std::shared_ptr<const Feature_Space>
                      (new Null_Feature_Space()));
}

GLZ_Classifier::~GLZ_Classifier()
{
}

distribution<float>
GLZ_Classifier::
extract(const Feature_Set & feature_set) const
{
    distribution<float> result(features.size());

    float NaN = std::numeric_limits<float>::quiet_NaN();

    Feature_Set::const_iterator
        prev_last = feature_set.begin(),
        fend = feature_set.end();
    
    for (unsigned i = 0;  i < features.size();  ++i) {
        const Feature & to_find = features[i].feature;

        Feature_Set::const_iterator first, last;

        // Optimization: assume that the features are there in the same order
        // as we wanted to access them, so that we can simply step through
        // rather than having to search for them each time.
        if (prev_last != fend && prev_last.feature() == to_find) {
            last = first = prev_last;
            do {
                ++last;
            } while (last != fend && last.feature() == to_find);
        }
        else {
            boost::tie(first, last) = feature_set.find(features[i].feature);
        }

        prev_last = last;

        switch (features[i].type) {
        case Feature_Spec::VALUE_IF_PRESENT:
            if (first == last || isnan((*first).second)) {
                result[i] = NaN;
                break;
            }
            // fall through

        case Feature_Spec::VALUE:
            if (last - first != 1)
                throw Exception("GLZ_Classifier::decode() feature "
                                + feature_space_->print(features[i].feature)
                                + " occurred " + ostream_format(last - first)
                                + " times; exactly 1 required");
            if (isnan((*first).second))
                throw Exception("GLZ_Classifier::decode() feature "
                                + feature_space_->print(features[i].feature)
                                + " was missing");
                
            result[i] = (*first).second;
            break;

        case Feature_Spec::PRESENCE:
            if (first == last || isnan((*first).second)) {
                result[i] = NaN;
                break;
            }
            result[i] = (*first).second;
            break;
            
        default:
            throw Exception("GLZ_Classifier::decode(): invalid type");
        }
    }
    
    return result;
}

distribution<float>
GLZ_Classifier::
decode(const Feature_Set & feature_set) const
{
    distribution<float> result = extract(feature_set);
    for (unsigned i = 0;  i < result.size();  ++i) {
        result[i] = decode_value(result[i], features[i]);
    }

    return result;
}

Label_Dist
GLZ_Classifier::predict(const Feature_Set & features,
                        PredictionContext * context) const
{
    distribution<float> features_c = extract(features);
    Label_Dist result = predict(features_c);
    return result;
}

Label_Dist
GLZ_Classifier::predict(const distribution<float> & features_c,
                        PredictionContext * context) const
{
    if (features_c.size() != features.size())
        throw Exception("wrong number of features");
    return do_predict_impl(&features_c[0], 0);
}

bool
GLZ_Classifier::
optimization_supported() const
{
    return true;
}

bool
GLZ_Classifier::
predict_is_optimized() const
{
    return optimized_;
}

bool
GLZ_Classifier::
optimize_impl(Optimization_Info & info)
{
    feature_indexes.clear();

    // Fill in the feature order
    for (unsigned i = 0;  i < features.size();  ++i) {
        map<Feature, int>::const_iterator it
            = info.feature_to_optimized_index.find(features[i].feature);
        if (it == info.feature_to_optimized_index.end())
            throw Exception("GLZ_Classifier::optimize(): feature not found");
        feature_indexes.push_back(it->second);
    }

    return optimized_ = true;
}

Label_Dist
GLZ_Classifier::
optimized_predict_impl(const float * features_c,
                       const Optimization_Info & info,
                       PredictionContext * context) const
{
    return do_predict_impl(features_c, &feature_indexes[0]);
}

void
GLZ_Classifier::
optimized_predict_impl(const float * features_c,
                       const Optimization_Info & info,
                       double * accum_out,
                       double weight,
                       PredictionContext * context) const
{
    do_predict_impl(features_c, &feature_indexes[0], accum_out, weight);
}

float
GLZ_Classifier::
optimized_predict_impl(int label,
                       const float * features_c,
                       const Optimization_Info & info,
                       PredictionContext * context) const
{
    return do_predict_impl(label, features_c, &feature_indexes[0]);
}

float
GLZ_Classifier::
decode_value(float feat_val, const Feature_Spec & spec) const
{
    if (JML_UNLIKELY(isnan(feat_val))) {
        switch (spec.type) {
        case Feature_Spec::VALUE:
            feat_val = 0.0;
#if 0
            throw Exception("GLZ_Classifier: feature "
                            + feature_space()->print(spec.feature)
                            + " is missing");
#endif
        case Feature_Spec::VALUE_IF_PRESENT:
        case Feature_Spec::PRESENCE:
            feat_val = 0.0;
            break;
        default:
            throw Exception("invalid feature spec type");
        }
    }
    else if (JML_UNLIKELY(spec.type == Feature_Spec::PRESENCE))
        feat_val = 1.0;
    else if (JML_UNLIKELY(!isfinite(feat_val)))
        throw Exception("GLZ_Classifier: feature "
                        + feature_space()->print(spec.feature)
                        + " is not finite");

    return feat_val;
}

double
GLZ_Classifier::
do_accum(const float * features_c,
         const int * indexes,
         int label) const
{
    double accum = 0.0;
    
    for (unsigned j = 0;  j < features.size();  ++j) {
        int idx = (indexes ? indexes[j] : j);
        float feat_val = decode_value(features_c[idx], features[j]);
        accum +=  feat_val * weights[label][j];
    }

    if (add_bias) accum += weights[label][features.size()];

    //cerr << "do accum " << label << " = " << accum << endl;
    
    
    return apply_link_inverse(accum, link);
}

Label_Dist
GLZ_Classifier::
do_predict_impl(const float * features_c,
                const int * indexes) const
{
    Label_Dist result(label_count());

    for (unsigned i = 0;  i < result.size();  ++i)
        result[i] = do_accum(features_c, indexes, i);
    

    return result;
}

void
GLZ_Classifier::
do_predict_impl(const float * features_c,
                const int * indexes,
                double * accum,
                double weight) const
{
    int nl = label_count();

    for (unsigned i = 0;  i < nl;  ++i)
        accum[i] += weight * do_accum(features_c, indexes, i);
}

float
GLZ_Classifier::
do_predict_impl(int label,
                const float * features_c,
                const int * indexes) const
{
    return do_accum(features_c, indexes, label);
}


std::vector<ML::Feature>
GLZ_Classifier::
all_features() const
{
    vector<ML::Feature> result;
    for (unsigned i = 0;  i < features.size();  ++i)
        if (i == 0 || features[i].feature != result.back())
            result.push_back(features[i].feature);
    return result;
}

Output_Encoding
GLZ_Classifier::
output_encoding() const
{
    if (label_count() == 1) return OE_PM_INF;
    else return OE_PROB;
}

Explanation
GLZ_Classifier::
explain(const Feature_Set & feature_set,
        int label,
        double weight,
        PredictionContext * context) const
{
    Explanation result(feature_space(), weight);

    for (unsigned j = 0;  j < features.size();  ++j) {
        float feat_val = decode_value(feature_set[features[j].feature],
                                      features[j]);
        result.feature_weights[features[j].feature]
            += weight * weights[label][j] * feat_val;
    }
    
    if (add_bias) result.bias += weight * weights[label][features.size()];

    return result;
}


std::string GLZ_Classifier::print() const
{
    return "GLZ_Classifier: link " + ML::print(link);
}

std::string GLZ_Classifier::summary() const
{
    return "GLZ_Classifier: link " + ML::print(link);
}

namespace {

static const std::string GLZ_CLASSIFIER_MAGIC = "GLZ_CLASSIFIER";
static const compact_size_t GLZ_CLASSIFIER_VERSION = 3;

} // file scope

std::string GLZ_Classifier::class_id() const
{
    return GLZ_CLASSIFIER_MAGIC;
}

void GLZ_Classifier::serialize(DB::Store_Writer & store) const
{
    store << GLZ_CLASSIFIER_MAGIC << GLZ_CLASSIFIER_VERSION;
    feature_space()->serialize(store, predicted_);
    store << (int)add_bias << weights << link;
    store << compact_size_t(features.size());
    for (unsigned i = 0;  i < features.size();  ++i) {
        feature_space_->serialize(store, features[i].feature);
        store << (char)features[i].type;
    }
    store << compact_size_t(0x12345);
}

void GLZ_Classifier::
reconstitute(DB::Store_Reader & store)
{
    string magic;
    compact_size_t version;
    store >> magic >> version;
    if (magic != GLZ_CLASSIFIER_MAGIC)
        throw Exception("Attempt to reconstitute \"" + magic
                                + "\" with boosted stumps reconstitutor");
    if (version != GLZ_CLASSIFIER_VERSION)
        throw Exception(format("Attemp to reconstitute GLZ classifier "
                               "version %zd, only %zd supported",
                               version.size_,
                               GLZ_CLASSIFIER_VERSION.size_));

    if (!feature_space_)
        throw Exception("GLZ_Classifier::reconstitute(): feature space not "
                        "initialised");
    
    predicted_ = MISSING_FEATURE;
    if (version >= 2) // added in version 2
        feature_space()->reconstitute(store, predicted_);
    
    int add_bias_;  store >> add_bias_;  add_bias = add_bias_;
    store >> weights >> link;
    
    compact_size_t nf(store);
    features.resize(nf);

    if (version < 3) {
        for (unsigned i = 0;  i < features.size();  ++i)
            feature_space_->reconstitute(store, features[i].feature);
    }
    else {
        for (unsigned i = 0;  i < features.size();  ++i) {
            feature_space_->reconstitute(store, features[i].feature);
            char c;  store >> c;
            features[i].type = Feature_Spec::Type(c);
        }
    }
    compact_size_t guard(store);
    if (guard != 0x12345)
        throw Exception("GLZ_Classifier::reconstitute(): bad guard value");
    
    init(feature_space_, predicted_, weights.size());

    optimized_ = false;
}

void GLZ_Classifier::
reconstitute(DB::Store_Reader & store,
             const std::shared_ptr<const Feature_Space> & fs)
{
    feature_space_ = fs;
    reconstitute(store);
}

GLZ_Classifier * GLZ_Classifier::make_copy() const
{
    return new GLZ_Classifier(*this);
}


/*****************************************************************************/
/* REGISTRATION                                                              */
/*****************************************************************************/

namespace {

Register_Factory<Classifier_Impl, GLZ_Classifier>
    GLZC_REGISTER(GLZ_CLASSIFIER_MAGIC);

} // file scope



} // namespace ML

