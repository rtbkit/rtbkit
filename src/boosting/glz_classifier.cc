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
#include "algebra/irls.h"
#include <boost/timer.hpp>
#include "training_index.h"
#include "config_impl.h"
#include <limits>
#include "utils/vector_utils.h"
#include "arch/backtrace.h"

using namespace std;
using namespace DB;
using namespace Stats;



namespace ML {


/*****************************************************************************/
/* GLZ_CLASSIFIER                                                            */
/*****************************************************************************/

GLZ_Classifier::GLZ_Classifier()
    : add_bias(true), link(LOGIT), optimized_(false)
{
}

GLZ_Classifier::
GLZ_Classifier(const boost::shared_ptr<const Feature_Space> & fs,
               const Feature & predicted)
    : Classifier_Impl(fs, predicted), add_bias(true), link(LOGIT),
      optimized_(false)
{
}

GLZ_Classifier::
GLZ_Classifier(DB::Store_Reader & store,
               const boost::shared_ptr<const Feature_Space> & fs)
{
    reconstitute(store, fs);
}

GLZ_Classifier::
GLZ_Classifier(DB::Store_Reader & store)
{
    reconstitute(store);
    set_feature_space(boost::shared_ptr<const Feature_Space>
                      (new Null_Feature_Space()));
}

GLZ_Classifier::~GLZ_Classifier()
{
}

distribution<float>
GLZ_Classifier::
decode(const Feature_Set & feature_set) const
{
    distribution<float> result(features.size());
    
    for (unsigned i = 0;  i < features.size();  ++i) {
        Feature_Set::const_iterator first, last;
        boost::tie(first, last) = feature_set.find(features[i]);
        if (last - first != 1)
            throw Exception("GLZ_Classifier::decode() feature "
                            + feature_space_->print(features[i])
                            + " occurred " + ostream_format(last - first)
                            + " times; exactly 1 required");
        result[i] = (*first).second;
    }
    
    return result;
}

Label_Dist
GLZ_Classifier::predict(const Feature_Set & features) const
{
    distribution<float> features_c = decode(features);
    Label_Dist result = predict(features_c);
    return result;
}

Label_Dist
GLZ_Classifier::predict(const distribution<float> & features_c) const
{
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
            = info.feature_to_optimized_index.find(features[i]);
        if (it == info.feature_to_optimized_index.end())
            throw Exception("GLZ_Classifier::optimize(): feature not found");
        feature_indexes.push_back(it->second);
    }

    cerr << "feature_indexes = " << feature_indexes << endl;
    
    return optimized_ = true;
}

extern bool debug_glz_predict;

Label_Dist
GLZ_Classifier::
optimized_predict_impl(const float * features_c,
                       const Optimization_Info & info) const
{
    return do_predict_impl(features_c, &feature_indexes[0]);
}

void
GLZ_Classifier::
optimized_predict_impl(const float * features_c,
                       const Optimization_Info & info,
                       double * accum_out,
                       double weight) const
{
    do_predict_impl(features_c, &feature_indexes[0], accum_out, weight);
}

float
GLZ_Classifier::
optimized_predict_impl(int label,
                       const float * features_c,
                       const Optimization_Info & info) const
{
    return do_predict_impl(label, features_c, &feature_indexes[0]);
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
        float feat_val = features_c[idx];
        if (!isfinite(feat_val))
            throw Exception("GLZ_Classifier: feature "
                            + feature_space()->print(features[j])
                            + " is not finite");
        
        accum +=  feat_val * weights[label][j];
    }
    
    if (add_bias) accum += weights[label][features.size()];
    
    if (debug_glz_predict) cerr << "  accum = " << accum << endl;

    return apply_link_inverse(accum, link);
}

Label_Dist
GLZ_Classifier::
do_predict_impl(const float * features_c,
                const int * indexes) const
{
    if (debug_glz_predict) {
        cerr << "predict 1: features = " << distribution<float>(features_c, features_c + 10) << endl;
        //backtrace();
    }

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
    if (debug_glz_predict) {
        cerr << "predict 2: features = " << distribution<float>(features_c, features_c + 10) << endl;
        //backtrace();
    }

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
    if (debug_glz_predict)
        cerr << "predict 3: features = " << distribution<float>(features_c, features_c + 10) << endl;

    return do_accum(features_c, indexes, label);
}


std::vector<ML::Feature> GLZ_Classifier::all_features() const
{
    return features;
}

Output_Encoding
GLZ_Classifier::
output_encoding() const
{
    if (label_count() == 1) return OE_PM_INF;
    else return OE_PROB;
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
static const compact_size_t GLZ_CLASSIFIER_VERSION = 2;

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
    for (unsigned i = 0;  i < features.size();  ++i)
        feature_space_->serialize(store, features[i]);
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
    for (unsigned i = 0;  i < features.size();  ++i)
        feature_space_->reconstitute(store, features[i]);
    
    compact_size_t guard(store);
    if (guard != 0x12345)
        throw Exception("GLZ_Classifier::reconstitute(): bad guard value");
    
    init(feature_space_, predicted_, weights.size());

    optimized_ = false;
}

void GLZ_Classifier::
reconstitute(DB::Store_Reader & store,
             const boost::shared_ptr<const Feature_Space> & fs)
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

