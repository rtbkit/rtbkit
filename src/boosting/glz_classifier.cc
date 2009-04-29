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


using namespace std;
using namespace DB;
using namespace Stats;



namespace ML {


/*****************************************************************************/
/* GLZ_CLASSIFIER                                                            */
/*****************************************************************************/

GLZ_Classifier::GLZ_Classifier()
    : add_bias(true), link(LOGIT)
{
}

GLZ_Classifier::
GLZ_Classifier(const boost::shared_ptr<const Feature_Space> & fs,
               const Feature & predicted)
    : Classifier_Impl(fs, predicted), add_bias(true), link(LOGIT)
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

distribution<float> GLZ_Classifier::
decode(const Feature_Set & feature_set) const
{
    boost::shared_ptr<const Dense_Feature_Space> dense_fs
        = boost::dynamic_pointer_cast<const Dense_Feature_Space>
            (feature_space_);
    if (dense_fs && features.size() == dense_fs->variable_count())
        return dense_fs->decode(feature_set);

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

distribution<float>
GLZ_Classifier::predict(const Feature_Set & features) const
{
    return predict(decode(features));
}

distribution<float>
GLZ_Classifier::predict(const distribution<float> & features_c) const
{
    //cerr << "glz_classifier: predict: features = " << features_c << endl;
    //cerr << "features.size() = " << features.size() << endl;

    distribution<float> features = features_c;
    if (add_bias) features.push_back(1.0);  // add bias term

    distribution<float> result(label_count());
    for (unsigned i = 0;  i < result.size();  ++i)
        result[i] = apply_link_inverse((features * weights[i]).total(), link);

    return result;
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
    return "GLZ_Classifier";
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

