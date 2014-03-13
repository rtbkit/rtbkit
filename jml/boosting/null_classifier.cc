/* null_classifier.cc
   Jeremy Barnes, 25 July 2003
   Copyright (c) 2003 Jeremy Barnes.  All rights reserved.
   $Source$

   Classifier that always outputs 0.
*/

#include "null_classifier.h"
#include "registry.h"
#include "classifier_persist_impl.h"
#include "config_impl.h"


using namespace std;
using namespace DB;


namespace ML {


/*****************************************************************************/
/* NULL_CLASSIFIER                                                           */
/*****************************************************************************/

Null_Classifier::Null_Classifier()
{
}

Null_Classifier::
Null_Classifier(const std::shared_ptr<const Feature_Space> & fs,
                const Feature & predicted)
    : Classifier_Impl(fs, predicted)
{
}

Null_Classifier::
Null_Classifier(const std::shared_ptr<const Feature_Space> & fs,
                const Feature & predicted,
                size_t label_count)
    : Classifier_Impl(fs, predicted, label_count)
{
}

Null_Classifier::~Null_Classifier()
{
}

distribution<float>
Null_Classifier::predict(const Feature_Set & features,
                         PredictionContext * context) const
{
    distribution<float> result(label_count(), 0.0);
    if (!result.empty()) result[0] = 1.0;
    return result;
}

std::string Null_Classifier::print() const
{
    return "";
}

std::vector<ML::Feature> Null_Classifier::all_features() const
{
    std::vector<ML::Feature> result;
    return result;
}

Output_Encoding
Null_Classifier::
output_encoding() const
{
    return OE_PROB;
}

std::string Null_Classifier::class_id() const
{
    return "NULL_CLASSIFIER";
}

namespace {

static const std::string NULL_CLASSIFIER_MAGIC = "NULL_CLASSIFIER";
static const compact_size_t NULL_CLASSIFIER_VERSION = 1;

} // file scope

void Null_Classifier::serialize(DB::Store_Writer & store) const
{
    store << NULL_CLASSIFIER_MAGIC << NULL_CLASSIFIER_VERSION
          << compact_size_t(label_count());
    feature_space()->serialize(store, predicted());
}

Null_Classifier::
Null_Classifier(DB::Store_Reader & store,
                const std::shared_ptr<const Feature_Space> & feature_space)
{
    string magic;
    compact_size_t version;
    store >> magic >> version;
    if (magic != NULL_CLASSIFIER_MAGIC)
        throw Exception("Attempt to reconstitute \"" + magic
                        + "\" with Null_Classifier reconstitutor");
    if (version > NULL_CLASSIFIER_VERSION)
        throw Exception(format("Attemp to reconstitute Null_Classifier "
                               "version %zd, only <= %zd supported",
                               version.size_,
                               NULL_CLASSIFIER_VERSION.size_));
    
    compact_size_t label_count(store);
    predicted_ = Feature(-1);
    if (version >= 1)  // added in version 1
        feature_space->reconstitute(store, predicted_);
    
    Classifier_Impl::init(feature_space, predicted_);
}

void Null_Classifier::
reconstitute(DB::Store_Reader & store,
             const std::shared_ptr<const Feature_Space> & features)
{
    /* Implement the strong exception guarantee. */
    Null_Classifier new_me(store, features);
    swap(new_me);
}

Null_Classifier * Null_Classifier::make_copy() const
{
    return new Null_Classifier(*this);
}

/*****************************************************************************/
/* REGISTRATION                                                              */
/*****************************************************************************/

namespace {

Register_Factory<Classifier_Impl, Null_Classifier>
NULL_REGISTER("NULL_CLASSIFIER");

} // file scope

} // namespace ML

