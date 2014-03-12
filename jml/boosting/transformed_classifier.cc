/* transformed_classifier.cc
   Jeremy Barnes, 27 February 2006
   Copyright (c) 2003 Jeremy Barnes.  All rights reserved.
   $Source$

   Implementation of the transformed classifier.
*/

#include "transformed_classifier.h"
#include "jml/db/persistent.h"
#include "classifier_persist_impl.h"
#include "config_impl.h"


using namespace std;
using namespace DB;



namespace ML {


/*****************************************************************************/
/* TRANSFORMED_CLASSIFIER                                                    */
/*****************************************************************************/

Transformed_Classifier::Transformed_Classifier()
{
}

Transformed_Classifier::
Transformed_Classifier(const Feature_Transformer & transformer,
                       const Classifier & classifier)
    : transformer_(transformer), classifier_(classifier)
{
    Classifier_Impl::init(classifier.feature_space(),
                          classifier.predicted(),
                          classifier.label_count());
}

Transformed_Classifier::~Transformed_Classifier()
{
}

distribution<float>
Transformed_Classifier::predict(const Feature_Set & features,
                                PredictionContext * context) const
{
    std::shared_ptr<Feature_Set> tr_features
        = transformer_.transform(features);
    distribution<float> result = classifier_.predict(*tr_features, context);
    return result;
}

std::string Transformed_Classifier::print() const
{
    return "Transformed_Classifier";
}

std::vector<ML::Feature> Transformed_Classifier::all_features() const
{
    return transformer_.features_for(classifier_.all_features());
}

Output_Encoding
Transformed_Classifier::
output_encoding() const
{
    return classifier_.output_encoding();
}

std::string Transformed_Classifier::class_id() const
{
    return "TRANSFORMED_CLASSIFIER";
}

namespace {

static const std::string TRANSFORMED_CLASSIFIER_MAGIC = "TRANSFORMED_CLASSIFIER";
static const compact_size_t TRANSFORMED_CLASSIFIER_VERSION = 0;

} // file scope

void Transformed_Classifier::serialize(DB::Store_Writer & store) const
{
    store << TRANSFORMED_CLASSIFIER_MAGIC << TRANSFORMED_CLASSIFIER_VERSION;
    store << classifier_;
    store << compact_size_t((bool)transformer_);
    if (transformer_) store << transformer_;
}

Transformed_Classifier::
Transformed_Classifier(DB::Store_Reader & store,
                   const std::shared_ptr<const Feature_Space> & feature_space)
{
    string magic;
    compact_size_t version;
    store >> magic >> version;
    
    if (magic != TRANSFORMED_CLASSIFIER_MAGIC)
        throw Exception("Attempt to reconstitute \"" + magic
                        + "\" with transformed classifier reconstitutor");
    if (version > TRANSFORMED_CLASSIFIER_VERSION)
        throw Exception(format("Attemp to reconstitute transformed classifier "
                               "version %zd, only <= %zd supported",
                               version.size_,
                               TRANSFORMED_CLASSIFIER_VERSION.size_));

    // Provide the feature space to the reconstitutor
    FS_Context context(feature_space);
    
    classifier_.reconstitute(store, feature_space);

    compact_size_t is_transformer(store);
    if (is_transformer) store >> transformer_;
    else transformer_ = Feature_Transformer();
    Classifier_Impl::init(feature_space, classifier_.predicted(),
                          classifier_.label_count());
}

void Transformed_Classifier::
reconstitute(DB::Store_Reader & store,
             const std::shared_ptr<const Feature_Space> & features)
{
    /* Implement the strong exception guarantee. */
    Transformed_Classifier new_me(store, features);
    swap(new_me);
}

Transformed_Classifier * Transformed_Classifier::make_copy() const
{
    return new Transformed_Classifier(*this);
}


/*****************************************************************************/
/* REGISTRATION                                                              */
/*****************************************************************************/

namespace {

Register_Factory<Classifier_Impl, Transformed_Classifier>
    REG1("TRANSFORMED_CLASSIFIER");

} // file scope


} // namespace ML

