/* decoded_classifier.cc
   Jeremy Barnes, 22 June 2003
   Copyright (c) 2003 Jeremy Barnes.  All rights reserved.
   $Source$

   Implementation of the decoded classifier.
*/

#include "decoded_classifier.h"
#include "jml/db/persistent.h"
#include "classifier_persist_impl.h"
#include "config_impl.h"

using namespace std;
using namespace DB;



namespace ML {


/*****************************************************************************/
/* DECODED_CLASSIFIER                                                        */
/*****************************************************************************/

Decoded_Classifier::Decoded_Classifier()
{
}

Decoded_Classifier::
Decoded_Classifier(const Classifier & classifier, const Decoder & decoder)
    : classifier_(classifier), decoder_(decoder)
{
    Classifier_Impl::init(classifier.feature_space(),
                          classifier.predicted(),
                          classifier.label_count());
}

Decoded_Classifier::~Decoded_Classifier()
{
}

distribution<float>
Decoded_Classifier::predict(const Feature_Set & features,
                            PredictionContext * context) const
{
    distribution<float> result = classifier_.predict(features);
    //cerr << "Decoded_Classifier: nondecoded = " << result << endl;
    //cerr << "Classifier is a " << classifier_.impl->class_id() << endl;
    if (decoder_) result = decoder_.apply(result);
    return result;
}

float
Decoded_Classifier::
predict(int label, const Feature_Set & features,
        PredictionContext * context) const
{
    /* We require predictions for all labels as some decoders work with
       all of them and won't work with just one. */
    return predict(features)[label];
}

bool
Decoded_Classifier::
optimization_supported() const
{
    return classifier_.impl->optimization_supported();
}

bool
Decoded_Classifier::
predict_is_optimized() const
{
    return classifier_.impl->predict_is_optimized();
}

bool
Decoded_Classifier::
optimize_impl(Optimization_Info & info)
{
    return classifier_.impl->optimize_impl(info);
}

Label_Dist
Decoded_Classifier::
optimized_predict_impl(const float * features,
                       const Optimization_Info & info,
                       PredictionContext * context) const
{
    Label_Dist result
        = classifier_.impl->optimized_predict_impl(features, info);
    if (decoder_) result = decoder_.apply(result);
    return result;
}

void
Decoded_Classifier::
optimized_predict_impl(const float * features,
                       const Optimization_Info & info,
                       double * accum,
                       double weight,
                       PredictionContext * context) const
{
    Label_Dist result = optimized_predict_impl(features, info);
    for (unsigned i = 0;  i < result.size();  ++i)
        accum[i] += weight * result[i];
}

float
Decoded_Classifier::
optimized_predict_impl(int label,
                       const float * features,
                       const Optimization_Info & info,
                       PredictionContext * context) const
{
    return optimized_predict_impl(features, info)[label];
}

std::string Decoded_Classifier::print() const
{
    return "Decoded_Classifier";
}

Explanation
Decoded_Classifier::
explain(const Feature_Set & feature_set,
        int label,
        double weight,
        PredictionContext * context) const
{
    return classifier_.impl->explain(feature_set, label, weight);
}

std::vector<ML::Feature> Decoded_Classifier::all_features() const
{
    return classifier_.all_features();
}

Output_Encoding
Decoded_Classifier::
output_encoding() const
{
    return decoder_.output_encoding(classifier_.output_encoding());
}

std::string Decoded_Classifier::class_id() const
{
    return "DECODED_CLASSIFIER";
}

namespace {

static const std::string DECODED_CLASSIFIER_MAGIC = "DECODED_CLASSIFIER";
static const compact_size_t DECODED_CLASSIFIER_VERSION = 0;

} // file scope

void Decoded_Classifier::serialize(DB::Store_Writer & store) const
{
    store << DECODED_CLASSIFIER_MAGIC << DECODED_CLASSIFIER_VERSION;
    store << classifier_;
    store << compact_size_t(decoder_.operator bool());
    //cerr << "decoded_.operator bool() = " << decoder_.operator bool() << endl;
    if (decoder_) store << decoder_;
}

Decoded_Classifier::
Decoded_Classifier(DB::Store_Reader & store,
                   const std::shared_ptr<const Feature_Space> & feature_space)
{
    //cerr << "decoded_classifier reconstitute" << endl;
    //cerr << "feature_space.get() = " << feature_space.get() << endl;
    string magic;
    compact_size_t version;
    store >> magic >> version;

    if (magic != DECODED_CLASSIFIER_MAGIC)
        throw Exception("Attempt to reconstitute \"" + magic
                        + "\" with decoded classifier reconstitutor");
    if (version > DECODED_CLASSIFIER_VERSION)
        throw Exception(format("Attemp to reconstitute decoded classifier "
                               "version %zd, only <= %zd supported",
                               version.size_,
                               DECODED_CLASSIFIER_VERSION.size_));

    //cerr << "magic done" << endl;

    // Provide the feature space to the reconstitutor
    FS_Context context(feature_space);

    //cerr << "classifier... ";
    classifier_.reconstitute(store, feature_space);
    //cerr << "done." << endl;
    //store >> classifier_;

    //cerr << "classifier_.predicted() = " << classifier_.predicted() << endl;

    compact_size_t is_decoder(store);
    //cerr << "is_decoder = " << is_decoder.size_ << endl;
    //cerr << "decoder... ";
    if (is_decoder) store >> decoder_;
    else decoder_ = Decoder();
    //cerr << "done." << endl;
    Classifier_Impl::init(feature_space, classifier_.predicted(),
                          classifier_.label_count());
    //cerr << "my feature space is " << this->feature_space().get() << endl;

    //cerr << "predicted() = " << predicted() << endl;
}

void Decoded_Classifier::
reconstitute(DB::Store_Reader & store,
             const std::shared_ptr<const Feature_Space> & features)
{
    /* Implement the strong exception guarantee. */
    Decoded_Classifier new_me(store, features);
    swap(new_me);
}

Decoded_Classifier * Decoded_Classifier::make_copy() const
{
    return new Decoded_Classifier(*this);
}


/*****************************************************************************/
/* REGISTRATION                                                              */
/*****************************************************************************/

namespace {

Register_Factory<Classifier_Impl, Decoded_Classifier>
    REG1("DECODED_CLASSIFIER");

} // file scope


} // namespace ML

