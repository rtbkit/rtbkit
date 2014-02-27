/* decoded_classifier.h                                            -*- C++ -*-
   Jeremy Barnes, 22 June 2003
   Copyright (c) 2003 Jeremy Barnes.  All rights reserved.
   $Source$

   An adaptor that takes a decoder and a classifier and turns them into
   another classifier, with its output decoded.
*/

#ifndef __boosting__decoded_classifier_h__
#define __boosting__decoded_classifier_h__

#include "classifier.h"
#include "decoder.h"


namespace ML {


/*****************************************************************************/
/* DECODED_CLASSIFIER                                                        */
/*****************************************************************************/

class Decoded_Classifier : public Classifier_Impl {
public:
    Decoded_Classifier();

    Decoded_Classifier(const Classifier & classifier,
                       const Decoder & decoder);

    Decoded_Classifier(DB::Store_Reader & store,
                       const std::shared_ptr<const Feature_Space>
                           & feature_space);

    virtual ~Decoded_Classifier();

    virtual Label_Dist
    predict(const Feature_Set & features,
            PredictionContext * context = 0) const;
    
    virtual float
    predict(int label, const Feature_Set & feaures,
            PredictionContext * context = 0) const;

    using Classifier_Impl::predict;

    /** Is optimization supported by the classifier? */
    virtual bool optimization_supported() const;

    /** Is predict optimized?  Default returns false; those classifiers which
        a) support optimized predict and b) have had optimize_predict() called
        will override to return true in this case.
    */
    virtual bool predict_is_optimized() const;
    /** Function to override to perform the optimization.  Default will
        simply modify the optimization info to indicate that optimization
        had failed.
    */
    virtual bool
    optimize_impl(Optimization_Info & info);

    /** Optimized predict for a dense feature vector.
        This is the worker function that all classifiers that implement the
        optimized predict should override.  The default implementation will
        convert to a Feature_Set and will call the non-optimized predict.
    */
    virtual Label_Dist
    optimized_predict_impl(const float * features,
                           const Optimization_Info & info,
                           PredictionContext * context = 0) const;
    
    virtual void
    optimized_predict_impl(const float * features,
                           const Optimization_Info & info,
                           double * accum,
                           double weight,
                           PredictionContext * context = 0) const;

    virtual float
    optimized_predict_impl(int label,
                           const float * features,
                           const Optimization_Info & info,
                           PredictionContext * context = 0) const;


    virtual Explanation explain(const Feature_Set & feature_set,
                                int label,
                                double weight = 1.0,
                                PredictionContext * context = 0) const;

    virtual std::string print() const;

    virtual std::vector<Feature> all_features() const;

    virtual Output_Encoding output_encoding() const;

    virtual std::string class_id() const;

    virtual void serialize(DB::Store_Writer & store) const;
    virtual void reconstitute(DB::Store_Reader & store,
                              const std::shared_ptr<const Feature_Space>
                                  & feature_space);

    virtual Decoded_Classifier * make_copy() const;

    const Classifier & classifier() const
    {
        return classifier_;
    }

    Classifier & classifier()
    {
        return classifier_;
    }

    const Decoder & decoder() const
    {
        return decoder_;
    }

    Decoder & decoder()
    {
        return decoder_;
    }

    void swap(Decoded_Classifier & other)
    {
        Classifier_Impl::swap(other);
        classifier_.swap(other.classifier_);
        decoder_.swap(other.decoder_);
    }

private:
    Classifier classifier_;
    Decoder decoder_;
};


} // namespace ML



#endif /* __boosting__decoded_classifier_h__ */
