/* transformed_classifier.h                                        -*- C++ -*-
   Jeremy Barnes, 27 February 2006.
   Copyright (c) 2006 Jeremy Barnes.  All rights reserved.
   $Source$

   A classifier that transforms its feature set before calling a
   child classifier.
*/

#ifndef __boosting__transformed_classifier_h__
#define __boosting__transformed_classifier_h__

#include "classifier.h"
#include "feature_transformer.h"

namespace ML {


/*****************************************************************************/
/* TRANSFORMED_CLASSIFIER                                                    */
/*****************************************************************************/

class Transformed_Classifier
    : public Classifier_Impl {
public:
    Transformed_Classifier();

    Transformed_Classifier(const Feature_Transformer & transformer,
                           const Classifier & classifier);

    Transformed_Classifier(DB::Store_Reader & store,
                           const std::shared_ptr<const Feature_Space>
                               & feature_space);

    virtual ~Transformed_Classifier();

    using Classifier_Impl::predict;

    virtual distribution<float>
    predict(const Feature_Set & features,
            PredictionContext * context = 0) const;
    
    virtual std::string print() const;
    
    virtual std::vector<Feature> all_features() const;

    virtual Output_Encoding output_encoding() const;
    
    virtual std::string class_id() const;
    
    virtual void serialize(DB::Store_Writer & store) const;
    virtual void reconstitute(DB::Store_Reader & store,
                              const std::shared_ptr<const Feature_Space>
                                  & feature_space);

    virtual Transformed_Classifier * make_copy() const;

    const Classifier & classifier() const
    {
        return classifier_;
    }

    Classifier & classifier()
    {
        return classifier_;
    }

    const Feature_Transformer & transformer() const
    {
        return transformer_;
    }

    Feature_Transformer & transformer()
    {
        return transformer_;
    }

    void swap(Transformed_Classifier & other)
    {
        Classifier_Impl::swap(other);
        transformer_.swap(other.transformer_);
        classifier_.swap(other.classifier_);
    }

private:
    Feature_Transformer transformer_;
    Classifier classifier_;
};


} // namespace ML

#endif /* __boosting__transformed_classifier_h__ */
