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
                       const boost::shared_ptr<const Feature_Space>
                           & feature_space);

    virtual ~Decoded_Classifier();

    virtual distribution<float>
    predict(const Feature_Set & features) const;
    
    using Classifier_Impl::predict;

    virtual std::string print() const;

    virtual std::vector<Feature> all_features() const;

    virtual Output_Encoding output_encoding() const;

    virtual std::string class_id() const;

    virtual void serialize(DB::Store_Writer & store) const;
    virtual void reconstitute(DB::Store_Reader & store,
                              const boost::shared_ptr<const Feature_Space>
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
