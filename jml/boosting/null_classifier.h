/* null_classifier.h                                               -*- C++ -*-
   Jeremy Barnes, 25 July 2003
   Copyright (c) 2003 Jeremy Barnes.  All rights reserved.
   $Source$

   A classifier that always outputs zero.
*/

#ifndef __boosting__null_classifier_h__
#define __boosting__null_classifier_h__


#include "classifier.h"



namespace ML {


/*****************************************************************************/
/* NULL_CLASSIFIER                                                           */
/*****************************************************************************/

/** Classifier that always returns the zero label. */

class Null_Classifier : public Classifier_Impl {
public:
    Null_Classifier();
    Null_Classifier(const std::shared_ptr<const Feature_Space> & fs,
                    const Feature & predicted);
    Null_Classifier(const std::shared_ptr<const Feature_Space> & fs,
                    const Feature & predicted, size_t label_count);

    virtual ~Null_Classifier();

    using Classifier_Impl::predict;

    /** Predict the score for all classes. */
    virtual distribution<float>
    predict(const Feature_Set & features,
            PredictionContext * context = 0) const;

    virtual std::string print() const;

    virtual std::vector<Feature> all_features() const;

    virtual Output_Encoding output_encoding() const;

    virtual std::string class_id() const;

    /** Serialization and reconstitution. */
    virtual void serialize(DB::Store_Writer & store) const;
    virtual void reconstitute(DB::Store_Reader & store,
                              const std::shared_ptr<const Feature_Space>
                                  & feature_space);
    Null_Classifier(DB::Store_Reader & store,
                    const std::shared_ptr<const Feature_Space> & fs);

    /** Allow polymorphic copying. */
    virtual Null_Classifier * make_copy() const;
};

} // namespace ML




#endif /* __boosting__classifier_h__ */
