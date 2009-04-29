/* glz_classifier.h                                                -*- C++ -*-
   Jeremy Barnes, 6 August 2003
   Copyright (c) 2003 Jeremy Barnes.  All rights reserved.
   $Source$

   Classifier based upon a generalised linear model.  Currently only for
   binary predictions.
*/

#ifndef __boosting__glz_classifier_h__
#define __boosting__glz_classifier_h__

#include "boosting/classifier.h"
#include "algebra/irls.h"


namespace ML {


/*****************************************************************************/
/* GLZ_CLASSIFIER                                                            */
/*****************************************************************************/

/** A classifier.  Basic class. */

class GLZ_Classifier : public Classifier_Impl {
public:
    GLZ_Classifier();

    /** Construct for the given feature space, to classify into the given
        number of labels. */
    GLZ_Classifier(const boost::shared_ptr<const Feature_Space> & fs,
                   const Feature & predicted);

    GLZ_Classifier(DB::Store_Reader & store,
                   const boost::shared_ptr<const Feature_Space> & fs);

    /** Reconstitute without a feature space.  For when we are using the
        non Feature_Set predict method, which doesn't need features to ever
        be encoded or decoded.

        \param store        store to reconstitute from
    */
    GLZ_Classifier(DB::Store_Reader & store);
    
    virtual ~GLZ_Classifier();


    /** Do we add a constant bias term to the GLZ?  Usually this will be
        yes (it is by default). */
    bool add_bias;
    
    /** Parameters.  For each label we have a vector of input parameters,
        one for each of the features. */
    std::vector<distribution<float> > weights;

    /** The features which correspond to our variables. */
    std::vector<Feature> features;

    /** The link function we are using. */
    Link_Function link;

    /** Turn a feature set into a dense vector. */
    distribution<float> decode(const Feature_Set & features) const;

    using Classifier_Impl::predict;

    /** Predict the score for all classes. */
    virtual distribution<float>
    predict(const Feature_Set & features) const;

    /** Predict, but from a feature vector rather than a feature set. */
    distribution<float> predict(const distribution<float> & features) const;

    virtual std::string print() const;

    virtual std::vector<Feature> all_features() const;

    virtual Output_Encoding output_encoding() const;

    virtual std::string class_id() const;

    /** Serialization and reconstitution. */
    virtual void serialize(DB::Store_Writer & store) const;
    virtual void reconstitute(DB::Store_Reader & store,
                              const boost::shared_ptr<const Feature_Space>
                                  & feature_space);

    /** Reconstitute without a feature space.  For when we are using the
        non Feature_Set predict method, which doesn't need features to ever
        be encoded or decoded.

        \param store        store to reconstitute from
    */
    void reconstitute(DB::Store_Reader & store);
                              
    /** Allow polymorphic copying. */
    virtual GLZ_Classifier * make_copy() const;
};

} // namespace ML



#endif /* __boosting__glz_classifier_h__ */
