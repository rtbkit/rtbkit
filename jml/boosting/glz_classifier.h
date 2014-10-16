/* glz_classifier.h                                                -*- C++ -*-
   Jeremy Barnes, 6 August 2003
   Copyright (c) 2003 Jeremy Barnes.  All rights reserved.
   $Source$

   Classifier based upon a generalised linear model.  Currently only for
   binary predictions.
*/

#ifndef __boosting__glz_classifier_h__
#define __boosting__glz_classifier_h__

#include "jml/boosting/classifier.h"
#include "jml/algebra/irls.h"


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
    GLZ_Classifier(const std::shared_ptr<const Feature_Space> & fs,
                   const Feature & predicted);

    GLZ_Classifier(DB::Store_Reader & store,
                   const std::shared_ptr<const Feature_Space> & fs);

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

    /** Importance.  Same as weights, but gives only the amount by which the
        un-normalized parameter was weighted. */
    std::vector<distribution<float> > importances;

    struct Feature_Spec {
        enum Type {
            VALUE,              ///< Copies the value directly
            VALUE_IF_PRESENT,   ///< Copies value if present, or zero otherwise
            PRESENCE            ///< True if present, false otherwise
            // TODO: add a different boolean for each category in categorical
        };

        Feature_Spec()
            : type(VALUE)
        {
        }

        Feature_Spec(const Feature & feature,
                    Type type = VALUE)
            : feature(feature), type(type)
        {
        }

        Feature feature;
        Type type;
    };

    /** The features which correspond to our variables. */
    std::vector<Feature_Spec> features;

    /** The link function we are using. */
    Link_Function link;

    /** Turn a feature set into a dense vector. */
    distribution<float> extract(const Feature_Set & features) const;

    /** Turn a feature set into a decoded dense vector */
    distribution<float> decode(const Feature_Set & features) const;

    using Classifier_Impl::predict;

    /** Predict the score for all classes. */
    virtual distribution<float>
    predict(const Feature_Set & features,
            PredictionContext * context) const;

    /** Predict, but from a feature vector rather than a feature set. */
    distribution<float> predict(const distribution<float> & features,
                                PredictionContext * context = 0) const;

    /** Predict, but from a mapped feature vector rather than a feature set. */
    distribution<float> predict(const float * features,
                                PredictionContext * context = 0) const;

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

#ifndef JML_TESTING_GLZ_CLASSIFIER
protected:
#endif
    // These actually do the prediction.  They will dereference the features
    // vector in the order given by indexes; if indexes is null then they
    // will do it in unit order

    Label_Dist
    do_predict_impl(const float * features,
                    const int * indexes) const;
    
    void
    do_predict_impl(const float * features,
                    const int * indexes,
                    double * accum,
                    double weight) const;
    float
    do_predict_impl(int label,
                    const float * features,
                    const int * indexes) const;

    // Internal function used to do the actual work
    double
    do_accum(const float * features_c,
             const int * indexes,
             int label) const;

    float decode_value(float feat_val, const Feature_Spec & spec) const;

public:
    virtual Explanation explain(const Feature_Set & feature_set,
                                int label,
                                double weight = 1.0,
                                PredictionContext * context = 0) const;

    virtual std::string print() const;

    virtual std::string summary() const;

    virtual std::vector<Feature> all_features() const;

    virtual Output_Encoding output_encoding() const;

    virtual std::string class_id() const;

    /** Serialization and reconstitution. */
    virtual void serialize(DB::Store_Writer & store) const;
    virtual void reconstitute(DB::Store_Reader & store,
                              const std::shared_ptr<const Feature_Space>
                                  & feature_space);

    /** Reconstitute without a feature space.  For when we are using the
        non Feature_Set predict method, which doesn't need features to ever
        be encoded or decoded.

        \param store        store to reconstitute from
    */
    void reconstitute(DB::Store_Reader & store);
                              
    /** Allow polymorphic copying. */
    virtual GLZ_Classifier * make_copy() const;

    bool optimized_;
    std::vector<int> feature_indexes;
};

} // namespace ML



#endif /* __boosting__glz_classifier_h__ */
