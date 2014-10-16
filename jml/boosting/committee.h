/* committee.h                                                     -*- C++ -*-
   Jeremy Barnes, 25 March 2006
   Copyright (c) 2006 Jeremy Barnes  All rights reserved.
   $Source$

   A classifier that is actually a committee of sub-classifiers.
*/

#ifndef __boosting__committee_h__
#define __boosting__committee_h__


#include "classifier.h"


namespace ML {


/*****************************************************************************/
/* COMMITTEE                                                                 */
/*****************************************************************************/

/** A committee of arbitrary classifiers.
*/

class Committee : public Classifier_Impl {
public:
    Committee();

    Committee(DB::Store_Reader & store,
                   const std::shared_ptr<const Feature_Space>
                       & feature_space);
    
    Committee(const std::shared_ptr<const Feature_Space>
                       & feature_space,
                   const Feature & predicted);

    std::vector<std::shared_ptr<Classifier_Impl> > classifiers;
    distribution<float> weights; ///< Weigths of each classifier
    distribution<float> bias;    ///< Bias to add to each label
    Output_Encoding encoding;    ///< What type of output we produce

    /** Swap two Committee objects.  Guaranteed not to throw an
        exception. */
    void swap(Committee & other)
    {
        Classifier_Impl::swap(other);
        classifiers.swap(other.classifiers);
        weights.swap(other.weights);
        bias.swap(other.bias);
    }

    void add(std::shared_ptr<Classifier_Impl> classifier, float weight = 1.0);
    
    using Classifier_Impl::predict;

    /** Predict the score for a single class. */
    virtual float predict(int label, const Feature_Set & features,
                          PredictionContext * context = 0) const;

    /** Predict all classes. */
    virtual distribution<float>
    predict(const Feature_Set & features,
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

    virtual Explanation explain(const Feature_Set & feature_set,
                                int label,
                                double weight = 1.0,
                                PredictionContext * context = 0) const;

    virtual std::string print() const;

    virtual std::vector<Feature> all_features() const;

    virtual Output_Encoding output_encoding() const;

    /** Serialization and reconstitution. */
    virtual void serialize(DB::Store_Writer & store) const;
    virtual void
    reconstitute(DB::Store_Reader & store,
                 const std::shared_ptr<const Feature_Space> & features);

    virtual std::string class_id() const { return "COMMITTEE"; }

    virtual Committee * make_copy() const;

private:
    bool optimized_;
};

} // namespace ML

#endif /* __boosting__committee_h__ */
