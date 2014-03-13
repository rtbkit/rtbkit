/* stump.h                                                         -*- C++ -*-
   Jeremy Barnes, 6 June 2003
   Copyright (c) 2003 Jeremy Barnes.  All rights reserved.
   $Source$

   A decision stump.
*/

#ifndef __boosting__stump_h__
#define __boosting__stump_h__

#include "classifier.h"
#include "feature_set.h"
#include <boost/multi_array.hpp>
#include "jml/utils/enum_info.h"
#include "split.h"

namespace ML {


/*****************************************************************************/
/* ACTION                                                                    */
/*****************************************************************************/

/* What to do once we have made a split. */
class Action {
public:
    Action()
    {
    }

    Action(const Label_Dist & pred_true,
           const Label_Dist & pred_false,
           const Label_Dist & pred_missing)
        : pred_false(pred_false), pred_true(pred_true),
          pred_missing(pred_missing)
    {
    }
    
    void swap(Action & other)
    {
        pred_true.swap(other.pred_true);
        pred_false.swap(other.pred_false);
        pred_missing.swap(other.pred_missing);
    }
    
    /** Apply to a distribution, updating in place */
    void apply(Label_Dist & result, const Split::Weights & weights) const
    {
        if (weights[false])   result += weights[false]   * pred_false;
        if (weights[true])    result += weights[true]    * pred_true;
        if (weights[MISSING]) result += weights[MISSING] * pred_missing;
    }

    /** Apply to a single label */
    float apply(int label, const Split::Weights & weights) const
    {
        return weights[false]   * pred_false[label]
            +  weights[true]    * pred_true[label]
            +  weights[MISSING] * pred_missing[label];
    }

    /** Apply and return a distribution */
    Label_Dist apply(const Split::Weights & weights) const
    {
        Label_Dist result;
        apply(result, weights);
        return result;
    }

    void serialize(DB::Store_Writer & store) const;
    void reconstitute(DB::Store_Reader & store);

    Action & operator *= (float val)
    {
        pred_false   *= val;
        pred_true    *= val;
        pred_missing *= val;
        return *this;
    }

    void merge(const Action & other, float scale);

    std::string print() const;

    Label_Dist pred_false;    ///< predictions if p doesn't hold
    Label_Dist pred_true;     ///< predictions if p holds
    Label_Dist pred_missing;  ///< predictions if feature absent
};



/*****************************************************************************/
/* STUMP                                                                     */
/*****************************************************************************/

/** The most basic weak learner.  Learns a rule based upon a predicate:
    - Is the feature present (presence variables)
    - Is the feature true    (boolean variables)
    - Is var > value         (real variables)

    It can deal well with both missing and probabilistic features, and is
    quite efficient to calculate.
*/

class Stump : public Classifier_Impl {
public:
    /** Update algorithm for stump training. */
    enum Update {
        NORMAL = 0,   ///< Normal update
        GENTLE = 1,   ///< Gentle update
        PROB   = 2    ///< Probabilistic update
    };

    static Output_Encoding update_to_encoding(Update update_alg);

    /** Default construct.  Must be initialised before use. */
    Stump();

    /** Construct it by reconstituting it from a store. */
    Stump(DB::Store_Reader & store,
          const std::shared_ptr<const Feature_Space> & feature_space);
    
    /** Construct a rule for a predicate that can be true, false, or not
        have a value due to a missing feature. */
    Stump(const Feature & predicted,
          const Feature & feature,
          float arg,
          const Label_Dist & pred_true,
          const Label_Dist & pred_false,
          const Label_Dist & pred_missing,
          Update update,
          std::shared_ptr<const Feature_Space> feature_space,
          float Z = -INFINITY);
    
    /** Construct a rule for a there/not there classifier. */
    Stump(const Feature & predicted,
          const Feature & feature,
          float arg,
          const Label_Dist & pred_there,
          Update update,
          std::shared_ptr<const Feature_Space> feature_space,
          float Z = -INFINITY);

    /** Construct not filled in yet. */
    Stump(std::shared_ptr<const Feature_Space> feature_space,
          const Feature & predicted);

    /** Construct not filled in yet, when label may be unknown. */
    Stump(std::shared_ptr<const Feature_Space> feature_space,
          const Feature & predicted, size_t label_count);

    virtual ~Stump();


    /** Swap with another. */
    void swap(Stump & other);

    /** Merge with another.  The feature and the arg have to be the same for
        this to work.  An exception is thrown if not. */
    void merge(const Stump & other, float other_weight = 1.0);

    /** Scale the values of this stump.  Operates in-place. */
    void scale(float scale);

    /** Return a scaled version if the current stump.  The true, false and
        missing predictions are scaled by the given value. */
    Stump scaled(float scale) const;

    Split split;                       ///< How do we split?
    float Z;                           ///< Z score of the rule
    Action action;                     ///< What to do once split
    Output_Encoding encoding;          ///< What type of predictions?

    /** Predict the score for all classes. */
    Label_Dist predict(const Feature_Set & features,
                       PredictionContext * context = 0) const;

    using Classifier_Impl::predict;

    /** Predict, given the training data for a feature.  Note that the file
        stump_include.h needs to be included to use this method. */
    template<class FeatureExPtrPtr>
    void predict(Label_Dist & result,
                 FeatureExPtrPtr first, FeatureExPtrPtr last) const;

    /** Predict, given the training data for a feature.  Note that the file
        stump_include.h needs to be included to use this method. */
    template<class FeatureExPtrPtr>
    float predict(int label, FeatureExPtrPtr first, FeatureExPtrPtr last) const;

    /** Predict the score for a single class. */
    virtual float predict(int label, const Feature_Set & features,
                          PredictionContext * context = 0) const;

    virtual Output_Encoding output_encoding() const;

    virtual std::string print() const;

    virtual std::string summary() const;

    virtual std::vector<Feature> all_features() const;

    /** Serialization and reconstitution. */
    virtual void serialize(DB::Store_Writer & store) const;
    virtual void reconstitute(DB::Store_Reader & store,
                              const std::shared_ptr<const Feature_Space>
                                  & fs);

    virtual void serialize_lw(DB::Store_Writer & store) const;
    virtual void reconstitute_lw(DB::Store_Reader & store,
                                 const Feature & feature);

    virtual std::string class_id() const { return "STUMP"; }

    virtual Stump * make_copy() const;

    virtual Classifier_Impl *
    merge(const Classifier_Impl & other, float weight = 1.0) const;
};


} // namespace ML

DECLARE_ENUM_INFO(ML::Stump::Update, 3);

#endif /* __boosting__stump_h__ */
