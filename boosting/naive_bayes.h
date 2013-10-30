/* naive_bayes.h                                                   -*- C++ -*-
   Jeremy Barnes, 6 June 2003
   Copyright (c) 2003 Jeremy Barnes.  All rights reserved.
   $Source$

   Naïve Bayes classifier.
*/

#ifndef __boosting__naive_bayes_h__
#define __boosting__naive_bayes_h__

#include "classifier.h"
#include "feature_set.h"



namespace ML {


class Training_Data;


/*****************************************************************************/
/* NAIVE_BAYES                                                               */
/*****************************************************************************/

/** A weak learner, based upon Bayesian theory.  Given a label <i>l</i> and
    a feature <i>f</i>, it uses Bayes' formula to transform the output into
    an input:

    \f[
        P(l|f) = \frac{P(f|l) P(l)}{P(f)} .
    \f]
    
    It then makes the crucial "naive Bayes assumption" that given a label,
    each of the features occurs independently of all of the others:

    \f[
        P(l|f_1, f_2, \ldots, f_n) = \prod_{i=1}^{n} P(l|f) .
    \f]

    This yields the final formula

    \f[
        P(l|f_1, f_2, \ldots f_n) = P(l) \prod_{i=1}^{n} \frac{P(l|f)}{P(f)} .
    \f]

    The probabilities are learnt from a set of training data.

    The features <i>f<\i> are binary predicates (\f$f_i=p_i(\mathbf{x})\f$
    where <b>x<\b> represents the input data.  They can be encoded in 2
    ways, depending upon the application.  We can either learn separately
    P(l|\f$p_i(\mathbf{x})\f$ holds) and P(l|\f$p_i(\mathbf{x})\f$ holds),
    or we can ignore the cases where it doesn't hold at all (the "true only"
    case).
*/

class Naive_Bayes : public Classifier_Impl {
public:
    /** Default construct.  Must be initialised before use. */
    Naive_Bayes();

    /** Construct it by reconstituting it from a store. */
    Naive_Bayes(DB::Store_Reader & store,
                const std::shared_ptr<const Feature_Space> & feature_space);
    
    /** Construct not filled in yet. */
    Naive_Bayes(std::shared_ptr<const Feature_Space> feature_space,
                const Feature & predicted);

    virtual ~Naive_Bayes();


    void swap(Naive_Bayes & other);

    /** This structure holds information about a feature used by Naive Bayes
        to turn a (feature, value) pair into a predicate.
    */
    struct Bayes_Feature {
        Bayes_Feature(const Feature & feature = Feature(), float arg = 0)
            : feature(feature), arg(arg)
        {
        }

        Feature feature;    ///< Feature to split on
        float arg;          ///< Value to split on

        bool operator < (const Bayes_Feature & other) const
        {
            if (feature < other.feature) return true;
            else if (feature == other.feature && arg < other.arg) return true;
            else return false;
        }
    };

    std::vector<Bayes_Feature> features;
    boost::multi_array<float, 3> probs;  /* num features x 3 x num labels matrix */
    distribution<float> label_priors;
    distribution<float> missing_total; /* sum of all missing distributions. */

    using Classifier_Impl::predict;

    virtual float predict(int label, const Feature_Set & features,
                          PredictionContext * context = 0) const;

    virtual distribution<float>
    predict(const Feature_Set & features,
            PredictionContext * context = 0) const;

    virtual std::string print() const;

    virtual std::vector<Feature> all_features() const;

    virtual Output_Encoding output_encoding() const;

    virtual void serialize(DB::Store_Writer & store) const;
    virtual void reconstitute(DB::Store_Reader & store,
                              const std::shared_ptr<const Feature_Space>
                                  & feature_space);
    
    virtual std::string class_id() const;

    virtual Naive_Bayes * make_copy() const;

    void calc_missing_total();
};


} // namespace ML



#endif /* __boosting__stump_h__ */
