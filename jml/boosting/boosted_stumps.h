/* boosted_stumps.h                                                -*- C++ -*-
   Jeremy Barnes, 6 June 2003
   Copyright (c) 2003 Jeremy Barnes.  All rights reserved.
   $Source$

   A boosted decision stumps algorithm.
*/

#ifndef __boosting__boosted_stumps_h__
#define __boosting__boosted_stumps_h__

#include "classifier.h"
#include "stump.h"
#include "jml/utils/enum_info.h"
#include "jml/utils/floating_point.h"
#include "config.h"
#include <boost/iterator/transform_iterator.hpp>
#include "jml/utils/sgi_numeric.h"

namespace ML {


/*****************************************************************************/
/* BOOSTED_STUMPS                                                            */
/*****************************************************************************/

/** The boosted decision stumps classifier.  It has a whole bunch of decision
    stumps (decision trees with only one decision), which it combines together
    linearly (with weights based upon the boosting algorithm) in order to
    determine the best output.
*/

class Boosted_Stumps : public Classifier_Impl {
public:
    Boosted_Stumps();

    Boosted_Stumps(DB::Store_Reader & store,
                   const std::shared_ptr<const Feature_Space>
                       & feature_space);
    
    Boosted_Stumps(const std::shared_ptr<const Feature_Space>
                       & feature_space,
                   const Feature & predicted);

    /** This structure contains all of the stumps.  It is designed to allow
        for quick lookup.

        They are arranged firstly by feature, and then by the value of that
        feature.  In this way, it is efficient to match up a list of features
        with the stumps they contain.

        Stumps which are similar will be automatically combined.
    */
    typedef std::map<Split, Stump> stumps_type;
    stumps_type stumps;

    distribution<float> bias;   ///< Bias to add to each label

    distribution<float> sum_missing;  ///< Total of all missing values

    /** This controls the transformation done on the output.  Using the
        logistic function will transform the output into something
        resembling probabilities.
    */
    enum Output {
        RAW,       ///< take un-transformed outputs
        LOGIT,     ///< run the logistic function on the output
        LOGIT_NORM ///< run the logistic function, then normalize
    };

    /** How we transform the output.  Default is no transform (raw output). */
    Output output;

    /** Iterator types.  These are just like a normal one would be. */
    typedef boost::transform_iterator<std::select2nd<stumps_type::value_type>,
                                      stumps_type::iterator>
        iterator;
    typedef boost::transform_iterator<std::select2nd<stumps_type::value_type>,
                                      stumps_type::const_iterator>
        const_iterator;

    /* Iterators.  These do what you would expect them to... */
    iterator begin() { return iterator(stumps.begin()); }
    iterator end() { return iterator(stumps.end()); }
    const_iterator begin() const { return const_iterator(stumps.begin()); }
    const_iterator end() const { return const_iterator(stumps.end()); }

    /** Insert the given stump into the appropriate place in the stumps
        map.  Returns an iterator to it. */
    iterator insert(const Stump & stump, float weight = 1.0);

    /** Insert the given stumps into the appropriate places in the stumps
        map, scaling each by 1/stumps.size(). */
    void insert(const std::vector<Stump> & stumps);

    /** Insert the given stumps into the appropriate places in the stumps
        map, scaling by the given distribution */
    void insert(const std::vector<Stump> & stumps,
                const distribution<float> & scale);

    /** Find the stump for a given feature and value. */
    iterator find(const Split & split)
    {
        return iterator(stumps.find(split));
    }

    /** Find the stump for a given feature and value. */
    const_iterator find(const Split & split) const
    {
        return const_iterator(stumps.find(split));
    }
    
    /** Swap two Boosted_Stumps objects.  Guaranteed not to throw an
        exception. */
    void swap(Boosted_Stumps & other)
    {
        Classifier_Impl::swap(other);
        stumps.swap(other.stumps);
        std::swap(output, other.output);
        bias.swap(other.bias);
        sum_missing.swap(other.sum_missing);
        std::swap(predicted_, other.predicted_);
    }

    using Classifier_Impl::predict;

    /** Predict the score for a single class. */
    virtual float predict(int label, const Feature_Set & features,
                          PredictionContext * context = 0) const;

    /** Predict the score for all classes. */
    virtual distribution<float>
    predict(const Feature_Set & features,
            PredictionContext * context = 0) const;

    /** This is the core of the predict algorithm.  It is parameterised by how
        it updates its results, which allows us to reuse the same code for both
        the single and multiple label prediction.

        \param features     The feature set to predict.
        \param results      The object to update when we have weight for a
                            category.  It will be called once per feature.
        \param Results      The type of the object in \p results.  See below.

        The results object must have a method which implements the
        signature
        
        \code
        void operator () (const distribution<float> & dist, float weight,
                          const Feature & feature) const;
        \endcode

        The effect of this is that the weights in \p dist should be multiplied
        by \p weight and accumulated to the totals (dist will have one entry
        per label).  The \p feature parameter is for information only---it
        tells which feature this weight is associated with, and can be
        used to trace the effect of various features on the output.

        Note that the header file boosted_stumps_impl.h needs to be included
        in order to use this functionality.
    */
    template<class Results>
    void predict_core(const Feature_Set & features, const Results & results)
        const;

    /** Calculate the accuracy.  This can be done much quicker with the
        boosted stumps as it only needs to look at the index for the features
        that it has learned a stump for, and these are nicely indexed
        by the training data. */
    virtual std::pair<float, float>
    accuracy(const Training_Data & data,
             const distribution<float> & example_weights
                 = UNIFORM_WEIGHTS,
             const Optimization_Info * opt_info = 0) const;

    using Classifier_Impl::accuracy;

    /** Calculate the sum of the missing scores. */
    void calc_sum_missing();

    /** Combine two boosted stumps objects together.  This one merges the
        other one with this one, to make one big classifier.  This is
        essentially done by adding the rules for the other one to the rules
        for this one, and then compacting it.

        \param with         Boosted_Stumps object to combine with.  The rules
                            will be added to this stump.
        \param weight       Weight to combine with.  It will be scaled by this
                            amount.
    */
    void combine(const Boosted_Stumps & with, float weight = 1.0);

    /** Combine a whole stack of boosted stumps objects together.  This
        takes a series of pointers to boosted stumps objects and a weights
        object, and combines them together.

        \code
        // Break an enormous dataset into 10 parts and train each individually,
        // then combine them to get a final answer.

        Training_Data huge_data(...);

        distribution<float> weights(10, 0.1);
        bool randomize = true;
        vector<std::shared_ptr<Training_Data> > chunks
            = huge_data.partition(weights, randomize);

        vector<Boosted_Stumps> stumps(10);
        for (unsigned i = 0;  i < 10;  ++i)
            stumps[i].train(*chunks[i], ...);

        vector<Boosted_Stumps *> ptrs(10);
        for (unsigned i = 0;  i < 10;  ++i) ptrs[i] = &stumps[i];

        Boosted_Stumps combined
            = Boosted_Stumps::combine(ptrs, weights);
        \endcode
    */
    template<class StumpPtrVec>
    static Boosted_Stumps combine(const StumpPtrVec stumps,
                                  const distribution<float> & weights)
    {
        if (stumps.empty()) return Boosted_Stumps();
        Boosted_Stumps result(stumps[0]->feature_space(),
                              stumps[0]->label_count());
        for (unsigned i = 0;  i < stumps.size();  ++i) {
            for (const_iterator jt = stumps[i]->begin();
                 jt != stumps[i]->end();  ++jt) {
                Stump scaled = *jt;
                scaled.action *= weights[i];
                result.insert(scaled);
            }
        }
        return result;
    }
                                
    virtual std::string print() const;

    virtual std::vector<Feature> all_features() const;

    virtual Output_Encoding output_encoding() const;

    /** Serialization and reconstitution. */
    virtual void serialize(DB::Store_Writer & store) const;
    virtual void
    reconstitute(DB::Store_Reader & store,
                 const std::shared_ptr<const Feature_Space> & features);

    virtual std::string class_id() const { return "BOOSTED_STUMPS"; }

    virtual Boosted_Stumps * make_copy() const;

    virtual bool merge_into(const Classifier_Impl & other, float weight = 1.0);

    virtual Classifier_Impl *
    merge(const Classifier_Impl & other, float weight = 1.0) const;
    
private:
    /** For reconstituting old classifiers only */
    Boosted_Stumps(const std::shared_ptr<const Feature_Space>
                       & feature_space,
                   const Feature & predicted,
                   size_t label_count);

};


} // namespace ML

DECLARE_ENUM_INFO(ML::Boosted_Stumps::Output, 3);


#endif /* __boosting__boosted_stumps_h__ */

