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
                   const boost::shared_ptr<const Feature_Space>
                       & feature_space);
    
    Committee(const boost::shared_ptr<const Feature_Space>
                       & feature_space,
                   const Feature & predicted);

    std::vector<boost::shared_ptr<Classifier_Impl> > classifiers;
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

    void add(boost::shared_ptr<Classifier_Impl> classifier, float weight = 1.0);
    
    using Classifier_Impl::predict;

    /** Predict the score for a single class. */
    virtual float predict(int label, const Feature_Set & features) const;

    /** Predict all classes. */
    virtual distribution<float> predict(const Feature_Set & features) const;

    virtual std::string print() const;

    virtual std::vector<Feature> all_features() const;

    virtual Output_Encoding output_encoding() const;

    /** Serialization and reconstitution. */
    virtual void serialize(DB::Store_Writer & store) const;
    virtual void
    reconstitute(DB::Store_Reader & store,
                 const boost::shared_ptr<const Feature_Space> & features);

    virtual std::string class_id() const { return "COMMITTEE"; }

    virtual Committee * make_copy() const;
};

} // namespace ML

#endif /* __boosting__committee_h__ */
