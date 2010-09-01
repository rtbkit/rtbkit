/* weighted_training.h                                             -*- C++ -*-
   Jeremy Barnes, 16 February 2005
   Copyright (c) 2005 Jeremy Barnes.  All rights reserved.
   $Source$

   Functions to do weighted training.
*/

#ifndef __boosting__weighted_training_h__
#define __boosting__weighted_training_h__

#include "config.h"
#include "feature_set.h"
#include <map>
#include <vector>
#include <string>
#include "jml/stats/distribution.h"
#include <boost/multi_array.hpp>


namespace ML {


class Training_Data;
class Feature_Space;


/** This structure specifies the weights to apply to a dataset. */
struct Weight_Spec {

    /** The type of weight spec.  Can be either by bucket (in which case
        the frequency of the given value is used) or by value (in which
        case the actual value of the feature is used).
    */
    enum Type {
        BY_FREQUENCY,  
        BY_VALUE
    };

    Feature feature;                 ///< Feature to look for
    Type type;                       ///< Type of weight to use
    std::map<float, float> weights;  ///< Weight to give for each value
    float missing_weight;            ///< Weight to give when missing
    float beta;                      ///< Beta value
    bool group_feature;              ///< Whether or not it's a group feat

    Weight_Spec()
        : type(BY_FREQUENCY), missing_weight(0.0), beta(0.0),
          group_feature(false)
    {
    }
};

/** Generate a weights array for \c train_weighted(), possibly biased
    by label frequency.

    This function returns a weights array suitable to be passed as the
    third argument of \c train_weighted().

    It does so by assigning a weight to each example in the training
    data based upon the frequency of occurrence its labels in the
    training set.  The weight for the examples with label <i>l</i> is:
    \f[
      w_l = \left( \frac{1}{|x: \mbox{label}(x) = l|} \right)^{\beta} .
    \f]
    where the expression on the denominator simply counts the number of
    examples with the given label.

    The \f$\beta\f$ parameter as it controls how much more weight we
    give to rare examples.  At a negative value, we move towards
    ignoring infrequent labels (not likely what we want!).  At zero,
    we give no boost at all to any label; at one we boost them to the
    point that each label shares the weight evenly no matter how many
    infrequent, and greater than one we start to make infrequent labels
    more important than frequent ones (again, likely not what we want).
    Thus, normally \f$\beta\f$ will be between 0 and 1.

    \param data         the training data to generate the weights for.
    \param beta         the \f$\beta\f$ parameter in th formula above.
                        The default value of 0.0 will cause all of the
                        weights in the array to be uniform.
    \param feature      The feature upon which to equalize based on.
                        Each value of this feature will be equalized.
                        If it is equal to the Feature default constructor
                        (also the default argument of this parameter),
                        then the label will be used.

    \returns            a \c data.example_count() by
                        \c data.label_count() array of weights, with
                        examples weighted by the above formula.

    \post               the sum of all elements in weights is 1.0,
                        unless one of its dimensions is zero.
*/
Weight_Spec
get_weight_spec(const Training_Data & data,
                double beta,
                const Feature & feature,
                bool group_feature = false);

Weight_Spec
train_weight_spec(const Training_Data & data,
                  const Weight_Spec & spec);

std::vector<Weight_Spec>
get_weight_spec(const Training_Data & data,
                const std::vector<float> & betas,
                const std::vector<Feature> & features,
                const std::vector<bool> & group_features);

std::vector<Weight_Spec>
train_weight_spec(const Training_Data & data,
                  const std::vector<Weight_Spec> & untrained);

distribution<float>
apply_weight_spec(const Training_Data & data, const Weight_Spec & spec);

distribution<float>
apply_weight_spec(const Training_Data & data,
                  const std::vector<Weight_Spec> & specs);
    
boost::multi_array<float, 2>
expand_weights(const Training_Data & data,
               const distribution<float> & weights,
               const Feature & predicted);

/** Parse a weight spec.  Note that this method returns an untrained weight
    spec; you need to call train_weight_spec before using it. */
std::vector<Weight_Spec>
parse_weight_spec(const Feature_Space & fs,
                  std::string equalize_name, float beta,
                  std::string weight_spec, Feature group_feature);
    
} // namespace ML


#endif /* __boosting__weighted_training_h__ */
