/* boosted_stumps_impl.h                                           -*- C++ -*-
   Jeremy Barnes, 8 February 2004
   Copyright (c) 2004 Jeremy Barnes.  All rights reserved.
   $Source$

   Contains the implementation of the predict core, which is split off so
   that we can use it while tracing the operation of the algorithm.
*/

#ifndef __boosting__boosted_stumps_impl_h__
#define __boosting__boosted_stumps_impl_h__


#include "boosted_stumps.h"
#include "jml/utils/sgi_algorithm.h"
#include <iostream>


namespace ML {

/** This is the core of the predict algorithm.  It is parameterised by how
    it updates its results, which allows us to reuse the same code for both
    the single and multiple label prediction.
*/

template<class Results, class StumpIterator, class FeatureIterator>
StumpIterator
predict_feature_range(const Feature & feature,
                      StumpIterator stit, StumpIterator stit_end,
                      FeatureIterator fsit, FeatureIterator fsit_end,
                      const Results & results)
{
    /* Go through each of the stumps at this feature with each of the
       values. */
    while (stit != stit_end && stit->first.feature() == feature) {
        const Stump & stump = stit->second;
        assert(stump.split.feature() == feature);
        
        Split::Weights weights;
        stump.split.apply(fsit, fsit_end, weights);

        Label_Dist output = stump.action.apply(weights);
        results(output, 1.0, feature);
        ++stit;
    }

    return stit;
}

template<class Results>
void predict_even(const Feature_Set & features, const Results & results,
                  const Boosted_Stumps::stumps_type & stumps)
{
    typedef Boosted_Stumps::stumps_type stumps_type;

    /* Algorithm: we go through the features and the algorithms at the same
       time. */
    stumps_type::const_iterator stit = stumps.begin();
    Feature_Set::const_iterator fsit = features.begin();
        
    //cerr << "predict_core" << endl;
    while (stit != stumps.end()) {
        /* Find the number of times that we have the same feature. */
        const Feature & feature = stit->first.feature();
            
        /* Look in the feature set for this feature. */
        while (fsit != features.end() && (*fsit).first < feature)
            ++fsit;
            
        /* Get an iterator to the range of them, and find out how many. */
        Feature_Set::const_iterator fsit_end = fsit;
        while (fsit_end != features.end() && (*fsit_end).first == feature)
            ++fsit_end;
        
        stit = predict_feature_range(feature, stit, stumps.end(), fsit,
                                     fsit_end, results);
        
        fsit = fsit_end;
    }
}

#if 0
template<class Results>
void predict_many_stumps(const Feature_Set & features, const Results & results,
                         const Boosted_Stumps::stumps_type & stumps)
{
    typedef Boosted_Stumps::stumps_type stumps_type;
    //bool debug = false;

    /* Algorithm: we go through the features one at a time, looking up
       stumps, and adjust for the missing values. */

    Feature_Set::const_iterator fsit = features.begin(), fsit2;
    stumps_type::const_iterator stit = stumps.begin();

    while (fsit != features.end()) {
        fsit2 = fsit;
        const Feature & feature = (*fsit).first;
        while (fsit2 != features.end() && (*fsit2).first == feature)
            ++fsit2;

        /* Find the range of stumps for this feature. */
        std::pair<Feature, float> low(feature, -INFINITY);
        stumps_type::const_iterator st_first = stumps.lower_bound(low);
        
        stit = predict_feature_range(feature, st_first, stumps.end(),
                                     fsit, fsit2, results, true);
        
        fsit = fsit2;
    }
}
#endif

template<class Results>
void Boosted_Stumps::
predict_core(const Feature_Set & features, const Results & results) const
{
    using namespace std;

#if 0
    Feature_Set features = features_;
    features.sort();

    if (features != features_)
        throw Exception("Not sorted");
#endif

    predict_even(features, results, stumps);

    return;

#if 0
    bool debug = false;

    if (debug)
        std::cerr << "predict: features " << feature_space()->print(features)
             << std::endl;

    //cerr << "stumps.size() = " << stumps.size() << " features.size() = "
    //     << features.size() << endl;

    if (features.size() > 10 * stumps.size()) {
        /* Way more features in the feature set.  We just search for them. */
        predict_even(features, results, stumps);
    }
    else if (stumps.size() > 10 * features.size()) {
        /* Way more stumps than features.  We use the missing total as a
           starting point, and just look up the differences in the ones
           that are not missing. */
        results(sum_missing, 1.0, MISSING_FEATURE);
        predict_many_stumps(features, results, stumps);
    }
    else {
        /* About evenly matched.  We co-iterate through the two arrays at
           the same time. */
        predict_even(features, results, stumps);
    }
#endif
}

} // namespace ML



#endif /* __boosting__boosted_stumps_impl_h__ */
