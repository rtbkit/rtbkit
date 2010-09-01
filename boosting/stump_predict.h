/* stump_predict.h                                                 -*- C++ -*-
   Jeremy Barnes, 20 February 2004
   Copyright (c) 2004 Jeremy Barnes.  All rights reserved.
   $Source$

   Inline prediction methods for the stump.
*/

#ifndef __boosting__stump_predict_h__
#define __boosting__stump_predict_h__

#include "stump.h"


namespace ML {

template<class FeatureExPtrIter>
void Stump::predict(Label_Dist & result,
                    FeatureExPtrIter first, FeatureExPtrIter last) const
{
    Split::Weights weights;
    split.apply(first, last, weights);
    action.apply(result, weights);
}

template<class FeatureExPtrIter>
float Stump::
predict(int label, FeatureExPtrIter first, FeatureExPtrIter last) const
{
    Split::Weights weights;
    split.apply(first, last, weights);
    return action.apply(label, weights);
}

} // namespace ML


#endif /* __boosting__stump_predict_h__ */
