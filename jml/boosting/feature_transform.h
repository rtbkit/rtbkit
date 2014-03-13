/* feature_transform.h                                             -*- C++ -*-
   Jeremy Barnes, 27 February 2006
   Copyright (c) 2006 Jeremy Barnes.  All rights reserved.
   $Source$

   Basic transform for a given feature.
*/

#ifndef __boosting__feature_transform_h__
#define __boosting__feature_transform_h__


#include "feature_set.h"


namespace ML {


/*****************************************************************************/
/* VARIABLE_LIST                                                             */
/*****************************************************************************/

/** Contains a list of variables that are transformed. */

typedef std::vector<Feature> Variable_List;


/*****************************************************************************/
/* FEATURE_TRANSFORM                                                         */
/*****************************************************************************/

/** Base class for an indivudual transform.  Each transform is applied to a
    class.
*/

class Feature_Transform {
public:
    virtual ~Feature_Transform();

    /** Apply the given transform to the features.  The features should be
        modified in place. */
    virtual void apply(Mutable_Feature_Set & features) const = 0;

    /** Train the given transform for the given data set.  Default does
        nothing.
    */
    virtual void train(const Training_Data & data) const;

    /** Apply the given transform to generate a full training data set.  The
        default calls apply() directly; however statistics based transforms
        may need to subract the current example from the statistics before
        providing training data.
    */
    virtual void apply_training(Mutable_Feature_Set & features) const;

    /** Returns the set of input features that this transform requires. */
    virtual Variable_List input() const = 0;
    
    /** Returns the set of features that are added.  Default returns none*/
    virtual Variable_List output() const = 0;

    /** Returns the list of input features that are removed.  Default
        returns none. */
    virtual Variable_List removed() const;

    /** Print an ASCII representation. */
    virtual std::string print() const;

    /** Serialize the transform. */
    virtual void serialize(DB::Store_Writer & store) const = 0;

    /** Reconstitute the transform. */
    virtual void reconstitute(DB::Store_Reader & store) const = 0;
};


} // namespace ML


#endif /* __boosting__feature_transform_h__ */
