/* transform_list.h                                                -*- C++ -*-
   Jeremy Barnes, 27 February 2006
   Copyright (c) 2006 Jeremy Barnes.  All rights reserved.
   $Source$

   A Feature_Transform that applies a list of Feature_Transforms in order to
   arrive at a final feature set.
*/

#ifndef __boosting__transform_list_h__
#define __boosting__transform_list_h__

#include "feature_transformer.h"
#include "feature_space.h"
#include "feature_transform.h"

namespace ML {


/*****************************************************************************/
/* TRANSFORM_LIST                                                            */
/*****************************************************************************/

/** Object that will transform one feature set into another. */

class Transform_List : public Feature_Transformer_Impl {
public:
    Transform_List();
    Transform_List(DB::Store_Reader & store);
    Transform_List(std::shared_ptr<Feature_Space> fs);

    virtual ~Transform_List();
    
    /** Initialize with the given feature space. */
    void init(std::shared_ptr<Feature_Space> feature_space);

    /** Return the feature space that our features are input into. */
    virtual std::shared_ptr<Feature_Space> input_fs() const;

    /** Return the feature space that our features are output to. */
    virtual std::shared_ptr<Feature_Space> output_fs() const;

    /** Transform the given feature set.  Note that the features will be
        implicitly transformed into the output_fs feature space. */
    virtual std::shared_ptr<Mutable_Feature_Set>
    transform(const Feature_Set & features) const;

    virtual std::string class_id() const;

    virtual Transform_List * make_copy() const;

    virtual std::vector<Feature>
    features_for(const std::vector<Feature> & features) const;

    virtual void serialize(DB::Store_Writer & store) const;

    virtual void reconstitute(DB::Store_Reader & store);

    /** Parse a series of strings that represent transforms from a list.
        These strings can also be of the form "@file" which will cause
        transforms to be read from the given file.
    */
    void parse(const std::vector<std::string> & transforms);

    /** Parse a single string representing a tranform and add it to the
        current list. */
    void parse(const std::string & transform);

private:
    std::vector<std::shared_ptr<Feature_Transform> > transforms_;
    std::shared_ptr<Feature_Space> feature_space_;
};


} // namespace ML

#endif /* __boosting__transform_list_h__ */
