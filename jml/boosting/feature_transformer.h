/* feature_transformer.h                                          -*- C++ -*-
   Jeremy Barnes, 10 February 2005
   Copyright (c) 2005 Jeremy Barnes.  All rights reserved.
   $Source$

   A component that transforms features.
*/

#include "feature_set.h"
#include "jml/utils/unnamed_bool.h"
#include "jml/utils/smart_ptr_utils.h"


namespace ML {

class Feature_Space;


/*****************************************************************************/
/* FEATURE_TRANSFORMER_IMPL                                                  */
/*****************************************************************************/

/** Object that will transform one feature set into another. */

class Feature_Transformer_Impl {
public:
    virtual ~Feature_Transformer_Impl();
    
    /** Return the feature space that our features are input into. */
    virtual std::shared_ptr<Feature_Space> input_fs() const = 0;

    /** Return the feature space that our features are output to. */
    virtual std::shared_ptr<Feature_Space> output_fs() const = 0;

    /** Transform the given feature set.  Note that the features will be
        implicitly transformed into the output_fs feature space. */
    virtual std::shared_ptr<Mutable_Feature_Set>
    transform(const Feature_Set & features) const = 0;

    /** Tell us which input features are required in order to generate
        the given output features. */
    virtual std::vector<Feature>
    features_for(const std::vector<Feature> & features) const = 0;

    virtual void serialize(DB::Store_Writer & store) const = 0;

    virtual void reconstitute(DB::Store_Reader & store) = 0;

    virtual std::string class_id() const = 0;

    virtual Feature_Transformer_Impl * make_copy() const = 0;

    void poly_serialize(DB::Store_Writer & store);

    static std::shared_ptr<Feature_Transformer_Impl>
    poly_reconstitute(DB::Store_Reader & store);
};


/*****************************************************************************/
/* FEATURE_TRANSFORMER                                                       */
/*****************************************************************************/

/** Object that transforms features. */

class Feature_Transformer {
public:
    Feature_Transformer();
    Feature_Transformer(DB::Store_Reader & store);
    Feature_Transformer(std::shared_ptr<Feature_Transformer_Impl> impl);
    Feature_Transformer(const Feature_Transformer & other)
    {
        if (other.impl_)
            impl_.reset(other.impl_->make_copy());
    }

    Feature_Transformer & operator = (const Feature_Transformer & other)
    {
        Feature_Transformer new_me(other);
        swap(new_me);
        return *this;
    }

    void init(std::shared_ptr<Feature_Transformer_Impl> impl)
    {
        impl_ = impl;
    }

    std::shared_ptr<Feature_Space> input_fs() const
    {
        if (!impl_) return std::shared_ptr<Feature_Space>();
        else return impl_->input_fs();
    }

    std::shared_ptr<Feature_Space> output_fs() const
    {
        if (!impl_) return std::shared_ptr<Feature_Space>();
        else return impl_->output_fs();
    }

    void serialize(DB::Store_Writer & store) const;
    void reconstitute(DB::Store_Reader & store);
    
    void swap(Feature_Transformer & other)
    {
        impl_.swap(other.impl_);
    }

    std::shared_ptr<Mutable_Feature_Set>
    transform(const Feature_Set & features) const
    {
        if (impl_)
            return impl_->transform(features);
        else {
            return make_sp
                (new Mutable_Feature_Set(features.begin(), features.end()));
        }
    }
    
    std::vector<Feature>
    features_for(const std::vector<Feature> & features) const
    {
        if (impl_)
            return impl_->features_for(features);
        else return features;
    }

    /** Operator bool, which tells us if there is any implementation or not. */
    JML_IMPLEMENT_OPERATOR_BOOL(!!impl_);
private:
    std::shared_ptr<Feature_Transformer_Impl> impl_;
};

DB::Store_Writer &
operator << (DB::Store_Writer & store, const Feature_Transformer & tr);

DB::Store_Reader &
operator >> (DB::Store_Reader & store, Feature_Transformer & tr);


} // namespace ML
