/* transform_list.cc
   Jeremy Barnes, 27 February 2006
   Copyright (c) 2006 Jeremy Barnes.  All rights reserved.
   $Source$

   Implementation of the Transform_List class.
*/

#include "transform_list.h"
#include "registry.h"


namespace ML {


/*****************************************************************************/
/* TRANSFORM_LIST                                                            */
/*****************************************************************************/

Transform_List::
Transform_List()
{
}

Transform_List::
Transform_List(DB::Store_Reader & store)
{
    reconstitute(store);
}

Transform_List::
Transform_List(std::shared_ptr<Feature_Space> fs)
    : feature_space_(fs)
{
}

Transform_List::
~Transform_List()
{
}

std::shared_ptr<Feature_Space>
Transform_List::
input_fs() const
{
    return feature_space_;
}

std::shared_ptr<Feature_Space>
Transform_List::
output_fs() const
{
    return feature_space_;
}

std::shared_ptr<Mutable_Feature_Set>
Transform_List::
transform(const Feature_Set & features) const
{
    throw Exception("Transform_List::transform(): not implemented");
}

std::string
Transform_List::
class_id() const
{
    return "TRANSFORM_LIST";
}

Transform_List *
Transform_List::
make_copy() const
{
    return new Transform_List(*this);
}

std::vector<Feature>
Transform_List::
features_for(const std::vector<Feature> & features) const
{
    throw Exception("Transform_List::features_for(): not implemented");
}

void
Transform_List::
serialize(DB::Store_Writer & store) const
{
    throw Exception("Transform_List::serialize(): not implemented");
}

void
Transform_List::
reconstitute(DB::Store_Reader & store)
{
    throw Exception("Transform_List::reconstitute(): not implemented");
}

void
Transform_List::
parse(const std::vector<std::string> & transforms)
{
    for (unsigned i = 0;  i < transforms.size();  ++i) {
        parse(transforms[i]);
    }
}

void
Transform_List::
parse(const std::string & transform)
{
    

    throw Exception("Transform_List::parse(): not implemented");
}


/*****************************************************************************/
/* REGISTRATION                                                              */
/*****************************************************************************/

namespace {

Register_Factory<Feature_Transformer_Impl, Transform_List>
    TL_REG("TRANSFORM_LIST");

} // file scope


} // namespace ML
