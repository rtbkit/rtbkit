/* null_feature_space.cc
   Jeremy Barnes, 21 July 2003
   Copyright (c) 2003 Jeremy Barnes.  All rights reserved.
   $Source$

   Implementation of stubs for the Null_Feature_Space.
*/

#include "null_feature_space.h"
#include "registry.h"

using namespace std;


namespace ML {


/*****************************************************************************/
/* NULL_FEATURE_SPACE                                                        */
/*****************************************************************************/

Null_Feature_Space::Null_Feature_Space()
{
}

Null_Feature_Space::Null_Feature_Space(DB::Store_Reader & store)
{
    reconstitute(store);
}

Null_Feature_Space::~Null_Feature_Space()
{
}

Feature_Info Null_Feature_Space::info(const Feature & feature) const
{
    return Feature_Info();
}

std::string Null_Feature_Space::print(const Feature & feature) const
{
    return "";
}

bool Null_Feature_Space::
parse(Parse_Context & context, Feature & feature) const
{
    throw Exception("Null_Feature_Space::parse() not implemented");
}

void Null_Feature_Space::
expect(Parse_Context & context, Feature & feature) const
{
    throw Exception("Null_Feature_Space::expect() not implemented");
}

void Null_Feature_Space::
serialize(DB::Store_Writer & store, const Feature & feature) const
{
    //store << feature;
}

void Null_Feature_Space::
reconstitute(DB::Store_Reader & store, Feature & feature) const
{
    //store >> feature;
}

std::string Null_Feature_Space::
print(const Feature_Set & fs) const
{
    return "";
}

void Null_Feature_Space::
serialize(DB::Store_Writer & store, const Feature_Set & fs) const
{
}

void Null_Feature_Space::
reconstitute(DB::Store_Reader & store, Feature_Set & fs) const
{
}

std::string Null_Feature_Space::class_id() const
{
    return "NULL_FS";
}

void Null_Feature_Space::serialize(DB::Store_Writer & store) const
{
    /* Don't write anything. */
}

void Null_Feature_Space::
reconstitute(DB::Store_Reader & store,
             const std::shared_ptr<const Feature_Space> & feature_space)
{
    /* Don't read anything. */
}

void Null_Feature_Space::
reconstitute(DB::Store_Reader & store)
{
}

Null_Feature_Space * Null_Feature_Space::make_copy() const
{
    return new Null_Feature_Space(*this);
}

std::string Null_Feature_Space::print() const
{
    return "";
}

/*****************************************************************************/
/* REGISTRATION                                                              */
/*****************************************************************************/

Register_Factory<Feature_Space, Null_Feature_Space> NULL_REG("NULL_FS");


} // namespace ML

