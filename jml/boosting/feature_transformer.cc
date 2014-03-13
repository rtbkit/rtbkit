/* feature_transformer.cc
   Jeremy Barnes, 27 February 2006
   Copyright (c) 2006 Jeremy Barnes.  All rights reserved.
   $Source$

   Implementation of the base Feature_Transformer methods.
*/

#include "feature_transformer.h"
#include "jml/db/persistent.h"
#include "config_impl.h"
#include "registry.h"


using namespace std;
using namespace DB;

namespace ML {


/*****************************************************************************/
/* FEATURE_TRANSFORMER_IMPL                                                  */
/*****************************************************************************/

Feature_Transformer_Impl::
~Feature_Transformer_Impl()
{
}

void
Feature_Transformer_Impl::
poly_serialize(DB::Store_Writer & store)
{
    Registry<Feature_Transformer_Impl>::singleton().serialize(store, this);
}

std::shared_ptr<Feature_Transformer_Impl>
Feature_Transformer_Impl::
poly_reconstitute(DB::Store_Reader & store)
{
    return Registry<Feature_Transformer_Impl>::singleton().reconstitute(store);
}


/*****************************************************************************/
/* FEATURE_TRANSFORMER                                                       */
/*****************************************************************************/

Feature_Transformer::Feature_Transformer()
{
}

Feature_Transformer::
Feature_Transformer(DB::Store_Reader & store)
{
    reconstitute(store);
}

namespace {

const std::string FT_MAGIC = "FEATURE_TRANSFORMER";
const std::string FT_CANARY = "END FEATURE_TRANSFORMER";

} // file scope

void
Feature_Transformer::
serialize(DB::Store_Writer & store) const
{
    store << FT_MAGIC << compact_size_t(0);
    if (impl_) {
        store << compact_size_t(1);
        impl_->poly_serialize(store);
    }
    else store << compact_size_t(0);

    store << FT_CANARY;
}

void
Feature_Transformer::
reconstitute(DB::Store_Reader & store)
{
    string id;
    store >> id;

    if (id != FT_MAGIC)
        throw Exception("Feature_Transformer::reconstitute(): invalid magic \""
                        + id + "\"");

    compact_size_t version(store);
    if (version != 0)
        throw Exception("Feature_Transformer::reconstitute(): invalid version");

    compact_size_t has_impl(store);

    if (has_impl) impl_ = Feature_Transformer_Impl::poly_reconstitute(store);
    else impl_.reset();

    store >> id;
    if (id != FT_CANARY)
        throw Exception("Feature_Transformer::reconstitute(): invalid canary \""
                        + id + "\"");
}
    
DB::Store_Writer &
operator << (DB::Store_Writer & store, const Feature_Transformer & tr)
{
    tr.serialize(store);
    return store;
}

DB::Store_Reader &
operator >> (DB::Store_Reader & store, Feature_Transformer & tr)
{
    tr.reconstitute(store);
    return store;
}

} // namespace ML
