/* account_key.cc
   Jeremy Barnes, 24 November 2012
   Copyright (c) 2012 Datacratic.  All rights reserved.

   Account key functions.
*/

#include "rtbkit/common/account_key.h"
#include "jml/db/persistent.h"

using namespace std;
using namespace ML;


namespace RTBKIT {

void validateSlug(const std::string & slug)
{
    if (slug.empty())
        throw ML::Exception("campaign/strategy slug cannot be empty");
    if (slug.size() > 256)
        throw ML::Exception("campaign/strategy slug has max length of 256");

    for (char c: slug)
        if (c != '_' && c != '.' && !isalnum(c))
            throw ML::Exception("campaign/strategy slug has invalid ASCII code %c/%i: %s", c, c, slug.c_str());
}


/*****************************************************************************/
/* ACCOUNT KEY                                                               */
/*****************************************************************************/

void
AccountKey::
serialize(ML::DB::Store_Writer & store) const
{
    store << (unsigned char)0;
    store.save(static_cast<AccountKeyBase>(*this));
}

void
AccountKey::
reconstitute(ML::DB::Store_Reader & store)
{
    unsigned char version = 0;
    store >> version;
    if (version != 0)
        throw ML::Exception("invalid AccountKey version");
    store.load(static_cast<AccountKeyBase &>(*this));
}

} // namespace RTBKIT
