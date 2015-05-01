/* json_holder.cc                                                  -*- C++ -*-
   Jeremy Barnes, 1 June 2012
   Copyright (c) 2012 Datacratic.  All rights reserved.

   Hold JSON in its natural format.
*/

#include "rtbkit/common/json_holder.h"
#include <boost/algorithm/string.hpp>
#include "jml/db/persistent.h"

using namespace std;
using namespace ML;


namespace RTBKIT {


void
JsonHolder::
serialize(ML::DB::Store_Writer & store) const
{
    unsigned char version = 0;
    makeString();
    store << version << str;
}

void
JsonHolder::
reconstitute(ML::DB::Store_Reader & store)
{
    unsigned char version;
    store >> version;
    if (version != 0)
        throw ML::Exception("invalid JsonHolder serialization version");
    string s;
    store >> s;
    *this = s;
}

void
JsonHolder::
makeString() const
{
    if (!str.empty()) return;
    if (!parsed) return;
    if (parsed->isNull()) return;
    str = parsed->toString();
    boost::trim(str);
}

void
JsonHolder::
makeJson() const
{
    if (parsed) return;
    if (str.empty())
        parsed.reset(new Json::Value());
    else parsed.reset(new Json::Value(Json::parse(str)));
}

const std::string JsonHolder::nullStr("null");

std::ostream & operator << (std::ostream & stream, const JsonHolder & json)
{
    return stream << json.toString();
}

} // namespace RTBKIT


namespace Datacratic {

struct StdUtf8StringDescription : public DefaultDescription<std::string>
{
    virtual void parseJsonTyped(std::string * val,
                                JsonParsingContext & context) const
    {
        *val = context.expectStringUtf8().rawString();
    }

    virtual void printJsonTyped(const std::string * val,
                                JsonPrintingContext & context) const
    {
        context.writeStringUtf8(Datacratic::UnicodeString(*val));
    }
};

DefaultDescription<RTBKIT::JsonHolder>::
DefaultDescription()
{
    addField("str", &RTBKIT::JsonHolder::str, "", new StdUtf8StringDescription);
}

} // namespace Datacratic
