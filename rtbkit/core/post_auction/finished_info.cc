/** finished_info.cc                                 -*- C++ -*-
    Rémi Attab, 18 Apr 2014
    Copyright (c) 2014 Datacratic.  All rights reserved.

    Implementation of finished info.

*/

#include "finished_info.h"

using namespace std;
using namespace ML;

namespace RTBKIT {

/*****************************************************************************/
/* FINISHED INFO                                                             */
/*****************************************************************************/

Json::Value
FinishedInfo::
bidToJson() const
{
    Json::Value result = bid.toJson();
    result["timestamp"] = bidTime.secondsSinceEpoch();
    return result;
}

Json::Value
FinishedInfo::
winToJson() const
{
    Json::Value result;
    if (!hasWin()) return result;

    result["timestamp"] = winTime.secondsSinceEpoch();
    result["reportedStatus"] = (reportedStatus == BS_WIN ? "WIN" : "LOSS");
    result["winPrice"] = winPrice.toJson();
    result["rawWinPrice"] = rawWinPrice.toJson();
    result["meta"] = winMeta;

    return result;
}

void
FinishedInfo::
addVisit(Date visitTime,
         const std::string & visitMeta,
         const SegmentList & channels)
{
    Visit visit;
    visit.visitTime = visitTime;
    visit.channels = channels;
    visit.meta = visitMeta;
    visits.push_back(visit);
}

Json::Value
FinishedInfo::
visitsToJson() const
{
    Json::Value result;
    for (unsigned i = 0;  i < visits.size();  ++i) {
        Json::Value & v = result[i];
        const Visit & visit = visits[i];
        v["timestamp"] = visit.visitTime.secondsSinceEpoch();
        v["meta"] = visit.meta;
        v["channels"] = visit.channels.toJson();
    }
    return result;
}

Json::Value
FinishedInfo::
toJson() const
{
    throw ML::Exception("FinishedInfo::toJson()");
    Json::Value result;
    return result;
}

void
FinishedInfo::Visit::
serialize(DB::Store_Writer & store) const
{
    unsigned char version = 1;
    store << version << visitTime << channels << meta;
}

void
FinishedInfo::Visit::
reconstitute(DB::Store_Reader & store)
{
    unsigned char version;
    store >> version;
    if (version != 1)
        throw ML::Exception("invalid version");
    store >> visitTime >> channels >> meta;
}

IMPL_SERIALIZE_RECONSTITUTE(FinishedInfo::Visit);

} // namepsace RTBKIT
