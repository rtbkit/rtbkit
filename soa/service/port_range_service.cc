/** port_range_service.cc                                 -*- C++ -*-
    Rémi Attab, 12 Mar 2013
    Copyright (c) 2013 Datacratic.  All rights reserved.

    Port range allocation service.

*/

#include "port_range_service.h"
#include "soa/jsoncpp/value.h"
#include "soa/jsoncpp/reader.h"
#include "jml/utils/exc_assert.h"

#include <fstream>
#include <cstring>

using namespace std;
using namespace ML;

namespace Datacratic {


/******************************************************************************/
/* PORT RANGE                                                                 */
/******************************************************************************/

Json::Value
PortRange::
toJson() const
{
    if (last == first + 1)
        return first;

    Json::Value result;
    result[0] = "range";
    result[1] = first;
    result[2] = last;
    return result;
}

PortRange
PortRange::
fromJson(const Json::Value & val)
{
    if (val.isNull())
        return PortRange();
    else if (val.isNumeric())
        return val.asInt();
    else if (val.isArray()) {
        string type = val[0].asString();
        if (type == "range") {
            int first = val[1].asInt();
            int last = val[2].asInt();
            return PortRange(first, last);
        }
    }
    throw ML::Exception("unknown port range " + val.toString());
}


/******************************************************************************/
/* UTILS                                                                      */
/******************************************************************************/

namespace {

void dumpRangeMap(ostream& stream, const map<string, PortRange> rangeMap)
{
    for (const auto& entry : rangeMap) {
        stream << entry.first << ": ";

        if (entry.second.last - entry.second.first > 1) {
            stream << "["
                << entry.second.first << ", " << entry.second.last
                << "]";
        }
        else stream << entry.second.first;

        stream << endl;
    }
}

} // namespace anonymous

/******************************************************************************/
/* DEFAULT PORT RANGE SERVICE                                                 */
/******************************************************************************/

DefaultPortRangeService::
DefaultPortRangeService(unsigned rangeSize) :
    rangeSize(rangeSize), currentPort(15000)
{}

PortRange
DefaultPortRangeService::
getRange(const string& name)
{
    lock_guard<mutex> guard(lock);

    PortRange newRange(currentPort, currentPort + rangeSize);
    auto ret = rangeMap.insert(make_pair(name, newRange));

    if (ret.second) currentPort += rangeSize;

    return ret.first->second;
}

void
DefaultPortRangeService::
dump(ostream& stream) const
{
    lock_guard<mutex> guard(lock);

    dumpRangeMap(stream, rangeMap);
}


/******************************************************************************/
/* JSON PORT RANGE SERVICE                                                    */
/******************************************************************************/

JsonPortRangeService::
JsonPortRangeService(const Json::Value& json)
{
    vector<string> members = json.getMemberNames();

    for (size_t i = 0; i < members.size(); ++i) {
        const Json::Value& entry = json[members[i]];

        PortRange newRange;

        if (entry.isArray()) {
            newRange.first = entry[0].asInt();
            newRange.last = entry[1].asInt();
        }
        else if (entry.isInt()) {
            newRange = entry.asInt();
        }
        else ExcCheck(false, "Invalid entry type.");

        auto ret = rangeMap.insert(make_pair(members[i], newRange));

        ExcAssert(ret.second);
    }
}

PortRange
JsonPortRangeService::
getRange(const string& name)
{
    auto it = rangeMap.find(name);
    ExcCheck(it != rangeMap.end(), "No port range specified for " + name);

    return it->second;
}

void
JsonPortRangeService::
dump(ostream& stream) const
{
    dumpRangeMap(stream, rangeMap);
}


} // namepsace Datacratic
