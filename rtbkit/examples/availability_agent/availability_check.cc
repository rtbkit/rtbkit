/** availability_check.cc                                 -*- C++ -*-
    Rémi Attab, 26 Jul 2012
    Copyright (c) 2012 Recoset.  All rights reserved.

    Datastore implementation for the availability check stuff.

*/

#include "availability_check.h"
#include "jml/arch/timers.h"
#include "jml/utils/guard.h"
#include "rtbkit/core/router/router_types.h"

using namespace std;
using namespace ML;
using namespace RTBKIT;

namespace Datacratic {

/******************************************************************************/
/* AVAILABILITY CHECK                                                         */
/******************************************************************************/

AvailabilityCheck::
AvailabilityCheck(size_t size) :
    size(0), pos(0), requests(size)
{
    pos = 0;
    for (size_t i = 0; i < requests.size(); ++i)
        requests[i] = NULL;

    filters.initWithDefaultFilters();
}


void
AvailabilityCheck::
addRequest(const BidRequest& newRequest)
{
    ML::Timer tm;

    BidRequest* oldRequest = requests[pos];
    requests[pos] = new BidRequest(newRequest);
    pos = (pos + 1) % requests.size();

    if (oldRequest)
        gcLock.defer([=]{ delete oldRequest; });

    if (size < requests.size()) size++;
    if (onEvent) {
        onEvent("newRequests", ET_COUNT, 1);
        onEvent("ringSize", ET_LEVEL, size);

        uint64_t elapsedMicros = tm.elapsed_wall() * 1000000.0;
        onEvent("addRequestMicros", ET_OUTCOME, elapsedMicros);
    }
}

Json::Value
AvailabilityCheck::
checkConfig(const AgentConfig& config)
{
    ML::Timer tm;

    AgentInfo info;
    info.config = std::make_shared<AgentConfig>(config);
    info.stats = std::make_shared<AgentStats>();
    const auto& stats = *info.stats;

    filters.addConfig("config", info);
    ML::Call_Guard guard([&] { filters.removeConfig("config"); });

    uint64_t requestCount = 0;
    uint64_t biddableCount = 0;

    {
        GcLock::SharedGuard guard(gcLock);

        for (size_t i = 0; i < requests.size(); ++i) {
            BidRequest* br = requests[i];
            if (br == NULL) continue;

            requestCount++;

            FilterPool::ConfigList result = filters.filter(*br, nullptr);
            if (!result.empty()) biddableCount++;
        }
    }

    auto makeFilter = [](const string& name, uint64_t value) {
        Json::Value val(Json::arrayValue);
        val.append(name);
        val.append(value);
        return val;
    };

    // These are in the same order as the filters in isBiddableRequest()
    Json::Value filters(Json::arrayValue);
    filters.append(makeFilter("noSpot", stats.noSpots));
#if 0
    filters.append(makeFilter("noAgid", stats.noAgid));
#endif
    filters.append(makeFilter("hourOfWeek", stats.hourOfWeekFiltered));
    filters.append(makeFilter("exchangeFiltered", stats.exchangeFiltered));
    filters.append(makeFilter("locationFiltered", stats.locationFiltered));
    filters.append(makeFilter("languageFiltered", stats.languageFiltered));
    filters.append(makeFilter("segmentsMissing", stats.segmentsMissing));
    filters.append(makeFilter("segmentsFiltered", stats.segmentFiltered));
    filters.append(makeFilter("userPartitionFiltered", stats.userPartitionFiltered));
    filters.append(makeFilter("hostFiltered", stats.urlFiltered));
    filters.append(makeFilter("urlFiltered", stats.urlFiltered));

    Json::Value report(Json::objectValue);
    report["total"] = requestCount;
    report["biddable"] = biddableCount;
    report["filters"] = filters;

    if (onEvent) {
        onEvent("checks", ET_COUNT, 1);

        uint64_t elapsedMicros = tm.elapsed_wall() * 1000000.0;
        onEvent("checkConfigMicros", ET_OUTCOME, elapsedMicros);
    }


    return report;
}



} // namepsace Recoset
