/* rtb_router_types.cc
   Jeremy Barnes, 1 March 2012
   Copyright (c) 2012 Datacratic.  All rights reserved.

   Types for the RTB router.
*/

#include "router_types.h"
#include "rtbkit/core/agent_configuration/agent_config.h"
#include "jml/db/persistent.h"

using namespace std;
using namespace ML;

namespace RTBKIT {

void
DutyCycleEntry::
clear()
{
    starting = Date::now();
    nsSleeping = nsProcessing =  nEvents = 0;
    nsConfig = nsBid = nsAuction = nsWin = nsLoss = nsBidResult = 0;
    nsRemoveInFlightAuction = nsRemoveSubmittedAuction = 0;
    nsEraseLossTimeout = nsEraseAuction = 0;
    nsTimeout = nsSubmitted = nsImpression = nsClick = 0;
    nsStartBidding = 0;
    nsExpireInFlight = nsExpireSubmitted
        = nsExpireFinished = nsExpireBlacklist = nsExpireBanker
        = nsExpireDebug = nsOnExpireSubmitted = 0;
}
        
void
DutyCycleEntry::
operator += (const DutyCycleEntry & other)
{
    starting = std::min(starting, other.starting);
    ending = std::max(ending, other.ending);
    nsSleeping += other.nsSleeping;
    nsProcessing += other.nsProcessing;
    nEvents += other.nEvents;
    nsConfig += other.nsConfig;
    nsBid += other.nsBid;
    nsAuction += other.nsAuction;
    nsStartBidding += other.nsStartBidding;
    nsWin += other.nsWin;
    nsLoss += other.nsLoss;
    nsBidResult += other.nsBidResult;
    nsRemoveInFlightAuction += other.nsRemoveInFlightAuction;
    nsTimeout += other.nsTimeout;
    nsSubmitted += other.nsSubmitted;
    nsImpression += other.nsImpression;
    nsClick += other.nsClick;
    throw ML::Exception("not finished");
}

Json::Value
DutyCycleEntry::
toJson() const
{
    Json::Value result;

    double elapsedTime = ending.secondsSince(starting);

    Json::Value & times = result["times"];
    Json::Value & duty = result["duty"];

    result["elapsedTime"] = elapsedTime;
    result["events"] = nEvents;
    result["eventsPerSecond"] = nEvents / elapsedTime;
    result["nsProcessing"] = (double)nsProcessing;
    result["nsSleeping"] = (double)nsSleeping;
    result["dutyCycle"] = nsProcessing / (nsSleeping + nsProcessing + 0.0);

    auto doTime = [&] (const char * metric, uint64_t val)
        {
            times[metric] = val;
            duty[metric] = 100.0 * (double)val / (double)nsProcessing;
        };

    doTime("config", nsConfig);
    doTime("bid", nsBid);
    doTime("auction", nsAuction);
    doTime("win", nsWin);
    doTime("loss", nsLoss);
    doTime("bidResult", nsBidResult);
    doTime("removeInFlightAuction", nsRemoveInFlightAuction);
    doTime("eraseLossTimeout", nsEraseLossTimeout);
    doTime("eraseAuction", nsEraseAuction);
    doTime("expireInFlight", nsExpireInFlight);
    doTime("expireSubmitted", nsExpireSubmitted);
    doTime("expireFinished", nsExpireFinished);
    doTime("expireBlacklist", nsExpireBlacklist);
    doTime("expireBanker", nsExpireBanker);
    doTime("expireDebug", nsExpireDebug);
    doTime("onExpireSubmitted", nsOnExpireSubmitted);

    return result;
}

Json::Value
AgentInfo::
toJson(bool includeConfig, bool includeStats) const
{
    Json::Value result;
    result["configured"] = configured;
    result["lastHeartbeat"]
        = status->lastHeartbeat.print(4);
    result["numInFlight"] = status->numBidsInFlight;
    if (config && includeConfig) result["config"] = config->toJson(false);
    if (stats && includeStats) result["stats"] = stats->toJson();
    
    return result;
}

const std::string &
AgentInfo::
encodeBidRequest(const BidRequest & br) const
{
    throw ML::Exception("encodeBidRequest not there yet");
}

const std::string &
AgentInfo::
encodeBidRequest(const Auction & auction) const
{
    return auction.requestStr;
}

const std::string &
AgentInfo::
getBidRequestEncoding(const Auction & auction) const
{
    return auction.requestStrFormat;
}

void
AgentInfo::
setBidRequestFormat(const std::string & val)
{
    // TODO
}

AgentStats::
AgentStats()
    : auctions(0), bids(0), wins(0), losses(0), tooLate(0),
      invalid(0), noBudget(0),
      tooManyInFlight(0), noSpots(0), skippedBidProbability(0),
      urlFiltered(0), hourOfWeekFiltered(0),
      locationFiltered(0), languageFiltered(0),
      userPartitionFiltered(0),
      dataProfileFiltered(0),
      exchangeFiltered(0),
      segmentsMissing(0), segmentFiltered(0),
      augmentationTagsExcluded(0), userBlacklisted(0), notEnoughTime(0),
      requiredIdMissing(0),
      intoFilters(0), passedStaticFilters(0),
      passedStaticPhase1(0), passedStaticPhase2(0), passedStaticPhase3(0),
      passedDynamicFilters(0),
      bidErrors(0),
      filter1Excluded(0),
      filter2Excluded(0),
      filternExcluded(0),
      unknownWins(0), unknownLosses(0),
      requiredAugmentorIsMissing(0), augmentorValueIsNull(0)
{
}

Json::Value
AgentStats::
toJson() const
{
    Json::Value result;
    result["auctions"] = auctions;
    result["bids"] = bids;
    result["wins"] = wins;
    result["losses"] = losses;
    result["tooLate"] = tooLate;
    result["invalid"] = invalid;
    result["noBudget"] = noBudget;
    result["totalBid"] = totalBid.toJson();
    result["totalBidOnWins"] = totalBidOnWins.toJson();
    result["totalSpent"] = totalSpent.toJson();
    result["tooManyInFlight"] = tooManyInFlight;
    result["requiredIdMissing"] = requiredIdMissing;
    result["notEnoughTime"] = notEnoughTime;

    result["filter_noSpots"] = noSpots;
    result["filter_skippedBidProbability"] = skippedBidProbability;
    result["filter_urlFiltered"] = urlFiltered;
    result["filter_hourOfWeekFiltered"] = hourOfWeekFiltered;
    result["filter_locationFiltered"] = locationFiltered;
    result["filter_languageFiltered"] = languageFiltered;
    result["filter_userPartitionFiltered"] = userPartitionFiltered;
    result["filter_dataProfileFiltered"] = dataProfileFiltered;
    result["filter_exchangeFiltered"] = exchangeFiltered;
    result["filter_augmentationTagsExcluded"] = augmentationTagsExcluded;
    result["filter_userBlacklisted"] = userBlacklisted;
    result["filter_segmentsMissing"] = segmentsMissing;
    result["filter_segmentFiltered"] = segmentFiltered;

    result["bidErrors"] = bidErrors;
    result["unknownWins"] = unknownWins;
    result["unknownLosses"] = unknownLosses;
    result["intoFilters"] = intoFilters;
    result["passedStaticFilters"] = passedStaticFilters;
    result["passedStaticFiltersPhase1"] = passedStaticPhase1;
    result["passedStaticFiltersPhase2"] = passedStaticPhase2;
    result["passedStaticFiltersPhase3"] = passedStaticPhase3;
    result["passedDynamicFilters"] = passedDynamicFilters;

    result["filter1Excluded"] = filter1Excluded;
    result["filter2Excluded"] = filter2Excluded;
    result["filternExcluded"] = filternExcluded;

    result["requiredAugmentorIsMissing"] = requiredAugmentorIsMissing;
    result["augmentorValueIsNull"] = augmentorValueIsNull;

    return result;
}

Json::Value
FormatInfo::
toJson() const
{
    Json::Value result;
    result["numSpots"] = numSpots;
    result["numBids"] = numBids;
    return result;
}


/*****************************************************************************/
/* BIDDABLE SPOTS                                                            */
/*****************************************************************************/

Json::Value
BiddableSpots::
toJson() const
{
    /* Convert to JSON to send it on. */
    Json::Value result;

    for (unsigned i = 0;  i < size();  ++i) {
        Json::Value & val = result[i];
        val["spot"] = (*this)[i].first;
        Json::Value & creatives = val["creatives"];
        for (unsigned j = 0;  j < (*this)[i].second.size();
             ++j)
            creatives[j] = (*this)[i].second[j];
    }

    return result;
}

std::string
BiddableSpots::
toJsonStr() const
{
    /* Convert to JSON to send it on. */
    std::string result = "[";
    result.reserve(512);

    for (unsigned i = 0;  i < size();  ++i) {
        if (i != 0) result += ",";
        result += format("{\"spot\":%d,\"creatives\":[", (*this)[i].first);
        for (unsigned j = 0;  j < (*this)[i].second.size();
             ++j) {
            if (j > 0) result += ",";
            result += format("%d", (*this)[i].second[j]);
        }
        result += "]}";
    }
    result += "]";

    return result;
}

} // namespace RTBKIT

