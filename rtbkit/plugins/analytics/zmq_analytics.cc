/* zmq_analytics.cc
   Sirma Cagil Altay, 7 Mar 2016
   Copyright (c) 2015 Datacratic.  All rights reserved.

   This is the base class used to build a custom event logger plugin
*/

#include "zmq_analytics.h"


#include "soa/types/date.h"
#include "jml/arch/format.h"
#include "rtbkit/core/router/router.h"
#include "soa/types/id.h"
#include "rtbkit/core/post_auction/events.h"
#include "rtbkit/common/currency.h"
#include "boost/algorithm/string/trim.hpp"

using namespace Datacratic;

namespace RTBKIT {

ZmqAnalytics::ZmqAnalytics(const std::string & service_name, std::shared_ptr<ServiceProxies> proxies)
    : Analytics(service_name+"/logger", proxies),
      zmq_publisher_(getZmqContext())
{}

ZmqAnalytics::~ZmqAnalytics() {}

void ZmqAnalytics::init()
{
    zmq_publisher_.init(getServices()->config, serviceName());
}

void ZmqAnalytics::bindTcp(const std::string & port_range)
{
    zmq_publisher_.bindTcp(getServices()->ports->getRange(port_range));
}

void ZmqAnalytics::start()
{
    zmq_publisher_.start();
}

void ZmqAnalytics::shutdown()
{
    zmq_publisher_.shutdown();
}


/**********************************************************************************************
* USED IN ROUTER
**********************************************************************************************/

void ZmqAnalytics::logMarkMessage(const Router & router,
                                  const double & last_check)
{
    zmq_publisher_.publish("MARK",
                           Date::now().print(5),
                           Date::fromSecondsSinceEpoch(last_check).print(),
                           ML::format("active: %zd augmenting, %zd inFlight, "
                                      "%zd agents",
                                      router.augmentationLoop.numAugmenting(),
                                      router.inFlight.size(),                                             
                                      router.agents.size())
                           );
}

void ZmqAnalytics::logBidMessage(const std::string & agent,
                                 const Id & auctionId,
                                 const std::string & bids,
                                 const std::string & meta) 
{
    zmq_publisher_.publish("BID",
                           Date::now().print(5),
                           agent,
                           auctionId.toString(),
                           bids,
                           meta
                           );
}

void ZmqAnalytics::logAuctionMessage(const Id & auctionId,
                                     const std::string & auctionRequest)
{
    zmq_publisher_.publish("AUCTION", 
                           Date::now().print(5),
                           auctionId.toString(),
                           auctionRequest
                           );
}

void ZmqAnalytics::logConfigMessage(const std::string & agent,
                                    const std::string & config)
{
    zmq_publisher_.publish("CONFIG",
                           Date::now().print(5),
                           agent,
                           config
                          );
}

void ZmqAnalytics::logNoBudgetMessage(const std::string agent,
                                      const Id & auctionId,
                                      const std::string & bids,
                                      const std::string & meta)
{
    zmq_publisher_.publish("NOBUDGET",
                           Date::now().print(5),
                           agent,
                           auctionId.toString(),
                           bids,
                           meta
                          );
}

void ZmqAnalytics::logMessage(const std::string & msg,
                              const std::string agent,
                              const Id & auctionId,
                              const std::string & bids,
                              const std::string & meta)
{
    zmq_publisher_.publish(msg,
                           Date::now().print(5),
                           agent,
                           auctionId.toString(),
                           bids,
                           meta
                          );

}

void ZmqAnalytics::logUsageMessage(Router & router,
                                   const double & period)
{
    std::string p = std::to_string(period);

    for (auto it = router.lastAgentUsageMetrics.begin();
         it != router.lastAgentUsageMetrics.end();) {
        if (router.agents.count(it->first) == 0) {
            it = router.lastAgentUsageMetrics.erase(it);
        }
        else {
            it++;
        }
    }

    for (const auto & item : router.agents) {
        auto & info = item.second;
        auto & last = router.lastAgentUsageMetrics[item.first];

        Router::AgentUsageMetrics newMetrics(info.stats->intoFilters,
                                     info.stats->passedStaticFilters,
                                     info.stats->passedDynamicFilters,
                                     info.stats->auctions,
                                     info.stats->bids);
        Router::AgentUsageMetrics delta = newMetrics - last;

        zmq_publisher_.publish("USAGE",
                               Date::now().print(5),
                               "AGENT", 
                               p, 
                               item.first,
                               info.config->account.toString(),
                               delta.intoFilters,
                               delta.passedStaticFilters,
                               delta.passedDynamicFilters,
                               delta.auctions,
                               delta.bids,
                               info.config->bidProbability);
        last = move(newMetrics);
    }

    {
        Router::RouterUsageMetrics newMetrics;
        int numExchanges = 0;
        float acceptAuctionProbability(0.0);

        router.forAllExchanges([&](std::shared_ptr<ExchangeConnector> const & item) {
            ++numExchanges;
            newMetrics.numRequests += item->numRequests;
            newMetrics.numAuctions += item->numAuctions;
            acceptAuctionProbability += item->acceptAuctionProbability;
        });
        newMetrics.numBids = router.numBids;
        newMetrics.numNoPotentialBidders = router.numNoPotentialBidders;
        newMetrics.numAuctionsWithBid = router.numAuctionsWithBid;

        Router:: RouterUsageMetrics delta = newMetrics - router.lastRouterUsageMetrics;


        zmq_publisher_.publish("USAGE",
                               Date::now().print(5),
                               "ROUTER", 
                               p, 
                               delta.numRequests,
                               delta.numAuctions,
                               delta.numNoPotentialBidders,
                               delta.numBids,
                               delta.numAuctionsWithBid,
                               acceptAuctionProbability / numExchanges);

        router.lastRouterUsageMetrics = move(newMetrics);
    }
}

void ZmqAnalytics::logErrorMessage(const std::string & error,
                                   const std::vector<std::string> & message)
{
    zmq_publisher_.publish("ERROR",
                           Date::now().print(5),
                           error,
                           message
                          );
}

void ZmqAnalytics::logRouterErrorMessage(const std::string & function,
                                         const std::string & exception, 
                                         const std::vector<std::string> & message)
{
    zmq_publisher_.publish("ROUTERERROR",
                           Date::now().print(5),
                           function,
                           exception,
                           message
                          );
}


/**********************************************************************************************
* USED IN PA
**********************************************************************************************/

void ZmqAnalytics::logMatchedWinLoss(const MatchedWinLoss & matchedWinLoss) 
{
    zmq_publisher_.publish(
            "MATCHED" + matchedWinLoss.typeString(),                // 0
            Date::now().print(5),                                   // 1

            matchedWinLoss.auctionId.toString(),                    // 2
            std::to_string(matchedWinLoss.impIndex),                // 3
            matchedWinLoss.response.agent,                          // 4
            matchedWinLoss.response.account.at(1, ""),              // 5

            matchedWinLoss.winPrice.toString(),                     // 6
            matchedWinLoss.response.price.maxPrice.toString(),      // 7
            std::to_string(matchedWinLoss.response.price.priority), // 8

            matchedWinLoss.requestStr,                              // 9
            matchedWinLoss.response.bidData.toJsonStr(),            // 10
            matchedWinLoss.response.meta,                           // 11

            // This is where things start to get weird.

            std::to_string(matchedWinLoss.response.creativeId),     // 12
            matchedWinLoss.response.creativeName,                   // 13
            matchedWinLoss.response.account.at(0, ""),              // 14

            matchedWinLoss.uids.toJsonStr(),                        // 15
            matchedWinLoss.meta,                                    // 16

            // And this is where we lose all pretenses of sanity.

            matchedWinLoss.response.account.at(0, ""),              // 17
            matchedWinLoss.impId.toString(),                        // 18
            matchedWinLoss.response.account.toString(),             // 19

            // Ok back to sanity now.

            matchedWinLoss.requestStrFormat,                        // 20
            matchedWinLoss.rawWinPrice.toString(),                  // 21
            matchedWinLoss.augmentations.toString()                 // 22
        );
}

void ZmqAnalytics::logMatchedCampaignEvent(const MatchedCampaignEvent & matchedCampaignEvent)
{
    zmq_publisher_.publish(
            "MATCHED" + matchedCampaignEvent.label,    // 0
            Date::now().print(5),                      // 1

            matchedCampaignEvent.auctionId.toString(), // 2
            matchedCampaignEvent.impId.toString(),     // 3
            matchedCampaignEvent.requestStr,           // 4

            matchedCampaignEvent.bid,                  // 5
            matchedCampaignEvent.win,                  // 6
            matchedCampaignEvent.campaignEvents,       // 7
            matchedCampaignEvent.visits,               // 8

            matchedCampaignEvent.account.at(0, ""),    // 9
            matchedCampaignEvent.account.at(1, ""),    // 10
            matchedCampaignEvent.account.toString(),   // 11

            matchedCampaignEvent.requestStrFormat      // 12
    );
}

void ZmqAnalytics::logUnmatchedEvent(const UnmatchedEvent & unmatchedEvent)
{
    zmq_publisher_.publish(
            // Use event type not label since label is only defined for campaign events.
            "UNMATCHED" + string(print(unmatchedEvent.event.type)),             // 0
            Date::now().print(5),                                               // 1

            unmatchedEvent.reason,                                              // 2
            unmatchedEvent.event.auctionId.toString(),                          // 3
            unmatchedEvent.event.adSpotId.toString(),                           // 4

            std::to_string(unmatchedEvent.event.timestamp.secondsSinceEpoch()), // 5
            unmatchedEvent.event.metadata.toJson()                              // 6
        );
}

void ZmqAnalytics::logPostAuctionErrorEvent(const PostAuctionErrorEvent & postAuctionErrorEvent)
{
    zmq_publisher_.publish("PAERROR",
                           Date::now().print(5),
                           postAuctionErrorEvent.key,
                           postAuctionErrorEvent.message);
}

void ZmqAnalytics::logPAErrorMessage(const std::string & function,
                                     const std::string & exception, 
                                     const std::vector<std::string> & message)
{
    zmq_publisher_.publish("PAERROR",
                           Date::now().print(5),
                           function,
                           exception,
                           message
                          );
}


/**********************************************************************************************
* USED IN MOCK ADSERVER CONNECTOR
**********************************************************************************************/

void ZmqAnalytics::logMockWinMessage(const std::string & eventAuctionId,
                                     const std::string & eventWinPrice)
{
    zmq_publisher_.publish("WIN",
                           Date::now().print(3),
                           eventAuctionId,
                           eventWinPrice,
                           "0");
}

/**********************************************************************************************
* USED IN STANDARD ADSERVER CONNECTOR
**********************************************************************************************/

void ZmqAnalytics::logStandardWinMessage(const std::string & timestamp,
                                         const std::string & bidRequestId,
                                         const std::string & impId,
                                         const std::string & winPrice) 
{
    zmq_publisher_.publish("WIN",
                           timestamp,
                           bidRequestId,
                           impId,
                           winPrice);
}

void ZmqAnalytics::logStandardEventMessage(const std::string & eventType,
                                           const std::string & timestamp,
                                           const std::string & bidRequestId,
                                           const std::string & impId,
                                           const std::string & userIds)
{
    zmq_publisher_.publish(eventType,
                           timestamp,
                           bidRequestId,
                           impId,
                           userIds);
}

/**********************************************************************************************
* USED IN OTHER ADSERVER CONNECTORS
**********************************************************************************************/

void ZmqAnalytics::logAdserverEvent(const std::string & type,
                                    const std::string & bidRequestId,
                                    const std::string & impId)
{
    zmq_publisher_.publish(type,
                           type,
                           bidRequestId,
                           impId);
}

void ZmqAnalytics::logAdserverWin(const std::string & timestamp,
                                  const std::string & auctionId,
                                  const std::string & adSpotId,
                                  const std::string & accountKey,
                                  const std::string & winPrice,
                                  const std::string & dataCost)
{
    zmq_publisher_.publish("WIN",
                           timestamp,
                           auctionId,
                           adSpotId,
                           accountKey,
                           winPrice,
                           dataCost);
}

void ZmqAnalytics::logAuctionEventMessage(const std::string & event,
                                          const std::string & timestamp,
                                          const std::string & auctionId,
                                          const std::string & adSpotId,
                                          const std::string & userId)
{
    zmq_publisher_.publish(event,
                           timestamp,
                           auctionId,
                           adSpotId,
                           userId);
}

void ZmqAnalytics::logEventJson(const std::string & event,
                                const std::string & timestamp,
                                const std::string & json)
{
    zmq_publisher_.publish(event,
                           timestamp,
                           json);
}

void ZmqAnalytics::logDetailedWin(const std::string timestamp,
                                  const std::string & json,
                                  const std::string & auctionId,
                                  const std::string & spotId,
                                  const std::string & price,
                                  const std::string & userIds,
                                  const std::string & campaign,
                                  const std::string & strategy,
                                  const std::string & bidTimeStamp)
{
    zmq_publisher_.publish("WIN",
                           timestamp,
                           json,
                           auctionId,
                           spotId,
                           price,
                           userIds,
                           campaign,
                           strategy,
                           bidTimeStamp);
}

} // namespace RTBKIT

namespace {

struct AtInit {
    AtInit()
    {
        PluginInterface<Analytics>::registerPlugin(
            "zmq",
            [](const std::string & service_name, std::shared_ptr<ServiceProxies> proxies)
            {
                return new RTBKIT::ZmqAnalytics(service_name, std::move(proxies));
            });
    }
} atInit;
    
} // anonymous namespace
