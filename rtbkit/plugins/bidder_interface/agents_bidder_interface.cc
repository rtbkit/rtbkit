/* agents_bidder_interface.cc
   Eric Robert, 2 April 2014
   Copyright (c) 2011 Datacratic.  All rights reserved.
*/

#include "rtbkit/common/messages.h"
#include "rtbkit/core/router/router.h"
#include "agents_bidder_interface.h"

using namespace Datacratic;
using namespace RTBKIT;

AgentsBidderInterface::AgentsBidderInterface(std::string const &serviceName,
                                             std::shared_ptr<ServiceProxies> proxies,
                                             Json::Value const & config)
    : BidderInterface(proxies, serviceName) {
}

AgentsBidderInterface::~AgentsBidderInterface() {
    this->shutdown();
}

void AgentsBidderInterface::sendAuctionMessage(std::shared_ptr<Auction> const & auction,
                                               double timeLeftMs,
                                               std::map<std::string, BidInfo> const & bidders) {

    for(auto & item : bidders) {
        auto & agent = item.first;
        auto & spots = item.second.imp;
        auto & info = router->agents[agent];
        WinCostModel wcm = auction->exchangeConnector->getWinCostModel(*auction, *info.config);

        bridge->sendAgentMessage(agent,
                                 "AUCTION",
                                 auction->start,
                                 auction->id,
                                 info.getBidRequestEncoding(*auction),
                                 info.encodeBidRequest(*auction),
                                 spots.toJsonStr(),
                                 std::to_string(timeLeftMs),
                                 auction->agentAugmentations[agent],
                                 wcm.toJson());
    }
}


void AgentsBidderInterface::sendWinLossMessage(
        const std::shared_ptr<const AgentConfig>& agentConfig, MatchedWinLoss const & event) {
    std::string channel =
        event.type == MatchedWinLoss::LateWin ? "LATEWIN" : event.typeString();

    bridge->sendAgentMessage(event.response.agent,
                              channel,
                              event.timestamp,
                              event.confidenceString(),

                              event.auctionId.toString(),
                              std::to_string(event.impIndex),
                              event.winPrice.toString(),

                              event.requestStrFormat,
                              event.requestStr,
                              event.response.bidData.toJsonStr(),
                              event.response.meta,
                              event.augmentations.toJson());

}

void AgentsBidderInterface::sendLossMessage(
        const std::shared_ptr<const AgentConfig>& agentConfig,
        std::string const & agent, std::string const & id) {
    bridge->sendAgentMessage(agent,
                             "LOSS",
                             Date::now(),
                             "guaranteed",
                             id,
                             0,
                             Amount().toString());
}

void AgentsBidderInterface::sendCampaignEventMessage(
        const std::shared_ptr<const AgentConfig>& agentConfig,
        std::string const & agent, MatchedCampaignEvent const & event) {
    bridge->sendAgentMessage(agent,
                             "CAMPAIGN_EVENT",
                             event.label,
                             Date::now(),

                             event.auctionId.toString(),
                             event.impId.toString(),
                             std::to_string(event.impIndex),

                             event.requestStrFormat,
                             event.requestStr,
                             event.augmentations.toJson(),

                             event.bid,
                             event.win,
                             event.campaignEvents,
                             event.visits);

}

void AgentsBidderInterface::sendBidLostMessage(
        const std::shared_ptr<const AgentConfig>& agentConfig,
        std::string const & agent, std::shared_ptr<Auction> const & auction) {
    bridge->sendAgentMessage(agent,
                             "LOST",
                             Date::now(),
                             "guaranteed",
                             auction->id,
                             0,
                             Amount().toString());
/*
-                    this->sendBidResponse(it->first,
-                                          info,
-                                          BS_LOSTBID,
-                                          this->getCurrentTime(),
-                                          "guaranteed", id);
-
+                    bpc->sendLostBidMessage(it->first, inFlight[id]);
*/
}

void AgentsBidderInterface::sendBidDroppedMessage(
        const std::shared_ptr<const AgentConfig>& agentConfig,
        std::string const & agent, std::shared_ptr<Auction> const & auction) {
    bridge->sendAgentMessage(agent,
                             "DROPPEDBID",
                             Date::now(),
                             "guaranteed",
                             auction->id,
                             0,
                             Amount().toString());
/*
-                        this->sendBidResponse(agent,
-                                              info,
-                                              BS_DROPPEDBID,
-                                              this->getCurrentTime(),
-                                              "guaranteed",
-                                              auctionId,
-                                              0, Amount(),
-                                              auctionInfo.auction.get());
+                        bpc->sendDroppedBid(agent, *auctionInfo.auction)
*/
}

void AgentsBidderInterface::sendBidInvalidMessage(
        const std::shared_ptr<const AgentConfig>& agentConfig,
        std::string const & agent, std::string const & reason,
        std::shared_ptr<Auction> const & auction) {
    bridge->sendAgentMessage(agent,
                             "INVALID",
                             Date::now(),
                             reason,
                             auction->id,
                             0,
                             Amount().toString());
/*
-            this->sendBidResponse
-                (agent, info, BS_INVALID, this->getCurrentTime(),
-                 formatted, auctionId,
-                 i, Amount(),
-                 auctionInfo.auction.get(),
-                 biddata, Json::Value(),
-                 auctionInfo.auction->agentAugmentations[agent]);
+            bpc->sendInvalidBid(agent, formatted, *auctionInfo.auction);
*/
}

void AgentsBidderInterface::sendNoBudgetMessage(
        const std::shared_ptr<const AgentConfig>& agentConfig,
        std::string const & agent, std::shared_ptr<Auction> const & auction) {
    bridge->sendAgentMessage(agent,
                             "NOBUDGET",
                             Date::now(),
                             "guaranteed",
                             auction->id,
                             0,
                             Amount().toString());
/*
-            this->sendBidResponse(agent, info, BS_NOBUDGET,
-                    this->getCurrentTime(),
-                    "guaranteed", auctionId, 0, Amount(),
-                    auctionInfo.auction.get(),
-                    biddata, meta, agentAugmentations);
+            bpc->sendNoBudget(agent, auctionInfo.auction.get(), biddata, meta);
*/
}

void AgentsBidderInterface::sendTooLateMessage(
        const std::shared_ptr<const AgentConfig>& agentConfig,
        std::string const & agent, std::shared_ptr<Auction> const & auction) {
    bridge->sendAgentMessage(agent,
                             "TOOLATE",
                             Date::now(),
                             "guaranteed",
                             auction->id,
                             0,
                             Amount().toString());

/*
-            case Auction::WinLoss::LOSS:    status = BS_LOSS;     break;
-            case Auction::WinLoss::TOOLATE: status = BS_TOOLATE;  break;
-            case Auction::WinLoss::INVALID: status = BS_INVALID;  break;

-            const string& agentAugmentations =
-                auctionInfo.auction->agentAugmentations[agent];
-
-            this->sendBidResponse(agent, info, status,
-                    this->getCurrentTime(),
-                    "guaranteed", auctionId, 0, Amount(),
-                    auctionInfo.auction.get(),
-                    biddata, meta, agentAugmentations);
*/

/*
-            string confidence = "guaranteed";
-
-            //cerr << fName << "sending agent message of type " << msg << endl;
-            sendBidResponse(response.agent, info, bidStatus,
-                            this->getCurrentTime(),
-                            confidence, auctionId,
-                            0, Amount(),
-                            auction.get(),
-                            response.bidData,
-                            response.meta,
-                            auction->agentAugmentations[response.agent]);
*/
}

void AgentsBidderInterface::sendMessage(
        const std::shared_ptr<const AgentConfig>& agentConfig,
        std::string const & agent, std::string const & message) {
    bridge->sendAgentMessage(agent,
                             message,
                             Date::now());
}

void AgentsBidderInterface::sendErrorMessage(
        const std::shared_ptr<const AgentConfig>& agentConfig,
        std::string const & agent, std::string const & error,
        std::vector<std::string> const & payload) {
    bridge->sendAgentMessage(agent,
                             "ERROR",
                             Date::now(),
                             error,
                             payload);
}

void AgentsBidderInterface::sendPingMessage(
        const std::shared_ptr<const AgentConfig>& agentConfig,
        std::string const & agent, int ping) {
    if(ping == 0) {
        bridge->sendAgentMessage(agent,
                                 "PING0",
                                 Date::now(),
                                 "null");
    }
    else {
        bridge->sendAgentMessage(agent,
                                 "PING1",
                                 Date::now(),
                                 "null");
    }
}

//
// factory
//

namespace {

struct AtInit {
    AtInit()
    {
      PluginInterface<BidderInterface>::registerPlugin("agents",
          [](std::string const &serviceName,
             std::shared_ptr<ServiceProxies> const &proxies,
             Json::Value const &json)
          {
              return new AgentsBidderInterface(serviceName, proxies, json);
          });
    }
} atInit;

}
