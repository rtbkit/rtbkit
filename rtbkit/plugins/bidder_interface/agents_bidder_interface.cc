/* agents_bidder_interface.cc
   Eric Robert, 2 April 2014
   Copyright (c) 2011 Datacratic.  All rights reserved.
*/

#include "agents_bidder_interface.h"
#include "rtbkit/common/messages.h"
#include "rtbkit/core/router/router.h"

using namespace Datacratic;
using namespace RTBKIT;

AgentsBidderInterface::AgentsBidderInterface(std::string const & name,
                                             std::shared_ptr<ServiceProxies> proxies,
                                             Json::Value const & config) {
}

void AgentsBidderInterface::sendAuctionMessage(std::shared_ptr<Auction> const & auction,
                                            double timeLeftMs,
                                            std::map<std::string, BidInfo> const & bidders) {
    for(auto & item : bidders) {
        auto & agent = item.first;
        auto & spots = item.second.imp;
        auto & info = router->agents[agent];
        WinCostModel wcm = auction->exchangeConnector->getWinCostModel(*auction, *info.config);
        router->sendAgentMessage(agent,
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

void AgentsBidderInterface::sendWinMessage(std::string const & agent,
                                        std::string const & id,
                                        Amount price) {
    router->sendAgentMessage(agent,
                             "WIN",
                             Date::now(),
                             "guaranteed",
                             id,
                             0,
                             price.toString());
}

void AgentsBidderInterface::sendLossMessage(std::string const & agent,
                                         std::string const & id) {
    router->sendAgentMessage(agent,
                             "LOSS",
                             Date::now(),
                             "guaranteed",
                             id,
                             0,
                             Amount().toString());
}

void AgentsBidderInterface::sendBidLostMessage(std::string const & agent,
                                            std::shared_ptr<Auction> const & auction) {
    router->sendAgentMessage(agent,
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

void AgentsBidderInterface::sendBidDroppedMessage(std::string const & agent,
                                               std::shared_ptr<Auction> const & auction) {
    router->sendAgentMessage(agent,
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

void AgentsBidderInterface::sendBidInvalidMessage(std::string const & agent,
                                               std::string const & reason,
                                               std::shared_ptr<Auction> const & auction) {
    router->sendAgentMessage(agent,
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

void AgentsBidderInterface::sendNoBudgetMessage(std::string const & agent,
                                             std::shared_ptr<Auction> const & auction) {
    router->sendAgentMessage(agent,
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

void AgentsBidderInterface::sendTooLateMessage(std::string const & agent,
                                            std::shared_ptr<Auction> const & auction) {
    router->sendAgentMessage(agent,
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

void AgentsBidderInterface::sendMessage(std::string const & agent,
                                     std::string const & message) {
    router->sendAgentMessage(agent,
                             message,
                             Date::now());
}

void AgentsBidderInterface::sendErrorMessage(std::string const & agent,
                                          std::string const & error,
                                          std::vector<std::string> const & payload) {
    router->sendAgentMessage(agent,
                             "ERROR",
                             Date::now(),
                             error,
                             payload);
}

void AgentsBidderInterface::sendPingMessage(std::string const & agent,
                                         int ping) {
    if(ping == 0) {
        router->sendAgentMessage(agent,
                                 "PING0",
                                 Date::now(),
                                 "null");
    }
    else {
        router->sendAgentMessage(agent,
                                 "PING1",
                                 Date::now(),
                                 "null");
    }
}

void AgentsBidderInterface::send(std::shared_ptr<PostAuctionEvent> const & event) {
    /*
    router->sendAgentMessage(agent,
                             event->label,
                             Date::now(),
                             "guaranteed",
                             event->auctionId,
                             event->adSpotId,
                             event->winPrice.toString());*/
}

//
// factory
//

struct AtInit {
    AtInit()
    {
        BidderInterface::registerFactory("agents", [](std::string const & name , std::shared_ptr<ServiceProxies> const & proxies, Json::Value const & json) {
            return new AgentsBidderInterface(name, proxies, json);
        });
    }
} atInit;

