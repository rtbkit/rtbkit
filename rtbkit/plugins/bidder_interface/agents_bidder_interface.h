/* agents_bidder_interface.h                                       -*- C++ -*-
   Eric Robert, 2 April 2014
   Copyright (c) 2011 Datacratic.  All rights reserved.
*/

#pragma once

#include "rtbkit/common/bidder_interface.h"
#include "soa/jsoncpp/json.h"
#include <iostream>

namespace RTBKIT {

struct AgentsBidderInterface : public BidderInterface
{
    AgentsBidderInterface(std::string const & serviceName = "bidderService",
                          std::shared_ptr<ServiceProxies> proxies = std::make_shared<ServiceProxies>(),
                          Json::Value const & config = Json::Value());

    ~AgentsBidderInterface();

    void sendAuctionMessage(std::shared_ptr<Auction> const & auction,
                            double timeLeftMs,
                            std::map<std::string, BidInfo> const & bidders);

    void sendWinLossMessage(const std::shared_ptr<const AgentConfig>& agentConfig,
                            MatchedWinLoss const & event);

    void sendLossMessage(const std::shared_ptr<const AgentConfig>& agentConfig,
                         std::string const & agent,
                         std::string const & id);

    void sendCampaignEventMessage(const std::shared_ptr<const AgentConfig>& agentConfig,
                                  std::string const & agent,
                                  MatchedCampaignEvent const & event);

    void sendBidLostMessage(const std::shared_ptr<const AgentConfig>& agentConfig,
                            std::string const & agent,
                            std::shared_ptr<Auction> const & auction);

    void sendBidDroppedMessage(const std::shared_ptr<const AgentConfig>& agentConfig,
                               std::string const & agent,
                               std::shared_ptr<Auction> const & auction);

    void sendBidInvalidMessage(const std::shared_ptr<const AgentConfig>& agentConfig,
                               std::string const & agent,
                               std::string const & reason,
                               std::shared_ptr<Auction> const & auction);

    void sendNoBudgetMessage(const std::shared_ptr<const AgentConfig>& agentConfig,
                             std::string const & agent,
                             std::shared_ptr<Auction> const & auction);

    void sendTooLateMessage(const std::shared_ptr<const AgentConfig>& agentConfig,
                            std::string const & agent,
                            std::shared_ptr<Auction> const & auction);

    void sendMessage(const std::shared_ptr<const AgentConfig>& agentConfig,
                     std::string const & agent,
                     std::string const & message);

    void sendErrorMessage(const std::shared_ptr<const AgentConfig>& agentConfig,
                          std::string const & agent,
                          std::string const & error,
                          std::vector<std::string> const & payload);

    void sendPingMessage(const std::shared_ptr<const AgentConfig>& agentConfig,
                         std::string const & agent,
                         int ping);

};

}

