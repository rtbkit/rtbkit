/* agents_bidder_interface.h
   Eric Robert, 2 April 2014
   Copyright (c) 2011 Datacratic.  All rights reserved.
*/

#pragma once

#include "rtbkit/common/bidder_interface.h"

namespace RTBKIT {

struct AgentsBidderInterface : public BidderInterface
{
    AgentsBidderInterface(std::string const & name = "bidder",
                          std::shared_ptr<ServiceProxies> proxies = std::make_shared<ServiceProxies>(),
                          Json::Value const & config = Json::Value());

    void sendAuctionMessage(std::shared_ptr<Auction> const & auction,
                            double timeLeftMs,
                            std::map<std::string, BidInfo> const & bidders);

    void sendWinMessage(std::string const & agent,
                        std::string const & id,
                        Amount price);

    void sendLossMessage(std::string const & agent,
                         std::string const & id);

    void sendBidLostMessage(std::string const & agent,
                            std::shared_ptr<Auction> const & auction);

    void sendBidDroppedMessage(std::string const & agent,
                               std::shared_ptr<Auction> const & auction);

    void sendBidInvalidMessage(std::string const & agent,
                               std::string const & reason,
                               std::shared_ptr<Auction> const & auction);

    void sendNoBudgetMessage(std::string const & agent,
                             std::shared_ptr<Auction> const & auction);

    void sendTooLateMessage(std::string const & agent,
                            std::shared_ptr<Auction> const & auction);

    void sendMessage(std::string const & agent,
                     std::string const & message);

    void sendErrorMessage(std::string const & agent,
                          std::string const & error,
                          std::vector<std::string> const & payload);

    void sendPingMessage(std::string const & agent,
                         int ping);

    void send(std::shared_ptr<PostAuctionEvent> const & event);
};

}

