/* bidding_interface.h
   Eric Robert, 2 April 2014
   Copyright (c) 2011 Datacratic.  All rights reserved.
*/

#pragma once

#include "soa/service/service_base.h"
#include "soa/service/zmq_endpoint.h"
#include "soa/service/typed_message_channel.h"
#include "rtbkit/common/auction_events.h"
#include "rtbkit/core/router/router_types.h"
#include "rtbkit/core/post_auction/events.h"

namespace RTBKIT {

class Router;
class AgentBridge;

struct BidderInterface : public ServiceBase
{
    BidderInterface(ServiceBase & parent,
                    std::string const & name = "bidder");

    BidderInterface(std::shared_ptr<ServiceProxies> proxies = std::make_shared<ServiceProxies>(),
                    std::string const & name = "bidder");

    void init(AgentBridge * value, Router * r = nullptr);
    virtual void start();

    virtual
    void sendAuctionMessage(std::shared_ptr<Auction> const & auction,
                            double timeLeftMs,
                            std::map<std::string, BidInfo> const & bidders) = 0;

    virtual
    void sendWinLossMessage(MatchedWinLoss const & event) = 0;

    virtual
    void sendLossMessage(std::string const & agent,
                         std::string const & id) = 0;

    virtual
    void sendCampaignEventMessage(std::string const & agent,
                                  MatchedCampaignEvent const & event) = 0;

    virtual
    void sendBidLostMessage(std::string const & agent,
                            std::shared_ptr<Auction> const & auction) = 0;

    virtual
    void sendBidDroppedMessage(std::string const & agent,
                               std::shared_ptr<Auction> const & auction) = 0;

    virtual
    void sendBidInvalidMessage(std::string const & agent,
                               std::string const & reason,
                               std::shared_ptr<Auction> const & auction) = 0;

    virtual
    void sendNoBudgetMessage(std::string const & agent,
                             std::shared_ptr<Auction> const & auction) = 0;

    virtual
    void sendTooLateMessage(std::string const & agent,
                            std::shared_ptr<Auction> const & auction) = 0;

    virtual
    void sendMessage(std::string const & agent,
                     std::string const & message) = 0;

    virtual
    void sendErrorMessage(std::string const & agent,
                          std::string const & error,
                          std::vector<std::string> const & payload) = 0;

    virtual
    void sendPingMessage(std::string const & agent,
                         int ping) = 0;

    //
    // factory
    //

    static std::shared_ptr<BidderInterface> create(std::string name,
                                                    std::shared_ptr<ServiceProxies> const & proxies,
                                                    Json::Value const & json);

    typedef std::function<BidderInterface * (std::string name,
                                              std::shared_ptr<ServiceProxies> const & proxies,
                                              Json::Value const & json)> Factory;

    static void registerFactory(std::string const & name, Factory factory);

protected:

    Router * router;
    AgentBridge * bridge;
};

}

