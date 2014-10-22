/* bidding_interface.h                                             -*- C++ -*-
   Eric Robert, 2 April 2014
   Copyright (c) 2011 Datacratic.  All rights reserved.
*/

#pragma once

#include "soa/service/service_base.h"
#include "soa/service/zmq_endpoint.h"
#include "soa/service/typed_message_channel.h"
#include "soa/service/loop_monitor.h"
#include "rtbkit/common/auction_events.h"
#include "rtbkit/core/router/router_types.h"
#include "rtbkit/core/post_auction/events.h"

namespace RTBKIT {

class Router;
class AgentBridge;

struct BidderInterface : public ServiceBase
{
    BidderInterface(ServiceBase & parent,
                    std::string const & serviceName = "bidderService");

    BidderInterface(std::shared_ptr<ServiceProxies> proxies = std::make_shared<ServiceProxies>(),
                    std::string const & serviceName = "bidderService");

    BidderInterface(const BidderInterface &other) = delete;
    BidderInterface &operator=(const BidderInterface &other) = delete;

    void setInterfaceName(const std::string &name);
    std::string interfaceName() const;

    virtual void init(AgentBridge * bridge, Router * r = nullptr);
    virtual void shutdown();

    virtual void start();

    virtual
    void sendAuctionMessage(std::shared_ptr<Auction> const & auction,
                            double timeLeftMs,
                            std::map<std::string, BidInfo> const & bidders) = 0;

    virtual
    void sendWinLossMessage(const std::shared_ptr<const AgentConfig>& agentConfig,
                            MatchedWinLoss const & event) = 0;

    virtual
    void sendLossMessage(const std::shared_ptr<const AgentConfig>& agentConfig,
                         std::string const & agent,
                         std::string const & id) = 0;

    virtual
    void sendCampaignEventMessage(const std::shared_ptr<const AgentConfig>& agentConfig,
                                  std::string const & agent,
                                  MatchedCampaignEvent const & event) = 0;

    virtual
    void sendBidLostMessage(const std::shared_ptr<const AgentConfig>& agentConfig,
                            std::string const & agent,
                            std::shared_ptr<Auction> const & auction) = 0;

    virtual
    void sendBidDroppedMessage(const std::shared_ptr<const AgentConfig>& agentConfig,
                               std::string const & agent,
                               std::shared_ptr<Auction> const & auction) = 0;

    virtual
    void sendBidInvalidMessage(const std::shared_ptr<const AgentConfig>& agentConfig,
                               std::string const & agent,
                               std::string const & reason,
                               std::shared_ptr<Auction> const & auction) = 0;

    virtual
    void sendNoBudgetMessage(const std::shared_ptr<const AgentConfig>& agentConfig,
                             std::string const & agent,
                             std::shared_ptr<Auction> const & auction) = 0;

    virtual
    void sendTooLateMessage(const std::shared_ptr<const AgentConfig>& agentConfig,
                            std::string const & agent,
                            std::shared_ptr<Auction> const & auction) = 0;

    virtual
    void sendMessage(const std::shared_ptr<const AgentConfig>& agentConfig,
                     std::string const & agent,
                     std::string const & message) = 0;

    virtual
    void sendErrorMessage(const std::shared_ptr<const AgentConfig>& agentConfig,
                          std::string const & agent,
                          std::string const & error,
                          std::vector<std::string> const & payload) = 0;

    virtual
    void sendPingMessage(const std::shared_ptr<const AgentConfig>& agentConfig,
                         std::string const & agent,
                         int ping) = 0;

    virtual void registerLoopMonitor(LoopMonitor *monitor) const { }

    //
    // factory
    //

    static std::shared_ptr<BidderInterface>
    create(std::string serviceName,
           std::shared_ptr<ServiceProxies> const & proxies,
           Json::Value const & json);

    typedef std::function<BidderInterface * (std::string serviceName,
                                             std::shared_ptr<ServiceProxies> const & proxies,
                                             Json::Value const & json)> Factory;

    static void registerFactory(std::string const & name, Factory factory);

    std::string name;
    Router * router;
    AgentBridge * bridge;

};

}
