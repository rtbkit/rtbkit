/* multi_bidder_interface.cc
   Mathieu Stefani, 18 July 2014
   Copyright (c) 2014 Datacratic.  All rights reserved.
   
   Implementation of the MultiBidderInterface
*/

#include "multi_bidder_interface.h"

using namespace Datacratic;
using namespace RTBKIT;

MultiBidderInterface::MultiBidderInterface(
        const std::string &serviceName,
        std::shared_ptr<ServiceProxies> proxies,
        const Json::Value &config
        ) 
    : BidderInterface(proxies, serviceName)
{
    ExcCheck(config["type"].asString() == "multi",
             "Constructing bad BidderInterface type");

    auto interfaces = config["interfaces"];
    if (interfaces.empty()) {
        throw ML::Exception("MultiBidderInterface must at least specify one interface");
    }

    for (const auto &interface: interfaces) {

        auto it = interface.begin();

        auto config = *it;
        std::string name = it.memberName();

        auto bidder = BidderInterface::create(name + ".bidder", proxies,
                                             config);
        bidder->setInterfaceName(name);

        bidderInterfaces.insert(
                std::make_pair(name, bidder));
    }
}

void MultiBidderInterface::init(AgentBridge *bridge, Router *router)
{
    for (auto &iface: bidderInterfaces) {
        iface.second->init(bridge, router);
    }

    this->bridge = bridge;
    this->router = router;
}


void MultiBidderInterface::start() {
    for (const auto &iface: bidderInterfaces) {
        iface.second->start();
    }
}

void MultiBidderInterface::shutdown() {
    for (const auto &iface: bidderInterfaces) {
        iface.second->shutdown();
    }
}

void MultiBidderInterface::sendAuctionMessage(std::shared_ptr<Auction> const & auction,
                                             double timeLeftMs,
                                             std::map<std::string, BidInfo> const & bidders) {

    typedef std::map<std::string, BidInfo> Bidders;

    typedef std::map<std::shared_ptr<BidderInterface>, Bidders> Aggregate;
    Aggregate aggregate;

    for (const auto &bidder: bidders) {
        const auto &agentConfig = bidder.second.agentConfig;

        auto iface = findInterface(agentConfig->bidderInterface, bidder.first);
        stats_.incr(iface, &InterfaceStats::auctions);

        aggregate[iface].insert(bidder);
    }

    for (const auto &iface: aggregate) {
        iface.first->sendAuctionMessage(auction, timeLeftMs, iface.second);
    }
}

void MultiBidderInterface::sendLossMessage(
        const std::shared_ptr<const AgentConfig>& agentConfig,
        std::string const & agent, std::string const & id) {
    dispatchBidderInterface(agent, agentConfig,
                            &BidderInterface::sendLossMessage,
                            agentConfig, agent, id);
}

void MultiBidderInterface::sendWinLossMessage(
        const std::shared_ptr<const AgentConfig>& agentConfig,
        MatchedWinLoss const & event) {

    dispatchBidderInterface(event.response.agent, agentConfig,
                            &BidderInterface::sendWinLossMessage,
                            agentConfig, event);
}


void MultiBidderInterface::sendBidLostMessage(
        const std::shared_ptr<const AgentConfig>& agentConfig,
        std::string const & agent, std::shared_ptr<Auction> const & auction) {
    dispatchBidderInterface(agent, agentConfig,
                            &BidderInterface::sendBidLostMessage,
                            agentConfig, agent, auction);
}

void MultiBidderInterface::sendCampaignEventMessage(
        const std::shared_ptr<const AgentConfig>& agentConfig,
        std::string const & agent, MatchedCampaignEvent const & event) {

    dispatchBidderInterface(agent, agentConfig,
                            &BidderInterface::sendCampaignEventMessage,
                            agentConfig, agent, event);
}

void MultiBidderInterface::sendBidDroppedMessage(
        const std::shared_ptr<const AgentConfig>& agentConfig,
        std::string const & agent, std::shared_ptr<Auction> const & auction) {

    dispatchBidderInterface(agent, agentConfig,
                            &BidderInterface::sendBidDroppedMessage,
                            agentConfig, agent, auction);
}

void MultiBidderInterface::sendBidInvalidMessage(
        const std::shared_ptr<const AgentConfig>& agentConfig,
        std::string const & agent, std::string const & reason,
        std::shared_ptr<Auction> const & auction) {

    dispatchBidderInterface(agent, agentConfig,
                            &BidderInterface::sendBidInvalidMessage,
                            agentConfig, agent, reason, auction);
}

void MultiBidderInterface::sendNoBudgetMessage(
        const std::shared_ptr<const AgentConfig>& agentConfig,
        std::string const & agent, std::shared_ptr<Auction> const & auction) {

    dispatchBidderInterface(agent, agentConfig,
                            &BidderInterface::sendNoBudgetMessage,
                            agentConfig, agent, auction);
}

void MultiBidderInterface::sendTooLateMessage(
        const std::shared_ptr<const AgentConfig>& agentConfig,
        std::string const & agent, std::shared_ptr<Auction> const & auction) {

    dispatchBidderInterface(agent, agentConfig,
                            &BidderInterface::sendTooLateMessage,
                            agentConfig, agent, auction);
}

void MultiBidderInterface::sendMessage(
        const std::shared_ptr<const AgentConfig>& agentConfig,
        std::string const & agent, std::string const & message) {
    if (message == "NEEDCONFIG") {
        dispatchAllInterfaces(&BidderInterface::sendMessage, agentConfig, agent, message);
    }
    else {
        dispatchBidderInterface(agent, agentConfig,
                                &BidderInterface::sendMessage,
                                agentConfig, agent, message);
    }
}

void MultiBidderInterface::sendErrorMessage(
        const std::shared_ptr<const AgentConfig>& agentConfig,
        std::string const & agent, std::string const & error,
        std::vector<std::string> const & payload) {
    dispatchBidderInterface(agent, agentConfig,
                            &BidderInterface::sendErrorMessage,
                            agentConfig, agent, error, payload);
}

void MultiBidderInterface::sendPingMessage(
        const std::shared_ptr<const AgentConfig>& agentConfig,
        std::string const & agent, int ping) {
    dispatchBidderInterface(agent, agentConfig,
                            &BidderInterface::sendPingMessage,
                            agentConfig, agent, ping);
}

void MultiBidderInterface::registerLoopMonitor(LoopMonitor *monitor) const {
    for (const auto& iface: bidderInterfaces) {
        iface.second->registerLoopMonitor(monitor);
    }
}


//
// factory
//

namespace {

struct AtInit {
    AtInit()
    {
      PluginInterface<BidderInterface>::registerPlugin("multi",
          [](std::string const &serviceName,
             std::shared_ptr<ServiceProxies> const &proxies,
             Json::Value const &json)
          {
              return new MultiBidderInterface(serviceName, proxies, json);
          });
    }
} atInit;

}
  
