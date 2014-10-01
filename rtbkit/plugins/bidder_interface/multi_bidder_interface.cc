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

void MultiBidderInterface::sendLossMessage(std::string const & agent,
                                          std::string const & id) {
    dispatchBidderInterface(agent,
                            &BidderInterface::sendLossMessage,
                            agent, id);
}

void MultiBidderInterface::sendWinLossMessage(MatchedWinLoss const & event) {

    dispatchBidderInterface(event.response.agent,
                            event.response.agentConfig,
                            &BidderInterface::sendWinLossMessage,
                            event);
}


void MultiBidderInterface::sendBidLostMessage(std::string const & agent,
                                             std::shared_ptr<Auction> const & auction) {
    dispatchBidderInterface(agent,
                            &BidderInterface::sendBidLostMessage,
                            agent, auction);
}

void MultiBidderInterface::sendCampaignEventMessage(std::string const & agent,
                                                   MatchedCampaignEvent const & event) {

    dispatchBidderInterface(agent,
                            &BidderInterface::sendCampaignEventMessage,
                            agent, event);
}

void MultiBidderInterface::sendBidDroppedMessage(std::string const & agent,
                                                std::shared_ptr<Auction> const & auction) {

    dispatchBidderInterface(agent,
                            &BidderInterface::sendBidDroppedMessage,
                            agent, auction);
}

void MultiBidderInterface::sendBidInvalidMessage(std::string const & agent,
                                                std::string const & reason,
                                                std::shared_ptr<Auction> const & auction) {

    dispatchBidderInterface(agent,
                            &BidderInterface::sendBidInvalidMessage,
                            agent, reason, auction);
}

void MultiBidderInterface::sendNoBudgetMessage(std::string const & agent,
                                              std::shared_ptr<Auction> const & auction) {

    dispatchBidderInterface(agent,
                            &BidderInterface::sendNoBudgetMessage,
                            agent, auction);
}

void MultiBidderInterface::sendTooLateMessage(std::string const & agent,
                                             std::shared_ptr<Auction> const & auction) {

    dispatchBidderInterface(agent,
                            &BidderInterface::sendTooLateMessage,
                            agent, auction);
}

void MultiBidderInterface::sendMessage(std::string const & agent,
                                      std::string const & message) {
    if (message == "NEEDCONFIG") {
        dispatchAllInterfaces(&BidderInterface::sendMessage, agent, message);
    }
    else {
        dispatchBidderInterface(agent, &BidderInterface::sendMessage, agent, message);
    }
}

void MultiBidderInterface::sendErrorMessage(std::string const & agent,
                                           std::string const & error,
                                           std::vector<std::string> const & payload) {
    dispatchBidderInterface(agent, &BidderInterface::sendErrorMessage, agent, error, payload);
}

void MultiBidderInterface::sendPingMessage(std::string const & agent,
                                          int ping) {
    dispatchBidderInterface(agent, &BidderInterface::sendPingMessage, agent, ping);
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
        BidderInterface::registerFactory("multi",
        [](std::string const & serviceName,
           std::shared_ptr<ServiceProxies> const & proxies,
           Json::Value const & json)
        {
            return new MultiBidderInterface(serviceName, proxies, json);
        });
    }
} atInit;

}
