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
        std::cout << config << std::endl;
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
    using namespace std;

    typedef std::map<std::string, BidInfo> Bidders;

    typedef std::map<std::shared_ptr<BidderInterface>, Bidders> Aggregate;
    Aggregate aggregate;

    for (const auto &bidder: bidders) {
        const auto &agentConfig = bidder.second.agentConfig;
        auto selector = BidderInterfaceSelector::fromAgentConfig(
                                agentConfig, bidderInterfaces);

        auto iface = selector.pickBidderInterface(auction);

        auto &stat = stats_.statsForAgent(bidder.first);
        ++stat.totalAuctions;
        stat.incrForInterface(iface->interfaceName(),
                              &InterfaceStats::auctions);

        aggregate[iface].insert(bidder);
    }

    for (const auto &iface: aggregate) {
        iface.first->sendAuctionMessage(auction, timeLeftMs, iface.second);
    }
}

void MultiBidderInterface::sendLossMessage(std::string const & agent,
                                          std::string const & id) {
    auto iface = dispatchBidderInterface(agent, Id(id),
                            &BidderInterface::sendLossMessage,
                            agent, id);

    stats_.incrForInterface(agent, iface->interfaceName(),
                            &InterfaceStats::loss);
}

void MultiBidderInterface::sendWinLossMessage(MatchedWinLoss const & event) {
    auto iface = dispatchBidderInterface(event.response.agentConfig, event.auctionId,
                            &BidderInterface::sendWinLossMessage,
                            event);

    const auto &agent = event.response.agent;
    size_t InterfaceStats::*member =
        event.type == MatchedWinLoss::Loss ? &InterfaceStats::wins : &InterfaceStats::loss;

    stats_.incrForInterface(agent, iface->interfaceName(), member);
}


void MultiBidderInterface::sendBidLostMessage(std::string const & agent,
                                             std::shared_ptr<Auction> const & auction) {
    dispatchBidderInterface(agent, auction->id,
                            &BidderInterface::sendBidLostMessage,
                            agent, auction);
}

void MultiBidderInterface::sendCampaignEventMessage(std::string const & agent,
                                                   MatchedCampaignEvent const & event) {
    dispatchBidderInterface(agent, event.auctionId,
                            &BidderInterface::sendCampaignEventMessage,
                            agent, event);
}

void MultiBidderInterface::sendBidDroppedMessage(std::string const & agent,
                                                std::shared_ptr<Auction> const & auction) {

    dispatchBidderInterface(agent, auction->id,
                            &BidderInterface::sendBidDroppedMessage,
                            agent, auction);
}

void MultiBidderInterface::sendBidInvalidMessage(std::string const & agent,
                                                std::string const & reason,
                                                std::shared_ptr<Auction> const & auction) {
    dispatchBidderInterface(agent, auction->id,
                            &BidderInterface::sendBidInvalidMessage,
                            agent, reason, auction);
}

void MultiBidderInterface::sendNoBudgetMessage(std::string const & agent,
                                              std::shared_ptr<Auction> const & auction) {
    dispatchBidderInterface(agent, auction->id,
                            &BidderInterface::sendNoBudgetMessage,
                            agent, auction);
}

void MultiBidderInterface::sendTooLateMessage(std::string const & agent,
                                             std::shared_ptr<Auction> const & auction) {
    dispatchBidderInterface(agent, auction->id,
                            &BidderInterface::sendTooLateMessage,
                            agent, auction);
}

void MultiBidderInterface::sendMessage(std::string const & agent,
                                      std::string const & message) {
}

void MultiBidderInterface::sendErrorMessage(std::string const & agent,
                                           std::string const & error,
                                           std::vector<std::string> const & payload) {
}

void MultiBidderInterface::sendPingMessage(std::string const & agent,
                                          int ping) {
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
