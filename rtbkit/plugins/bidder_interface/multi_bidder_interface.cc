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

struct BidderInterfaceSelector {
    BidderInterfaceSelector() : totalProbability(0)
    { }

    void addInterface(const std::shared_ptr<BidderInterface>  &interface,
                      double probability)
    {
        ExcCheck(probability >= 0.0 && probability <= 1.0,
                  "Invalid probability");
        Entry entry;
        entry.interface = interface;
        entry.probability = static_cast<int>(probability * 100);
        totalProbability += entry.probability;
        entries.push_back(std::move(entry));
    }


    std::shared_ptr<BidderInterface>
    pickBidderInterface(const std::shared_ptr<Auction> &auction)
    {
        return pickBidderInterface(auction->id);
    }

    std::shared_ptr<BidderInterface>
    pickBidderInterface(const Id &auctionId)
    {
        ExcCheck(totalProbability == 100, "The total probability is not 100");
        std::sort(begin(entries), end(entries),
            [](const Entry &lhs, const Entry &rhs)
        {
            return lhs.probability < rhs.probability;
        });

        int rng = auctionId.hash() % totalProbability;
        int cumulativeProbability = 0;
        for (const auto &entry: entries) {
            cumulativeProbability += entry.probability;
            if (rng <= cumulativeProbability) {
                return entry.interface;
            }
        }

        ExcCheck(false, "Invalid code path");
    }

    static BidderInterfaceSelector fromAgentConfig(
            const std::shared_ptr<const AgentConfig> &config,
            const std::map<std::string, std::shared_ptr<BidderInterface>> &interfaces)
    {
        BidderInterfaceSelector selector;
        const auto &bidderConfig = config->bidderConfig;

        using namespace std;
        for (auto it = begin(bidderConfig); it != end(bidderConfig); ++it) {
            std::string ifaceName = it.memberName();

            auto ifaceIt = interfaces.find(ifaceName);
            if (ifaceIt == end(interfaces)) {
                throw ML::Exception(
                        "Could not find a BidderInterface for configuration '%s'",
                         ifaceName.c_str());
            }

            double probability = (*it)["probability"].asDouble();

            selector.addInterface(ifaceIt->second, probability);

        }

        return selector;
    }


private:
    struct Entry {
        std::shared_ptr<BidderInterface> interface;
        int probability;
    };

    std::vector<Entry> entries;
    int totalProbability;
};

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

        stats_[bidder.first].interfacesStats[iface->interfaceName()].auctionsSent++;
        stats_[bidder.first].totalAuctions++;
        aggregate[iface].insert(bidder);
    }

    for (const auto &iface: aggregate) {
        iface.first->sendAuctionMessage(auction, timeLeftMs, iface.second);
    }
}

void MultiBidderInterface::sendLossMessage(std::string const & agent,
                                          std::string const & id) {
}

void MultiBidderInterface::sendWinLossMessage(MatchedWinLoss const & event) {
    
}


void MultiBidderInterface::sendBidLostMessage(std::string const & agent,
                                             std::shared_ptr<Auction> const & auction) {
}

void MultiBidderInterface::sendCampaignEventMessage(std::string const & agent,
                                                   MatchedCampaignEvent const & event) {
    
}

void MultiBidderInterface::sendBidDroppedMessage(std::string const & agent,
                                                std::shared_ptr<Auction> const & auction) {
}

void MultiBidderInterface::sendBidInvalidMessage(std::string const & agent,
                                                std::string const & reason,
                                                std::shared_ptr<Auction> const & auction) {
}

void MultiBidderInterface::sendNoBudgetMessage(std::string const & agent,
                                              std::shared_ptr<Auction> const & auction) {
}

void MultiBidderInterface::sendTooLateMessage(std::string const & agent,
                                             std::shared_ptr<Auction> const & auction) {
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
