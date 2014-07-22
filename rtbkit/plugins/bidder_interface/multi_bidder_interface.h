/* multi_bidder_interface.h
   Mathieu Stefani, 17 July 2014
   Copyright (c) 2014 Datacratic.  All rights reserved.
   
   The Multi Bidder Interface class
*/

#pragma once

#include "rtbkit/common/bidder_interface.h"
#include "rtbkit/core/router/router.h"

namespace RTBKIT {

struct MultiBidderInterface : public BidderInterface {

    struct InterfaceStats {
        size_t auctions;
        size_t wins;
        size_t loss;
        size_t campaignEvents;
        size_t noBudget;
        size_t tooLate;
    };

    struct Stat {
        size_t totalAuctions;
        std::map<std::string, InterfaceStats> interfacesStats;

        size_t incrForInterface(const std::string &interfaceName,
                                size_t InterfaceStats::*Member)
        {
            auto& stats = interfacesStats[interfaceName];
            auto& value = stats.*Member;
            ++value;
            return value;
        }

        void dump(std::ostream &stream, size_t indent = 8) const
        {
            auto percentage = [=](size_t value) {
                return std::to_string((value * 100) / totalAuctions) + "%";
            };

            for (const auto &stat: interfacesStats) {
                stream << std::string(indent, ' ') << "* " << stat.first
                       << std::endl;
                stream << std::string(indent + 4, ' ') << "+ "
                       << "auctions: " << stat.second.auctions << " ("
                       << percentage(stat.second.auctions) << ")" << std::endl;
            }
            stream << std::endl;
        }

    };

    struct Stats {
        Stat &statsForAgent(const std::string &agent) {
            return stats[agent];
        }

        size_t incrForInterface(
                     const std::string &agent,
                     const std::string &interfaceName,
                     size_t InterfaceStats::*Member)
        {
            auto& stat = statsForAgent(agent);
            return stat.incrForInterface(interfaceName, Member);
        }

        void dump(std::ostream &stream) const
        {
            auto header = [&](const std::string &str) {
                std::string text = "Stats for " + str;
                stream << text << std::endl;
                stream << std::string(text.size(), '-') << std::endl;
            };

            for (const auto &stat: stats)
            {
                header(stat.first);
                const std::string indent(4, ' ');
                std::cerr << indent << "- Total auctions: " << stat.second.totalAuctions
                          << std::endl
                          << indent << "- Stats per interface: " << std::endl;
                stat.second.dump(stream);
            }
        }

    private:
        std::map<std::string, Stat> stats;
    };


    MultiBidderInterface(
        const std::string &serviceName = "bidderService",
        std::shared_ptr<ServiceProxies> proxies = std::make_shared<ServiceProxies>(),
        const Json::Value &config = Json::Value()
        );

    void init(AgentBridge *value, Router * router = nullptr);
    virtual void start();
    virtual void shutdown();

    virtual
    void sendAuctionMessage(std::shared_ptr<Auction> const & auction,
                            double timeLeftMs,
                            std::map<std::string, BidInfo> const & bidders);

    virtual
    void sendWinLossMessage(MatchedWinLoss const & event);

    virtual
    void sendLossMessage(std::string const & agent,
                         std::string const & id);

    virtual
    void sendCampaignEventMessage(std::string const & agent,
                                  MatchedCampaignEvent const & event);

    virtual
    void sendBidLostMessage(std::string const & agent,
                            std::shared_ptr<Auction> const & auction);

    virtual
    void sendBidDroppedMessage(std::string const & agent,
                               std::shared_ptr<Auction> const & auction);

    virtual
    void sendBidInvalidMessage(std::string const & agent,
                               std::string const & reason,
                               std::shared_ptr<Auction> const & auction);

    virtual
    void sendNoBudgetMessage(std::string const & agent,
                             std::shared_ptr<Auction> const & auction);

    virtual
    void sendTooLateMessage(std::string const & agent,
                            std::shared_ptr<Auction> const & auction);

    virtual
    void sendMessage(std::string const & agent,
                     std::string const & message);

    virtual
    void sendErrorMessage(std::string const & agent,
                          std::string const & error,
                          std::vector<std::string> const & payload);

    virtual
    void sendPingMessage(std::string const & agent,
                         int ping);

    Stats stats() const {
       return stats_;
    } 

private:
    Stats stats_;
    std::map<std::string, std::shared_ptr<BidderInterface>> bidderInterfaces;

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


    #define CALL_MEMBER_FN(object, pointer)  ((object)->*(pointer))

    template<typename... Args>
    BidderInterface*
    dispatchBidderInterface(
            const std::string &agent,
            const Id &auctionId,
            void (BidderInterface::*Func)(Args...),
            Args&& ...args)
    {
        const auto &agentInfo = router->agents[agent];
        const auto &agentConfig = agentInfo.config;
        return dispatchBidderInterface(agentConfig, auctionId, Func,
                                std::forward<Args>(args)...);
    }

    template<typename... Args>
    BidderInterface *
    dispatchBidderInterface(
            const std::shared_ptr<const AgentConfig> &agentConfig,
            const Id &auctionId,
            void (BidderInterface::*Func)(Args...),
            Args&& ...args)
    {
        auto selector = BidderInterfaceSelector::fromAgentConfig(
                              agentConfig, bidderInterfaces);
        auto iface = selector.pickBidderInterface(auctionId).get();
        CALL_MEMBER_FN(iface, Func)(std::forward<Args>(args)...);
        return iface;
    }

};

} // namespace RTBKIT
