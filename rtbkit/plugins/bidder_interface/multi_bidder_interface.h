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

        void dump(std::ostream &stream, size_t indent = 4) const {

            #define CALL_MEMBER_VAR(object, field) ((object)->*(field))

            struct Field {
                size_t InterfaceStats::*ptr;
                const char * const name;
            } fields[] = {
                { &InterfaceStats::auctions, "Auctions" },
                { &InterfaceStats::wins, "Wins" },
                { &InterfaceStats::loss, "Loss" },
                { &InterfaceStats::campaignEvents, "Campaign events" },
                { &InterfaceStats::noBudget, "No budget" },
                { &InterfaceStats::tooLate, "Too late" }
            };

            size_t const totalFields = sizeof fields / sizeof *fields;
            for (size_t i = 0; i < totalFields; ++i) {
                Field const *f = fields + i;
                stream << f->name << ": "
                          << CALL_MEMBER_VAR(this, f->ptr)
                          << std::endl;
            }

            stream << std::endl;
            #undef CALL_MEMBER_VAR
        }


    };


    struct Stats {
        size_t incr(const std::shared_ptr<BidderInterface> &interface,
                    size_t InterfaceStats::*member)
        {
            return incr(interface->interfaceName(), member);
        }

        size_t incr(const std::string &interfaceName,
                    size_t InterfaceStats::*member)
        {
            auto& stat = interfacesStats[interfaceName];
            auto& value = stat.*member;
            return ++value;
        }

        void dump(std::ostream &stream) const
        {
            auto header = [&](const std::string &str) {
                std::string text = "Stats for " + str;
                stream << text << std::endl;
                stream << std::string(text.size(), '-') << std::endl;
            };

            for (const auto &stat: interfacesStats)
            {
                header(stat.first);
                stat.second.dump(stream);
            }
        }

    private:
        std::map<std::string, InterfaceStats> interfacesStats;
    };

    MultiBidderInterface(
        const std::string &serviceName = "bidderService",
        std::shared_ptr<ServiceProxies> proxies = std::make_shared<ServiceProxies>(),
        const Json::Value &config = Json::Value()
        );

    void init(AgentBridge *value, Router * router = nullptr);
    void start();
    void shutdown();

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

    void registerLoopMonitor(LoopMonitor *monitor) const;

    Stats stats() const {
        return stats_;
    }


private:
    std::map<std::string, std::shared_ptr<BidderInterface>> bidderInterfaces;
    std::shared_ptr<BidderInterface> findInterface(
                const std::string &name,
                const std::string &agent) {

        if (name.empty()) {
            throw ML::Exception("Empty bidderInterface for agent '%s'",
                                agent.c_str());
        }

        auto it = bidderInterfaces.find(name);
        if (it == std::end(bidderInterfaces)) {
            throw ML::Exception("Unknown interface '%s' for agent '%s'",
                                name.c_str(), agent.c_str());
        }

        return it->second;
    }

    #define CALL_MEMBER_FN(object, pointer)  ((object)->*(pointer))

    template<typename Func, typename... Args>
    BidderInterface *
    dispatchBidderInterface(
            const std::string &agentName,
            const std::shared_ptr<const AgentConfig> &agentConfig,
            Func func,
            Args&& ...args)
    {

        ExcAssert(agentConfig != nullptr);
        auto iface = findInterface(agentConfig->bidderInterface, agentName);

        CALL_MEMBER_FN(iface.get(), func)(std::forward<Args>(args)...);

        return iface.get();
    }

    template<typename Func, typename... Args>
    void
    dispatchAllInterfaces(Func func, Args&& ...args)
    {
        for (const auto &iface: bidderInterfaces) {
            CALL_MEMBER_FN(iface.second.get(), func)(std::forward<Args>(args)...);
        }
    }

    #undef CALL_MEMBER_FN

    Stats stats_;


};

} // namespace RTBKIT
