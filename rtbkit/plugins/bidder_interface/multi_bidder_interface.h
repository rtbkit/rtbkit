/* multi_bidder_interface.h
   Mathieu Stefani, 17 July 2014
   Copyright (c) 2014 Datacratic.  All rights reserved.
   
   The Multi Bidder Interface class
*/

#pragma once

#include "rtbkit/common/bidder_interface.h"

namespace RTBKIT {

struct MultiBidderInterface : public BidderInterface {

    struct InterfaceStats {
        size_t auctionsSent;
        size_t winsSent;
    };

    struct Stat {
        size_t totalAuctions;
        std::map<std::string, InterfaceStats> interfacesStats;
    };

    typedef std::map<std::string, Stat> Stats;

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

};

} // namespace RTBKIT
