/* http_bidder_interface.h                                         -*- C++ -*-
   Eric Robert, 2 April 2014
   Copyright (c) 2011 Datacratic.  All rights reserved.
*/

#pragma once

#include "rtbkit/common/bidder_interface.h"
#include "soa/service/http_client.h"
#include "soa/service/logs.h"

namespace RTBKIT {

struct Bids;

struct HttpBidderInterface : public BidderInterface
{
    HttpBidderInterface(std::string serviceName = "bidderService",
                        std::shared_ptr<ServiceProxies> proxies = std::make_shared<ServiceProxies>(),
                        Json::Value const & json = Json::Value());
    ~HttpBidderInterface();

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

    virtual void tagRequest(OpenRTB::BidRequest &request,
                            const std::map<std::string, BidInfo> &bidders) const;

    static Logging::Category print;
    static Logging::Category error;
    static Logging::Category trace;


private:

    struct AgentBidsInfo {
        std::shared_ptr<const AgentConfig> agentConfig;
        std::string agentName;
        Id auctionId;
        Bids bids;
        WinCostModel wcm;
    };

    typedef std::map<std::string, AgentBidsInfo> AgentBids;

    MessageLoop loop;
    std::shared_ptr<HttpClient> httpClientRouter;
    std::shared_ptr<HttpClient> httpClientAdserverWins;
    std::shared_ptr<HttpClient> httpClientAdserverEvents;
    std::shared_ptr<HttpClient> httpClientAdserverErrors;

    enum Format {
        FMT_STANDARD,
        FMT_DATACRATIC,
    };
    static Format readFormat(const std::string& fmt);

    std::string routerHost;
    std::string routerPath;
    Format routerFormat;

    std::string adserverHost;

    uint16_t adserverWinPort;
    std::string adserverWinPath;
    Format adserverWinFormat;

    uint16_t adserverEventPort;
    std::string adserverEventPath;
    Format adserverEventFormat;

    uint16_t adserverErrorPort;
    std::string adserverErrorPath;
    Format adserverErrorFormat;

    void submitBids(AgentBids &info);

    bool prepareRequest(OpenRTB::BidRequest &request,
                        const RTBKIT::BidRequest &originalRequest,
                        const std::shared_ptr<Auction> &auction,
                        const std::map<std::string, BidInfo> &bidders) const;
    bool prepareStandardRequest(OpenRTB::BidRequest &request,
                                const RTBKIT::BidRequest &originalRequest,
                                const std::shared_ptr<Auction> &auction,
                                const std::map<std::string, BidInfo> &bidders) const;
    bool prepareDatacraticRequest(OpenRTB::BidRequest &request,
                                  const RTBKIT::BidRequest &originalRequest,
                                  const std::shared_ptr<Auction> &auction,
                                  const std::map<std::string, BidInfo> &bidders) const;

    void sendBidErrorMessage(
            const std::shared_ptr<const AgentConfig>& agentConfig,
            std::string const & agent,
            std::shared_ptr<Auction> const & auction,
            std::string const & type, std::string const & reason = "");

    void injectBids(const std::string &agent, Id auctionId,
                    const Bids &bids, WinCostModel wcm);

    void recordError(const std::string &key);

};

}

