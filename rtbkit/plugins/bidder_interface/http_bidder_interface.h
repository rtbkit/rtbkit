/* http_bidder_interface.h
   Eric Robert, 2 April 2014
   Copyright (c) 2011 Datacratic.  All rights reserved.
*/

#pragma once

#include "rtbkit/common/bidder_interface.h"
#include "soa/service/http_client.h"

namespace RTBKIT {

struct Bids;

struct HttpBidderInterface : public BidderInterface
{
    HttpBidderInterface(std::string name = "bidder",
                        std::shared_ptr<ServiceProxies> proxies = std::make_shared<ServiceProxies>(),
                        Json::Value const & json = Json::Value());

    void start();
    void sendAuctionMessage(std::shared_ptr<Auction> const & auction,
                            double timeLeftMs,
                            std::map<std::string, BidInfo> const & bidders);

    void sendWinLossMessage(MatchedWinLoss const & event);

    void sendLossMessage(std::string const & agent,
                         std::string const & id);

    void sendCampaignEventMessage(std::string const & agent,
                                  MatchedCampaignEvent const & event);

    void sendBidLostMessage(std::string const & agent,
                            std::shared_ptr<Auction> const & auction);

    void sendBidDroppedMessage(std::string const & agent,
                               std::shared_ptr<Auction> const & auction);

    void sendBidInvalidMessage(std::string const & agent,
                               std::string const & reason,
                               std::shared_ptr<Auction> const & auction);

    void sendNoBudgetMessage(std::string const & agent,
                             std::shared_ptr<Auction> const & auction);

    void sendTooLateMessage(std::string const & agent,
                            std::shared_ptr<Auction> const & auction);

    void sendMessage(std::string const & agent,
                     std::string const & message);

    void sendErrorMessage(std::string const & agent,
                          std::string const & error,
                          std::vector<std::string> const & payload);

    void sendPingMessage(std::string const & agent,
                         int ping);

    virtual void tagRequest(OpenRTB::BidRequest &request,
                            const std::map<std::string, BidInfo> &bidders) const;

private:
    bool prepareRequest(OpenRTB::BidRequest &request,
                        const RTBKIT::BidRequest &originalRequest,
                        const std::map<std::string, BidInfo> &bidders) const;
    void submitBids(const std::string &agent, Id auctionId,
                         const Bids &bids, WinCostModel wcm);
    MessageLoop loop;
    std::shared_ptr<HttpClient> httpClient;
    std::string host;
    std::string path;
};

}

