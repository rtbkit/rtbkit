/* bid_price_calculator.h
   Eric Robert, 2 April 2014
   Copyright (c) 2011 Datacratic.  All rights reserved.
*/

#pragma once

#include "rtbkit/common/auction_events.h"
#include "rtbkit/core/router/router.h"
#include "soa/service/http_client.h"

namespace RTBKIT {

class Router;

struct BidPriceCalculator : public ServiceBase
{
    BidPriceCalculator(ServiceBase & parent,
                       std::string const & name = "bpc");

    BidPriceCalculator(std::shared_ptr<ServiceProxies> proxies = std::make_shared<ServiceProxies>(),
                       std::string const & name = "bpc");

    void init(Router * value);
    void start();
    void bindTcp();

    void sendAuctionMessage(std::shared_ptr<Auction> const & auction,
                            double timeLeftMs,
                            std::map<std::string, BidInfo> const & bidders,
                            const std::string &forwardHost = "");

    void sendWinMessage(std::string const & agent,
                        std::string const & id,
                        Amount price);

    void sendLossMessage(std::string const & agent,
                         std::string const & id);

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

    void send(std::shared_ptr<PostAuctionEvent> const & event);

    void useForwardingUri(const std::string &host, const std::string &resource);

private:
    void handlePostAuctionMessage(std::vector<std::string> const & items);

    Router * router;

    MessageLoop loop;
    TypedMessageSink<std::shared_ptr<PostAuctionEvent>> events;

    ZmqNamedEndpoint endpoint;
    std::shared_ptr<HttpClient> httpClient;
    std::pair<std::string, std::string> forwardInfo;
};

}

