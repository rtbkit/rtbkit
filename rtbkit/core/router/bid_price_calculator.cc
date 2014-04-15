/* bid_price_calculator.cc
   Eric Robert, 2 April 2014
   Copyright (c) 2011 Datacratic.  All rights reserved.
*/

#include "jml/db/persistent.h"
#include "rtbkit/common/messages.h"
#include "rtbkit/plugins/bid_request/openrtb_bid_request.h"
#include "rtbkit/openrtb/openrtb_parsing.h"
#include "bid_price_calculator.h"
#include "soa/service/http_client.h"

using namespace Datacratic;
using namespace RTBKIT;

namespace {
    DefaultDescription<OpenRTB::BidRequest> desc;
}


BidPriceCalculator::BidPriceCalculator(ServiceBase & parent,
                                       std::string const & name) :
    ServiceBase(name, parent),
    events(65536),
    endpoint(getZmqContext()),
    httpClient(nullptr)
{
}

BidPriceCalculator::BidPriceCalculator(std::shared_ptr<ServiceProxies> proxies,
                                       std::string const & name) :
    ServiceBase(name, proxies),
    events(65536),
    endpoint(getZmqContext()),
    httpClient(nullptr)
{
}

void BidPriceCalculator::init(Router * value) {
    router = value;

    registerServiceProvider(serviceName(), { "rtbBiddingService" });

    events.onEvent = std::bind<void>(&BidPriceCalculator::send,
                                    this,
                                    std::placeholders::_1);

    endpoint.messageHandler = std::bind(&BidPriceCalculator::handlePostAuctionMessage,
                                        this,
                                        std::placeholders::_1);
    endpoint.init(getServices()->config, ZMQ_XREP, serviceName() + "/events");
    loop.addSource("BidPriceCalculator::events", events);
}

void BidPriceCalculator::bindTcp() {
    endpoint.bindTcp(getServices()->ports->getRange("biddingService"));
}

void BidPriceCalculator::start() {
    loop.start();
}

void BidPriceCalculator::sendAuctionMessage(std::shared_ptr<Auction> const & auction,
                                            double timeLeftMs,
                                            std::map<std::string, BidInfo> const & bidders,
                                            const std::string &forwardHost) {
    for(auto & item : bidders) {
        auto & agent = item.first;
        auto & spots = item.second.imp;
        auto & info = router->agents[agent];
        const auto &config = info.config;
        WinCostModel wcm = auction->exchangeConnector->getWinCostModel(*auction, *info.config);
        if (config->external) {
            if (forwardInfo.first.empty()) {
                throw ML::Exception("Empty forward host");
            }

            OpenRTB::BidRequest openRtbRequest = toOpenRtb(*auction->request);
            StructuredJsonPrintingContext context;
            desc.printJson(&openRtbRequest, context);
            auto requestStr = context.output.toString();

            auto callbacks = std::make_shared<HttpClientSimpleCallbacks>(
                    [&](const HttpRequest &, HttpClientError errorCode,
                        int, const std::string &, const std::string &body)
                    {
                        if (errorCode != HttpClientError::NONE) {
                            auto toErrorString = [](HttpClientError code) -> std::string {
                                switch (code) {
                                    #define CASE(code) \
                                        case code: \
                                            return #code;
                                    CASE(HttpClientError::NONE)
                                    CASE(HttpClientError::UNKNOWN)
                                    CASE(HttpClientError::TIMEOUT)
                                    CASE(HttpClientError::HOST_NOT_FOUND)
                                    CASE(HttpClientError::COULD_NOT_CONNECT)
                                    #undef CASE
                                }
                                ExcCheck(false, "Invalid code path");
                                return "";
                            };
                            std::cerr << "Error requesting " << forwardHost
                                      << ": " << toErrorString(errorCode);
                          }
                          else {
                              std::cerr << "Response: " << body << std::endl;
                          }
                      });

            HttpRequest::Content reqContent { requestStr, "application/json" };
            RestParams headers { { "x-openrtb-version", "2.1" } };
            std::cerr << "Sending HTTP POST to: " << forwardInfo.first <<
                      " " << forwardInfo.second << std::endl;
            std::cerr << "Content " << reqContent.str << std::endl;
            httpClient->post(forwardInfo.second, callbacks, reqContent,
                            { } /* queryParams */, headers);
        }
        else {
            router->sendAgentMessage(agent,
                                     "AUCTION",
                                     auction->start,
                                     auction->id,
                                     info.getBidRequestEncoding(*auction),
                                     info.encodeBidRequest(*auction),
                                     spots.toJsonStr(),
                                     std::to_string(timeLeftMs),
                                     auction->agentAugmentations[agent],
                                     wcm.toJson());
        }
    }
}

void BidPriceCalculator::sendWinMessage(std::string const & agent,
                                        std::string const & id,
                                        Amount price) {
    router->sendAgentMessage(agent,
                             "WIN",
                             Date::now(),
                             "guaranteed",
                             id,
                             0,
                             price.toString());
}

void BidPriceCalculator::sendLossMessage(std::string const & agent,
                                         std::string const & id) {
    router->sendAgentMessage(agent,
                             "LOSS",
                             Date::now(),
                             "guaranteed",
                             id,
                             0,
                             Amount().toString());
}

void BidPriceCalculator::sendBidLostMessage(std::string const & agent,
                                            std::shared_ptr<Auction> const & auction) {
    router->sendAgentMessage(agent,
                             "LOST",
                             Date::now(),
                             "guaranteed",
                             auction->id,
                             0,
                             Amount().toString());
/*
-                    this->sendBidResponse(it->first,
-                                          info,
-                                          BS_LOSTBID,
-                                          this->getCurrentTime(),
-                                          "guaranteed", id);
-
+                    bpc->sendLostBidMessage(it->first, inFlight[id]);
*/
}

void BidPriceCalculator::sendBidDroppedMessage(std::string const & agent,
                                               std::shared_ptr<Auction> const & auction) {
    router->sendAgentMessage(agent,
                             "DROPPEDBID",
                             Date::now(),
                             "guaranteed",
                             auction->id,
                             0,
                             Amount().toString());
/*
-                        this->sendBidResponse(agent,
-                                              info,
-                                              BS_DROPPEDBID,
-                                              this->getCurrentTime(),
-                                              "guaranteed",
-                                              auctionId,
-                                              0, Amount(),
-                                              auctionInfo.auction.get());
+                        bpc->sendDroppedBid(agent, *auctionInfo.auction)
*/
}

void BidPriceCalculator::sendBidInvalidMessage(std::string const & agent,
                                               std::string const & reason,
                                               std::shared_ptr<Auction> const & auction) {
    router->sendAgentMessage(agent,
                             "INVALID",
                             Date::now(),
                             reason,
                             auction->id,
                             0,
                             Amount().toString());
/*
-            this->sendBidResponse
-                (agent, info, BS_INVALID, this->getCurrentTime(),
-                 formatted, auctionId,
-                 i, Amount(),
-                 auctionInfo.auction.get(),
-                 biddata, Json::Value(),
-                 auctionInfo.auction->agentAugmentations[agent]);
+            bpc->sendInvalidBid(agent, formatted, *auctionInfo.auction);
*/
}

void BidPriceCalculator::sendNoBudgetMessage(std::string const & agent,
                                             std::shared_ptr<Auction> const & auction) {
    router->sendAgentMessage(agent,
                             "NOBUDGET",
                             Date::now(),
                             "guaranteed",
                             auction->id,
                             0,
                             Amount().toString());
/*
-            this->sendBidResponse(agent, info, BS_NOBUDGET,
-                    this->getCurrentTime(),
-                    "guaranteed", auctionId, 0, Amount(),
-                    auctionInfo.auction.get(),
-                    biddata, meta, agentAugmentations);
+            bpc->sendNoBudget(agent, auctionInfo.auction.get(), biddata, meta);
*/
}

void BidPriceCalculator::sendTooLateMessage(std::string const & agent,
                                            std::shared_ptr<Auction> const & auction) {
    router->sendAgentMessage(agent,
                             "TOOLATE",
                             Date::now(),
                             "guaranteed",
                             auction->id,
                             0,
                             Amount().toString());

/*
-            case Auction::WinLoss::LOSS:    status = BS_LOSS;     break;
-            case Auction::WinLoss::TOOLATE: status = BS_TOOLATE;  break;
-            case Auction::WinLoss::INVALID: status = BS_INVALID;  break;

-            const string& agentAugmentations =
-                auctionInfo.auction->agentAugmentations[agent];
-
-            this->sendBidResponse(agent, info, status,
-                    this->getCurrentTime(),
-                    "guaranteed", auctionId, 0, Amount(),
-                    auctionInfo.auction.get(),
-                    biddata, meta, agentAugmentations);
*/

/*
-            string confidence = "guaranteed";
-
-            //cerr << fName << "sending agent message of type " << msg << endl;
-            sendBidResponse(response.agent, info, bidStatus,
-                            this->getCurrentTime(),
-                            confidence, auctionId,
-                            0, Amount(),
-                            auction.get(),
-                            response.bidData,
-                            response.meta,
-                            auction->agentAugmentations[response.agent]);
*/
}

void BidPriceCalculator::sendMessage(std::string const & agent,
                                     std::string const & message) {
    router->sendAgentMessage(agent,
                             message,
                             Date::now());
}

void BidPriceCalculator::sendErrorMessage(std::string const & agent,
                                          std::string const & error,
                                          std::vector<std::string> const & payload) {
    router->sendAgentMessage(agent,
                             "ERROR",
                             Date::now(),
                             error,
                             payload);
}

void BidPriceCalculator::sendPingMessage(std::string const & agent,
                                         int ping) {
    if(ping == 0) {
        router->sendAgentMessage(agent,
                                 "PING0",
                                 Date::now(),
                                 "null");
    }
    else {
        router->sendAgentMessage(agent,
                                 "PING1",
                                 Date::now(),
                                 "null");
    }
}

void BidPriceCalculator::send(std::shared_ptr<PostAuctionEvent> const & event) {
    /*
    router->sendAgentMessage(agent,
                             event->label,
                             Date::now(),
                             "guaranteed",
                             event->auctionId,
                             event->adSpotId,
                             event->winPrice.toString());*/
}

void BidPriceCalculator::handlePostAuctionMessage(std::vector<std::string> const & items) {
    std::string key = "messages." + items[1];
    recordHit(key);

    auto event = std::make_shared<PostAuctionEvent>(
                    ML::DB::reconstituteFromString<PostAuctionEvent>(items.at(2)));
    if(items[1] != event->label) {
        key += "." + event->label;
        recordHit(key);
    }
    else {
        event->label = items[1];
    }

    events.push(event);
}

void BidPriceCalculator::useForwardingUri(const std::string &host,
                                   const std::string &resource) {
    forwardInfo = { host, resource };
    httpClient.reset(new HttpClient(host));
    loop.addSource("BidPriceCalculator::httpClient", httpClient);
}
