/* http_bidder_interface.cc
   Eric Robert, 2 April 2014
   Copyright (c) 2011 Datacratic.  All rights reserved.
*/

#include "http_bidder_interface.h"
#include "jml/db/persistent.h"
#include "soa/service/http_client.h"
#include "soa/utils/generic_utils.h"
#include "rtbkit/common/messages.h"
#include "rtbkit/plugins/bid_request/openrtb_bid_request.h"
#include "rtbkit/openrtb/openrtb_parsing.h"
#include "rtbkit/core/router/router.h"

using namespace Datacratic;
using namespace RTBKIT;

namespace {
    DefaultDescription<OpenRTB::BidRequest> desc;

}


HttpBidderInterface::HttpBidderInterface(std::string name,
                                         std::shared_ptr<ServiceProxies> proxies,
                                         Json::Value const & json)
        : BidderInterface(proxies, name) {
    host = json["host"].asString();
    path = json["path"].asString();
    httpClient.reset(new HttpClient(host));
    loop.addSource("HttpBidderInterface::httpClient", httpClient);
}

void HttpBidderInterface::start() {
    loop.start();
}


void HttpBidderInterface::sendAuctionMessage(std::shared_ptr<Auction> const & auction,
                                             double timeLeftMs,
                                             std::map<std::string, BidInfo> const & bidders) {
    using namespace std;
    for(auto & item : bidders) {
        auto & agent = item.first;
        auto & info = router->agents[agent];
        const auto &config = info.config;
        BidRequest originalRequest = *auction->request;
        WinCostModel wcm = auction->exchangeConnector->getWinCostModel(*auction, *info.config);

        OpenRTB::BidRequest openRtbRequest = toOpenRtb(originalRequest);
        bool ok = prepareRequest(openRtbRequest, originalRequest, bidders);
        /* If we took too much time processing the request, then we don't send it.  */
        if (!ok) {
            return;
        }
        StructuredJsonPrintingContext context;
        desc.printJson(&openRtbRequest, context);
        auto requestStr = context.output.toString();

        /* We need to capture by copy inside the lambda otherwise we might get
           a dangling reference if we go out of scope before receiving the http response
        */
        auto callbacks = std::make_shared<HttpClientSimpleCallbacks>(
                [=](const HttpRequest &, HttpClientError errorCode,
                    int statusCode, const std::string &, std::string &&body)
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
                        router->throwException("http", "Error requesting %s: %s",
                                               host.c_str(),
                                               toErrorString(errorCode).c_str());
                      }
                      else if (statusCode == 200) {
                        // cerr << "Response: " << body << endl;
                         OpenRTB::BidResponse response;
                         ML::Parse_Context context("payload",
                               body.c_str(), body.size());
                         StreamingJsonParsingContext jsonContext(context);
                         static DefaultDescription<OpenRTB::BidResponse> respDesc;
                         respDesc.parseJson(&response, jsonContext);

                         for (const auto &seatbid: response.seatbid) {
                             Bids bids;

                             for (const auto &bid: seatbid.bid) {
                                 Bid theBid;

                                 int crid = bid.crid.toInt();
                                 int creativeIndex = indexOf(config->creatives,
                                     &Creative::id, crid);

                                 if (creativeIndex == -1) {
                                     router->throwException("http.response",
                                        "Unknown creative id: %d", crid);
                                 }

                                 theBid.creativeIndex = creativeIndex;
                                 theBid.price = USD_CPM(bid.price.val);
                                 theBid.priority = 0.0;

                                 int spotIndex = indexOf(openRtbRequest.imp,
                                                        &OpenRTB::Impression::id, bid.impid);
                                 if (spotIndex == -1) {
                                      router->throwException("http.response",
                                         "Unknown impression id: %s",
                                         bid.impid.toString().c_str());
                                 }

                                 theBid.spotIndex = spotIndex;

                                 bids.push_back(std::move(theBid));
                             }
                             submitBids(agent, auction->id, bids, wcm);
                         }
                     }
                }
        );

        HttpRequest::Content reqContent { requestStr, "application/json" };
        RestParams headers { { "x-openrtb-version", "2.1" } };
       // std::cerr << "Sending HTTP POST to: " << host << " " << path << std::endl;
       // std::cerr << "Content " << reqContent.str << std::endl;

        httpClient->post(path, callbacks, reqContent,
                         { } /* queryParams */, headers);
    }
}


void HttpBidderInterface::sendWinLossMessage(MatchedWinLoss const & event) {

}

void HttpBidderInterface::sendLossMessage(std::string const & agent,
                                          std::string const & id) {

}

void HttpBidderInterface::sendCampaignEventMessage(std::string const & agent,
                                                   MatchedCampaignEvent const & event) {

}

void HttpBidderInterface::sendBidLostMessage(std::string const & agent,
                                             std::shared_ptr<Auction> const & auction) {
}

void HttpBidderInterface::sendBidDroppedMessage(std::string const & agent,
                                                std::shared_ptr<Auction> const & auction) {
}

void HttpBidderInterface::sendBidInvalidMessage(std::string const & agent,
                                                std::string const & reason,
                                                std::shared_ptr<Auction> const & auction) {
}

void HttpBidderInterface::sendNoBudgetMessage(std::string const & agent,
                                              std::shared_ptr<Auction> const & auction) {
}

void HttpBidderInterface::sendTooLateMessage(std::string const & agent,
                                             std::shared_ptr<Auction> const & auction) {
}

void HttpBidderInterface::sendMessage(std::string const & agent,
                                      std::string const & message) {
}

void HttpBidderInterface::sendErrorMessage(std::string const & agent,
                                           std::string const & error,
                                           std::vector<std::string> const & payload) {
}

void HttpBidderInterface::sendPingMessage(std::string const & agent,
                                          int ping) {
    ExcCheck(ping == 0 || ping == 1, "Bad PING level, must be either 0 or 1");

    auto encodeDate = [](Date date) {
        return ML::format("%.5f", date.secondsSinceEpoch());
    };

    const std::string sentTime = encodeDate(Date::now());
    const std::string receivedTime = sentTime;
    const std::string pong = (ping == 0 ? "PONG0" : "PONG1");
    std::vector<std::string> message { agent, pong, sentTime, receivedTime };
    router->handleAgentMessage(message);
}

void HttpBidderInterface::tagRequest(OpenRTB::BidRequest &request,
                                     const std::map<std::string, BidInfo> &bidders) const
{

    for (const auto &bidder: bidders) {
        const auto &agentConfig = bidder.second.agentConfig;
        const auto &spots = bidder.second.imp;

        for (const auto &spot: spots) {
            const int adSpotIndex = spot.first;
            ExcCheck(adSpotIndex >= 0 && adSpotIndex < request.imp.size(),
                     "adSpotIndex out of range");
            auto &imp = request.imp[adSpotIndex];
            auto &ext = imp.ext;

            ext["allowed_ids"].append(agentConfig->externalId);
        }

    }

}

bool HttpBidderInterface::prepareRequest(OpenRTB::BidRequest &request,
                                         const RTBKIT::BidRequest &originalRequest,
                                         const std::map<std::string, BidInfo> &bidders) const {
    tagRequest(request, bidders);

    // We update the tmax value before sending the BidRequest to substract our processing time
    double processingTimeMs = originalRequest.timestamp.secondsUntil(Date::now()) * 1000;
    int oldTmax = request.tmax.value();
    int newTmax = oldTmax - static_cast<int>(std::round(processingTimeMs));
    if (newTmax <= 0) {
        return false;
    }
#if 0
    std::cerr << "old tmax = " << oldTmax << std::endl
              << "new tmax = " << newTmax << std::endl;
#endif
    ExcCheck(newTmax <= oldTmax, "Wrong tmax calculation");
    request.tmax.val = newTmax;
    return true;
}

void HttpBidderInterface::submitBids(const std::string &agent, Id auctionId,
                                     const Bids &bids, WinCostModel wcm)
{
     Json::FastWriter writer;
     std::vector<std::string> message { agent, "BID" };
     message.push_back(auctionId.toString());
     std::string bidsStr = writer.write(bids.toJson());
     std::string wcmStr = writer.write(wcm.toJson());
     message.push_back(std::move(bidsStr));
     message.push_back(std::move(wcmStr));

     router->doBid(message);
}

//
// factory
//

namespace {

struct AtInit {
    AtInit()
    {
        BidderInterface::registerFactory("http", [](std::string const & name , std::shared_ptr<ServiceProxies> const & proxies, Json::Value const & json) {
            return new HttpBidderInterface(name, proxies, json);
        });
    }
} atInit;

}

