/* http_bidder_interface.cc
   Eric Robert, 2 April 2014
   Copyright (c) 2011 Datacratic.  All rights reserved.
*/

#include "http_bidder_interface.h"
#include "jml/db/persistent.h"
#include "soa/service/http_client.h"
#include "rtbkit/common/messages.h"
#include "rtbkit/plugins/bid_request/openrtb_bid_request.h"
#include "rtbkit/openrtb/openrtb_parsing.h"
#include "rtbkit/core/router/router.h"

using namespace Datacratic;
using namespace RTBKIT;

namespace {
    DefaultDescription<OpenRTB::BidRequest> desc;

    void tagRequest(OpenRTB::BidRequest &request,
                    const std::map<std::string, BidInfo> &bidders)
    {

        for (const auto &bidder: bidders) {
            const auto &agentConfig = bidder.second.agentConfig;
            const auto &spots = bidder.second.imp;
            const auto &creatives = agentConfig->creatives;

            Json::Value creativesValue(Json::arrayValue);
            for (const auto &spot: spots) {
                const int adSpotIndex = spot.first;
                ExcCheck(adSpotIndex >= 0 && adSpotIndex < request.imp.size(),
                         "adSpotIndex out of range");
                auto &imp = request.imp[adSpotIndex];
                auto &ext = imp.ext;
                for (int creativeIndex: spot.second) {
                    ExcCheck(creativeIndex >= 0 && creativeIndex < creatives.size(),
                             "creativeIndex out of range");
                    const int creativeId = creatives[creativeIndex].id;
                    creativesValue.append(creativeId);
                }

                ext["allowed_ids"][std::to_string(agentConfig->externalId)] =
                    std::move(creativesValue);

            }

        }

    }
}

HttpBidderInterface::HttpBidderInterface(std::string name,
                                         std::shared_ptr<ServiceProxies> proxies,
                                         Json::Value const & json) {
    host = json["host"].asString();
    path = json["path"].asString();
    httpClient.reset(new HttpClient(host));
    loop.addSource("HttpBidderInterface::httpClient", httpClient);
}

void HttpBidderInterface::sendAuctionMessage(std::shared_ptr<Auction> const & auction,
                                             double timeLeftMs,
                                             std::map<std::string, BidInfo> const & bidders) {
    using namespace std;
    for(auto & item : bidders) {
        auto & agent = item.first;
        auto & info = router->agents[agent];
        WinCostModel wcm = auction->exchangeConnector->getWinCostModel(*auction, *info.config);

        OpenRTB::BidRequest openRtbRequest = toOpenRtb(*auction->request);
        tagRequest(openRtbRequest, bidders);
        StructuredJsonPrintingContext context;
        desc.printJson(&openRtbRequest, context);
        auto requestStr = context.output.toString();

        /* We need to capture by copy inside the lambda otherwise we might get
           a dangling reference if we go out of scope before receiving the http response
        */
        auto callbacks = std::make_shared<HttpClientSimpleCallbacks>(
                [=](const HttpRequest &, HttpClientError errorCode,
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
                        cerr << "Error requesting " << host
                                  << ": " << toErrorString(errorCode);
                      }
                      else {
                          std::cerr << "Response: " << body << std::endl;
                          OpenRTB::BidResponse response;
                          ML::Parse_Context context("payload",
                                  body.c_str(), body.size());
                          StreamingJsonParsingContext jsonContext(context);
                          DefaultDescription<OpenRTB::BidResponse> respDesc;
                          respDesc.parseJson(&response, jsonContext);

                          for (const auto &seatbid: response.seatbid) {
                              Bids bids;

                              for (const auto &bid: seatbid.bid) {
                                Bid theBid;
                                theBid.creativeIndex = bid.crid.toInt();
                                theBid.price = USD_CPM(bid.price.val);
                                theBid.priority = 0.0;

                                /* Looping over the impressions to find the corresponding
                                   adSpotIndex
                                */
                                auto impIt = find_if(
                                    begin(openRtbRequest.imp), end(openRtbRequest.imp),
                                    [&](const OpenRTB::Impression &imp) {
                                        return imp.id == bid.impid;
                                });
                                if (impIt == end(openRtbRequest.imp)) {
                                    throw ML::Exception(ML::format(
                                        "Unknown impression id: %s", bid.impid.toString()));
                                }

                                auto spotIndex = distance(begin(openRtbRequest.imp),
                                                                impIt);
                                theBid.spotIndex = spotIndex;

                                bids.push_back(std::move(theBid));
                             }

                             Json::FastWriter writer;
                             std::vector<std::string> message { agent, "BID" };
                             message.push_back(auction->id.toString());
                             std::string bidsStr = writer.write(bids.toJson());
                             std::string wcmStr = writer.write(wcm.toJson());
                             message.push_back(std::move(bidsStr));
                             message.push_back(std::move(wcmStr));

                             router->doBid(message);

                          }
                      }
                  });

        HttpRequest::Content reqContent { requestStr, "application/json" };
        RestParams headers { { "x-openrtb-version", "2.1" } };
        std::cerr << "Sending HTTP POST to: " << host << " " << path << std::endl;
        std::cerr << "Content " << reqContent.str << std::endl;

        httpClient->post(path, callbacks, reqContent,
                         { } /* queryParams */, headers);
    }
}

void HttpBidderInterface::sendWinMessage(std::string const & agent,
                                         std::string const & id,
                                         Amount price) {
}

void HttpBidderInterface::sendWinMessage(std::string const & agent,
                                         Amount price,
                                         FinishedInfo const & event) {
}

void HttpBidderInterface::sendLateWinMessage(std::string const & agent,
                                             Amount price,
                                             FinishedInfo const & event) {
}

void HttpBidderInterface::sendLossMessage(std::string const & agent,
                                          std::string const & id) {
}

void HttpBidderInterface::sendLossMessage(std::string const & agent,
                                          FinishedInfo const & event) {
}

void HttpBidderInterface::sendCampaignEventMessage(std::string const & agent,
                                                   std::string const & label,
                                                   FinishedInfo const & event) {
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
}

void HttpBidderInterface::send(std::shared_ptr<PostAuctionEvent> const & event) {
}

//
// factory
//

struct AtInit {
    AtInit()
    {
        BidderInterface::registerFactory("http", [](std::string const & name , std::shared_ptr<ServiceProxies> const & proxies, Json::Value const & json) {
            return new HttpBidderInterface(name, proxies, json);
        });
    }
} atInit;

