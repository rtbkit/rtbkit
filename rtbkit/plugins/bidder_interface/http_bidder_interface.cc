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
    for(auto & item : bidders) {
        auto & agent = item.first;
        auto & info = router->agents[agent];
        WinCostModel wcm = auction->exchangeConnector->getWinCostModel(*auction, *info.config);

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
                        std::cerr << "Error requesting " << host
                                  << ": " << toErrorString(errorCode);
                      }
                      else {
                          std::cerr << "Response: " << body << std::endl;
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

void HttpBidderInterface::sendLossMessage(std::string const & agent,
                                          std::string const & id) {
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

