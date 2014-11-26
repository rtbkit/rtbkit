/* http_bidder_interface.cc
   Eric Robert, 2 April 2014
   Copyright (c) 2011 Datacratic.  All rights reserved.
*/

#include "http_bidder_interface.h"
#include "jml/db/persistent.h"
#include "soa/service/http_client.h"
#include "soa/utils/generic_utils.h"
#include "rtbkit/common/messages.h"
#include "rtbkit/plugins/bid_request/openrtb_bid_request_parser.h"
#include "rtbkit/openrtb/openrtb_parsing.h"
#include "rtbkit/core/router/router.h"

using namespace Datacratic;
using namespace RTBKIT;

namespace {
    DefaultDescription<OpenRTB::BidRequest> desc;

    std::string httpErrorString(HttpClientError code)  {
        switch (code) {
            #define CASE(code) \
                case code: \
                    return #code;
            CASE(HttpClientError::None)
            CASE(HttpClientError::Unknown)
            CASE(HttpClientError::Timeout)
            CASE(HttpClientError::HostNotFound)
            CASE(HttpClientError::CouldNotConnect)
            CASE(HttpClientError::SendError)
            CASE(HttpClientError::RecvError)
            #undef CASE
        }
        ExcCheck(false, "Invalid code path");
        return "";
    }
}

namespace RTBKIT {

Logging::Category HttpBidderInterface::print("HttpBidderInterface");
Logging::Category HttpBidderInterface::error("HttpBidderInterface Error", HttpBidderInterface::print);
Logging::Category HttpBidderInterface::trace("HttpBidderInterface Trace", HttpBidderInterface::print);

}

HttpBidderInterface::HttpBidderInterface(std::string serviceName,
                                         std::shared_ptr<ServiceProxies> proxies,
                                         Json::Value const & json)
        : BidderInterface(proxies, serviceName) {

    int routerHttpActiveConnections = 0;
    int adserverHttpActiveConnections = 0;

    try {
        const auto& router = json["router"];
        const auto& adserver = json["adserver"];

        routerHost
            = router["host"].asString();
        routerPath
            = router["path"].asString();
        routerHttpActiveConnections
            = router.get("httpActiveConnections", 4).asInt();

        adserverHost
            = adserver["host"].asString();
        adserverWinPort
            = adserver["winPort"].asInt();
        adserverEventPort
            = adserver["eventPort"].asInt();
        adserverHttpActiveConnections
            = adserver.get("httpActiveConnections", 4).asInt();
    } catch (const std::exception & e) {
        THROW(error) << "configuration file is invalid" << std::endl
                   << "usage : " << std::endl
                   << "{" << std::endl << "\t\"router\" : {" << std::endl
                   << "\t\t\"host\" : <string : hostname with port>" << std::endl  
                   << "\t\t\"path\" : <string : resource name>" << std::endl
                   << "\t\t\"httpActiveConnections\" : <int : concurrent connections>"
                   << std::endl
                   << "\t\t"
                   << "\t}" << std::endl << "\t{" << std::endl 
                   << "\t{" << std::endl << "\t\"adserver\" : {" << std::endl
                   << "\t\t\"host\" : <string : hostname>" << std::endl  
                   << "\t\t\"winPort\" : <int : winPort>" << std::endl  
                   << "\t\t\"eventPort\" : <int eventPort>" << std::endl
                   << "\t\t\"httpActiveConnections\" : <int : concurrent connections>"
                   << std::endl
                   << "\t}" << std::endl << "}";
    }

    httpClientRouter.reset(new HttpClient(routerHost, routerHttpActiveConnections));
    /* We do not want curl to add an extra "Expect: 100-continue" HTTP header
     * and then pay the cost of an extra HTTP roundtrip. Thus we remove this
     * header
     */
    httpClientRouter->sendExpect100Continue(false);
    loop.addSource("HttpBidderInterface::httpClientRouter", httpClientRouter);

    std::string winHost = adserverHost + ':' + std::to_string(adserverWinPort);
    httpClientAdserverWins.reset(new HttpClient(winHost, adserverHttpActiveConnections));
    httpClientAdserverWins->sendExpect100Continue(false);
    loop.addSource("HttpBidderInterface::httpClientAdserverWins", httpClientAdserverWins);

    std::string eventHost = adserverHost + ':' + std::to_string(adserverEventPort);
    httpClientAdserverEvents.reset(new HttpClient(eventHost, adserverHttpActiveConnections));
    httpClientAdserverEvents->sendExpect100Continue(false);
    loop.addSource("HttpBidderInterface::httpClientAdserverEvents", httpClientAdserverEvents);

    loop.addPeriodic("HttpBidderInterface::reportQueues", 1.0, [=](uint64_t) {
        recordLevel(httpClientRouter->queuedRequests(), "queuedRequests");
    });

}

HttpBidderInterface::~HttpBidderInterface()
{
    shutdown();
}

void HttpBidderInterface::start() {
    loop.start();
}

void HttpBidderInterface::shutdown() {
    loop.shutdown();
}


void HttpBidderInterface::sendAuctionMessage(std::shared_ptr<Auction> const & auction,
                                             double timeLeftMs,
                                             std::map<std::string, BidInfo> const & bidders) {
    using namespace std;

    auto findAgent = [=](uint64_t externalId)
        -> pair<string, shared_ptr<const AgentConfig>> {

        auto it =
        find_if(begin(bidders), end(bidders),
                [&](const pair<string, BidInfo> &bidder)
        {
            std::string agent = bidder.first;
            const auto &info = router->agents[agent];
            return info.config->externalId == externalId;
        });

        if (it == end(bidders)) {
            return make_pair("", nullptr);
        }

        return make_pair(it->first, it->second.agentConfig);

    };

    BidRequest & originalRequest = *auction->request;
    std::shared_ptr<OpenRTBBidRequestParser> parser = OpenRTBBidRequestParser::openRTBBidRequestParserFactory("2.1");

    OpenRTB::BidRequest openRtbRequest = parser->toBidRequest(originalRequest);
    bool ok = prepareRequest(openRtbRequest, originalRequest, auction, bidders);
    /* If we took too much time processing the request, then we don't send it.  */
    if (!ok) {
        return;
    }
    StructuredJsonPrintingContext context;
    desc.printJson(&openRtbRequest, context);
    auto requestStr = context.output.toString();


    Date sentResponseTime = Date::now();
    /* We need to capture by copy inside the lambda otherwise we might get
       a dangling reference if we go out of scope before receiving the http response
    */
    auto callbacks = std::make_shared<HttpClientSimpleCallbacks>(
            [=](const HttpRequest &, HttpClientError errorCode,
                int statusCode, const std::string &, std::string &&body)
            {
                Date responseReceivedTime = Date::now();
                const double responseTime = responseReceivedTime.secondsSince(sentResponseTime);
                router->recordOutcome(1000.0 * responseTime, "httpResponseTimeMs");
                 //cerr << "Response: " << "HTTP " << statusCode << std::endl << body << endl;

                 /* We need to make sure that we re-inject bids into the router for each
                  * agent. When receiving a BidResponse, if the SeatBid array contains
                  * less bids than impressions, we still need to tell "no-bid" to the
                  * router for the agent that did not bid, otherwise the router will
                  * be artificially waiting for that particular bidder to bid, and will
                  * expire the auction.
                  */
                 AgentBids bidsToSubmit;
                 Bids bids;
                 bids.reserve(openRtbRequest.imp.size());
                 for (const auto &bidder: bidders) {
                     AgentBidsInfo info;
                     info.agentName = bidder.first;
                     info.agentConfig = bidder.second.agentConfig;
                     info.auctionId = auction->id;
                     info.bids = bids;
                     info.wcm = auction->exchangeConnector->getWinCostModel(
                                       *auction, *info.agentConfig);
                     bidsToSubmit[bidder.first] = info;
                 }

                 // Make sure to submit the bids no matter what
                 ML::Call_Guard guard([&]() { submitBids(bidsToSubmit, openRtbRequest.imp.size()); });

                 if (errorCode != HttpClientError::None) {
                     LOG(error) << "Error requesting " << routerHost << " ("
                                << httpErrorString(errorCode) << ")" << std::endl;
                     recordError("network");
                     goto error;
                 }

                 else if (statusCode == 200) {
                     OpenRTB::BidResponse response;
                     ML::Parse_Context context("payload",
                           body.c_str(), body.size());
                     StreamingJsonParsingContext jsonContext(context);
                     static DefaultDescription<OpenRTB::BidResponse> respDesc;
                     respDesc.parseJson(&response, jsonContext);

                     for (const auto &seatbid: response.seatbid) {

                         for (const auto &bid: seatbid.bid) {
                             if (!bid.ext.isMember("external-id")) {
                                 LOG(error) << "Missing external-id ext field in BidResponse: "
                                            << body << std::endl;
                                 recordError("response");
                                 goto error;
                             }

                             if (!bid.ext.isMember("priority")) {
                                 LOG(error) << "Missing priority ext field in BidResponse: "
                                            << body << std::endl;
                                 recordError("response");
                                 goto error;
                             }

                             uint64_t externalId = bid.ext["external-id"].asUInt();

                             string agent;
                             shared_ptr<const AgentConfig> config;
                             tie(agent, config) = findAgent(externalId);
                             if (config == nullptr) {
                                 LOG(error) << "Couldn't find config for externalId: "
                                            << externalId << std::endl;
                                 recordError("unknown");
                                 goto error;
                             }
                             ExcCheck(!agent.empty(), "Invalid agent");

                             Bid theBid;

                             if (!bid.crid) {
                                 LOG(error) << "crid not found in BidResponse: "
                                            << body << std::endl;
                                 recordError("unknown");
                                 goto error;
                             }

                             int crid = bid.crid.toInt();
                             int creativeIndex = indexOf(config->creatives,
                                 &Creative::id, crid);

                             if (creativeIndex == -1) {
                                  LOG(error) << "Unknown creative id: " << crid << std::endl;
                                  recordError("unknown");
                                  goto error;
                             }

                             theBid.creativeIndex = creativeIndex;
                             theBid.price = USD_CPM(bid.price.val);
                             theBid.priority = bid.ext["priority"].asDouble();

                             int spotIndex = indexOf(openRtbRequest.imp,
                                                    &OpenRTB::Impression::id, bid.impid);
                             if (spotIndex == -1) {
                                 LOG(error) <<"Unknown impression id: "
                                            << bid.impid.toString() << std::endl;
                                 recordError("unknown");
                                 goto error;
                             }

                             theBid.spotIndex = spotIndex;

                             auto &bidInfo = bidsToSubmit[agent];
                             bidInfo.bids.push_back(std::move(theBid));

                         }
                     }

                     return;

                 }
                 else if (statusCode != 204) {
                     LOG(error) << "Invalid HTTP status code: " << statusCode << std::endl;
                     recordError("response");
                     goto error;
                 }

                 // If an error occurs, we will jump here and return "no-bid"
                 error:
                     Bids nullBids;
                     const size_t impSize = openRtbRequest.imp.size();
                     nullBids.reserve(impSize);
                     fill_n(back_inserter(nullBids), impSize, Bid());
                     for (auto &bidsInfo: bidsToSubmit) {
                         bidsInfo.second.bids = nullBids;
                     }
            }
    );

    HttpRequest::Content reqContent { requestStr, "application/json" };

    RestParams headers { { "x-openrtb-version", "2.1" } };
   // std::cerr << "Sending HTTP POST to: " << routerHost << " " << routerPath << std::endl;
   // std::cerr << "Content " << reqContent.str << std::endl;

    httpClientRouter->post(routerPath, callbacks, reqContent,
                     { } /* queryParams */, headers);
}

void HttpBidderInterface::sendLossMessage(
        const std::shared_ptr<const AgentConfig>& agentConfig,
        std::string const & agent, std::string const & id) {

}

void HttpBidderInterface::sendWinLossMessage(
        const std::shared_ptr<const AgentConfig>& agentConfig,
        MatchedWinLoss const & event) {
    if (event.type == MatchedWinLoss::Loss) return;

    auto callbacks = std::make_shared<HttpClientSimpleCallbacks>(
        [=](const HttpRequest &, HttpClientError errorCode,
            int statusCode, const std::string &, std::string &&body)
        {
            if (errorCode != HttpClientError::None) {
                 LOG(error) << "Error requesting "
                            << adserverHost << ":" << adserverWinPort
                            << " (" << httpErrorString(errorCode) << ")" << std::endl;
                 recordError("network");
              }
        });

    Json::Value content;

    content["timestamp"] = event.timestamp.secondsSinceEpoch();
    content["bidRequestId"] = event.auctionId.toString();
    content["impid"] = event.impId.toString();
    content["userIds"] = event.uids.toJsonArray();
    // ratio cannot be casted to json::value ...
    content["price"] = (double) getAmountIn<CPM>(event.winPrice);

    //requestStr["passback"];
    
    HttpRequest::Content reqContent { content, "application/json" };
    httpClientAdserverWins->post("/", callbacks, reqContent,
                         { } /* queryParams */);
    
}


void HttpBidderInterface::sendBidLostMessage(
        const std::shared_ptr<const AgentConfig>& agentConfig,
        std::string const & agent, std::shared_ptr<Auction> const & auction) {
}

void HttpBidderInterface::sendCampaignEventMessage(
        const std::shared_ptr<const AgentConfig>& agentConfig,
        std::string const & agent, MatchedCampaignEvent const & event) {
    auto callbacks = std::make_shared<HttpClientSimpleCallbacks>(
        [=](const HttpRequest &, HttpClientError errorCode,
            int statusCode, const std::string &, std::string &&body)
        {
            if (errorCode != HttpClientError::None) {
                 LOG(error) << "Error requesting "
                            << adserverHost << ":" << adserverEventPort
                            << " (" << httpErrorString(errorCode) << ")" << std::endl;
                 recordError("network");
              }
        });
    
    Json::Value content;

    content["timestamp"] = event.timestamp.secondsSinceEpoch();
    content["bidRequestId"] = event.auctionId.toString();
    content["impid"] = event.impId.toString();
    content["type"] = event.label;
    
    HttpRequest::Content reqContent { content, "application/json" };
    httpClientAdserverEvents->post("/", callbacks, reqContent,
                         { } /* queryParams */);
    
}

void HttpBidderInterface::sendBidDroppedMessage(
        const std::shared_ptr<const AgentConfig>& agentConfig,
        std::string const & agent, std::shared_ptr<Auction> const & auction) {
}

void HttpBidderInterface::sendBidInvalidMessage(
        const std::shared_ptr<const AgentConfig>& agentConfig,
        std::string const & agent, std::string const & reason,
        std::shared_ptr<Auction> const & auction) {
}

void HttpBidderInterface::sendNoBudgetMessage(
        const std::shared_ptr<const AgentConfig>& agentConfig,
        std::string const & agent, std::shared_ptr<Auction> const & auction) {
}

void HttpBidderInterface::sendTooLateMessage(
        const std::shared_ptr<const AgentConfig>& agentConfig,
        std::string const & agent, std::shared_ptr<Auction> const & auction) {
}

void HttpBidderInterface::sendMessage(
        const std::shared_ptr<const AgentConfig>& agentConfig,
        std::string const & agent, std::string const & message) {
}

void HttpBidderInterface::sendErrorMessage(
        const std::shared_ptr<const AgentConfig>& agentConfig,
        std::string const & agent, std::string const & error,
        std::vector<std::string> const & payload) {
}

void HttpBidderInterface::sendPingMessage(
        const std::shared_ptr<const AgentConfig>& agentConfig,
        std::string const & agent, int ping) {
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

void HttpBidderInterface::registerLoopMonitor(LoopMonitor *monitor) const {
    monitor->addMessageLoop("httpBidderInterfaceLoop", &loop);
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

            ext["external-ids"].append(agentConfig->externalId);
        }

    }

}

bool HttpBidderInterface::prepareRequest(OpenRTB::BidRequest &request,
                                         const RTBKIT::BidRequest &originalRequest,
                                         const std::shared_ptr<Auction> &auction,
                                         const std::map<std::string, BidInfo> &bidders) const {
    tagRequest(request, bidders);

    // Take any augmentation data and fill in the ext field of the bid request with the data,
    // under the rtbkit "namespace"
    const auto& augmentations = auction->augmentations;
    if (!augmentations.empty()) {
        Json::Value augJson(Json::objectValue);
        for (const auto& augmentor: augmentations) {
            augJson[augmentor.first] = augmentor.second.toJson();
        }

        request.ext["rtbkit"]["augmentationList"] = augJson;
    }


    // We update the tmax value before sending the BidRequest to substract our processing time
    Date auctionExpiry = auction->expiry;
    double remainingTimeMs = auctionExpiry.secondsSince(Date::now()) * 1000;
    if (remainingTimeMs < 0) {
        return false;
    }

    request.tmax.val = remainingTimeMs;
    return true;
}


void HttpBidderInterface::injectBids(const std::string &agent, Id auctionId,
                                     const Bids &bids, WinCostModel wcm)
{
     BidMessage message;
     message.agents.push_back(agent);
     message.auctionId = auctionId;
     message.bids = bids;
     message.wcm = wcm;
     message.meta = "null";

     // We can not directly call router->doBid here because otherwise we would end up
     // calling doBid from the context of an other thread (the MessageLoop worker thread).
     // Since the object that handles in flight BidRequests for an agent is not
     // thread-safe, we can not call the doBid function from an other thread.
     // Instead, we use a queue to communicate with the router thread. We then avoid
     // an evil race condition.

     if (!router->doBidBuffer.tryPush(std::move(message))) {
         throw ML::Exception("Main router loop can not keep up with HttpBidderInterface");
     }
     router->wakeupMainLoop.signal();
}

void HttpBidderInterface::submitBids(AgentBids &info, size_t impressionsCount) {

    using namespace std;
    for (auto &bidsInfo: info) {

        auto &bids = bidsInfo.second;
        // We check whether the agent bid on all impressions. If not, then we
        // complete the resopnse with no-bids because the router is actually
        // asserting on the size of the bids array matching the size of
        // the impressions object
        const size_t diff = impressionsCount - bids.bids.size();
        if (diff > 0) {
            fill_n(back_inserter(bids.bids), diff, Bid());
        }
        injectBids(bidsInfo.first, bids.auctionId, bids.bids, bids.wcm);
    }
}

void HttpBidderInterface::recordError(const std::string &key) {
     recordHit("error.httpBidderInterface.total");
     recordHit("error.httpBidderInterface.%s", key);
}

//
// factory
//

namespace {

struct AtInit {
    AtInit()
    {
        BidderInterface::registerFactory("http",
        [](std::string const & serviceName,
           std::shared_ptr<ServiceProxies> const & proxies,
           Json::Value const & json)
        {
            return new HttpBidderInterface(serviceName, proxies, json);
        });
    }
} atInit;

}

