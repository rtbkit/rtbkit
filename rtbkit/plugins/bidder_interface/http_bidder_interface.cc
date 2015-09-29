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

auto HttpBidderInterface::readFormat(const std::string& fmt) -> Format {
    if (fmt == "standard") return FMT_STANDARD;
    if (fmt == "datacratic") return FMT_DATACRATIC;
    ExcCheck(false, "unknown format string: " + fmt);
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

        routerHost = router["host"].asString();
        routerPath = router["path"].asString();
        routerFormat = readFormat(router.get("format", "standard").asString());
        routerHttpActiveConnections = router.get("httpActiveConnections", 1024).asInt();

        adserverHost = adserver["host"].asString();

        adserverWinPort = adserver["winPort"].asInt();
        adserverWinPath = adserver.get("winPath", "/").asString();
        adserverWinFormat = readFormat(adserver.get("winFormat", "standard").asString());

        adserverEventPort = adserver["eventPort"].asInt();
        adserverEventPath = adserver.get("eventPath", "/").asString();
        adserverEventFormat = readFormat(adserver.get("eventFormat", "standard").asString());

        adserverErrorPort = adserver["errorPort"].asInt();
        adserverErrorPath = adserver.get("errorPath", "/").asString();
        adserverErrorFormat = readFormat(adserver.get("errorFormat", "standard").asString());

        adserverHttpActiveConnections = adserver.get("httpActiveConnections", 1024).asInt();
    } catch (const std::exception & e) {
        THROW(error) << "configuration file is invalid" << std::endl
                   << "usage : " << std::endl
                   << "{" << std::endl << "\t\"router\" : {" << std::endl
                   << "\t\t\"host\" : <string : hostname with port>" << std::endl  
                   << "\t\t\"path\" : <string : resource name>" << std::endl
                   << "\t\t\"format\" : <string : message format>" << std::endl
                   << "\t\t\"httpActiveConnections\" : <int : concurrent connections>"
                   << std::endl
                   << "\t\t"
                   << "\t}" << std::endl << "\t{" << std::endl 
                   << "\t{" << std::endl << "\t\"adserver\" : {" << std::endl
                   << "\t\t\"host\" : <string : hostname>" << std::endl  
                   << "\t\t\"winPort\" : <int : winPort>" << std::endl  
                   << "\t\t\"winPath\" : <string : resource name>" << std::endl
                   << "\t\t\"winFormat\" : <string : message format>" << std::endl
                   << "\t\t\"eventPort\" : <int eventPort>" << std::endl
                   << "\t\t\"eventPath\" : <string : resource name>" << std::endl
                   << "\t\t\"eventFormat\" : <string : message format>" << std::endl
                   << "\t\t\"errorPort\" : <int errorPort>" << std::endl
                   << "\t\t\"errorPath\" : <string : resource name>" << std::endl
                   << "\t\t\"errorFormat\" : <string : message format>" << std::endl
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

    std::string errorHost = adserverHost + ':' + std::to_string(adserverErrorPort);
    httpClientAdserverErrors.reset(new HttpClient(errorHost, adserverHttpActiveConnections));
    httpClientAdserverErrors->sendExpect100Continue(false);
    loop.addSource("HttpBidderInterface::httpClientAdserverErrors", httpClientAdserverErrors);

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
            /* Since it is possible to delete a configuration from the REST interface of
             * the agent configuration service, the user might delete the configuration
             * while some requests for this configuration are already in flight. When
             * that happens, since we're capturing our context by copy in the closure,
             * we hold a "private" copy of the current agents and their configurations,
             * which means that we might still hold configurations that have been deleted
             * and erased in the router.
             *
             * This is why we are checking if the agent still exists. If not, we're skipping
             * it. This is not ideal and introduces an extra check but this is the simplest way
             * Note that this will be trigger the "couldn't fint configuration for
             * externalId" error below. In other words, all requests that are "in flight"
             * for a configuration that has been deleted will trigger a logging message.
             * We will return a 204 for these requests
             */
            auto agentIt = router->agents.find(agent);
            if (agentIt == std::end(router->agents)) {
                return false;
            }
            const auto &info = agentIt->second;
            ExcAssert(info.config);
            return info.config->externalId == externalId;
        });

        if (it == end(bidders)) {
            return make_pair("", nullptr);
        }

        return make_pair(it->first, it->second.agentConfig);

    };

    BidRequest & originalRequest = *auction->request;

    std::string openRtbVersion;
    if (!originalRequest.protocolVersion.empty())
        openRtbVersion = originalRequest.protocolVersion;
    else
        openRtbVersion = "2.1";

    std::shared_ptr<OpenRTBBidRequestParser> parser = OpenRTBBidRequestParser::openRTBBidRequestParserFactory(openRtbVersion);


    OpenRTB::BidRequest openRtbRequest = parser->toBidRequest(originalRequest);
    bool ok = prepareRequest(openRtbRequest, originalRequest, auction, bidders);
    /* If we took too much time processing the request, then we don't send it.  */
    if (!ok) {
        return;
    }

    string requestStr;
    if (routerFormat == FMT_DATACRATIC) {
        Json::Value jReq(Json::objectValue);
        jReq["id"] = openRtbRequest.id.toString();

        DefaultDescription<OpenRTB::Impression> impDesc;
        for (auto & imp : openRtbRequest.imp) {
            StructuredJsonPrintingContext ctx;
            impDesc.printJson(&imp, ctx);
            jReq["imp"].append(ctx.output);
        }

        jReq["ext"]["datacratic"] = openRtbRequest.ext["datacratic"];
        jReq["ext"]["rtbkit"] = openRtbRequest.ext["rtbkit"];
        requestStr = jReq.toString();
    } else {
        StructuredJsonPrintingContext context;
        desc.printJson(&openRtbRequest, context);
        requestStr = context.output.toString();
    }

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
                recordOutcome(1000.0 * responseTime, "httpResponseTimeMs");
               // cerr << "Response: " << "HTTP " << statusCode << std::endl << body << endl;

                 /* We need to make sure that we re-inject bids into the router for each
                  * agent. When receiving a BidResponse, if the SeatBid array contains
                  * less bids than impressions, we still need to tell "no-bid" to the
                  * router for the agent that did not bid, otherwise the router will
                  * be artificially waiting for that particular bidder to bid, and will
                  * expire the auction.
                  */
                 AgentBids bidsToSubmit;

                for (const auto &bidder: bidders) {
                     AgentBidsInfo info;
                     info.agentName = bidder.first;
                     info.agentConfig = bidder.second.agentConfig;
                     info.auctionId = auction->id;
                     info.wcm = auction->exchangeConnector->getWinCostModel(
                                       *auction, *info.agentConfig);

                     const BiddableSpots& imps = bidder.second.imp;
                     info.bids.reserve(imps.size());
                     for (size_t i = 0; i < imps.size(); ++i) {
                         Bid bid;
                         bid.spotIndex = imps[i].first;
                         info.bids.push_back(bid);
                     }

                     bidsToSubmit[bidder.first] = info;
                 }

                 // Make sure to submit the bids no matter what
                 ML::Call_Guard submitGuard([&] { submitBids(bidsToSubmit); });

                 if (errorCode != HttpClientError::None) {
                     LOG(error) << "Error requesting " << routerHost << " ("
                         << httpErrorString(errorCode) << ")" << std::endl;
                     recordError("network");
                     return;
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
                             Bid theBid;
                             string agent;
                             shared_ptr<const AgentConfig> config;

                             if (routerFormat == FMT_STANDARD) {
                                 if (!bid.ext.isMember("external-id")) {
                                     LOG(error) << "Missing external-id ext field in BidResponse: " << body << std::endl;
                                     recordError("response");
                                     return;
                                 }
                                 uint64_t externalId = bid.ext["external-id"].asUInt();

                                 if (!bid.ext.isMember("priority")) {
                                     LOG(error) << "Missing priority ext field in BidResponse: " << body << std::endl;
                                     recordError("response");
                                     return;
                                 }
                                 theBid.priority = bid.ext["priority"].asDouble();


                                 tie(agent, config) = findAgent(externalId);
                                 if (config == nullptr) {
                                     LOG(error) << "Couldn't find config for externalId: " << externalId << std::endl;
                                     recordError("unknown");
                                     return;
                                 }
                             }

                             else if (routerFormat == FMT_DATACRATIC) {

                                 for (const auto& entry : bidders) {
                                     config = entry.second.agentConfig;
                                     if (config->account[1] == bid.cid.toString()) {
                                         agent = entry.first;
                                         break;
                                     }
                                 }

                                 if (agent.empty()) {
                                     LOG(error) << "Couldn't find config for cid: " << bid.cid << std::endl;
                                     recordError("unknown");
                                     return;
                                 }

                                 theBid.ext = bid.ext["rtbkit"]["meta"];
                                 theBid.priority = bid.ext["rtbkit"]["priority"].asDouble();
                             }

                             else ExcAssert(false);

                             ExcCheck(!agent.empty(), "Invalid agent");

                             if (!bid.crid) {
                                 LOG(error) << "crid not found in BidResponse: " << body << std::endl;
                                 recordError("unknown");
                                 return;
                             }

                             int crid = bid.crid.toInt();
                             int creativeIndex = indexOf(config->creatives,
                                 &Creative::id, crid);

                             if (creativeIndex == -1) {
                                 LOG(error) << "Unknown creative id: " << crid << std::endl;
                                 recordError("unknown");
                                 return;
                             }

                             theBid.creativeIndex = creativeIndex;
                             theBid.price = USD_CPM(bid.price.val);

                             int spotIndex = indexOf(openRtbRequest.imp,
                                                    &OpenRTB::Impression::id, bid.impid);
                             if (spotIndex == -1) {
                                 LOG(error) <<"Unknown impression id: " << bid.impid.toString() << std::endl;
                                 recordError("unknown");
                                 return;
                             }

                             auto &bidInfo = bidsToSubmit[agent];
                             theBid.spotIndex = spotIndex;
                             bidInfo.bids.bidForSpot(spotIndex) = theBid;
                         }
                     }

                 }
                 else if (statusCode != 204) {
                     LOG(error) << "Invalid HTTP status code: " << statusCode << std::endl
                               << body << std::endl;
                     recordError("response");
                     return;
                 }

            }
    );

    HttpRequest::Content reqContent { requestStr, "application/json" };

    RestParams headers { { "x-openrtb-version", openRtbVersion } };
   // std::cerr << "Sending HTTP POST to: " << routerHost << " " << routerPath << std::endl;
   // std::cerr << "Content " << reqContent.str << std::endl;

    httpClientRouter->post(routerPath, callbacks, reqContent,
                     { } /* queryParams */, headers);
}

void HttpBidderInterface::sendLossMessage(
        const std::shared_ptr<const AgentConfig>& agentConfig,
        std::string const & agent, std::string const & id) {

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

    if (adserverEventFormat == FMT_DATACRATIC) {
        content["id"] = id;

        Json::Value entry;
        {
            entry["cid"] = agent;
            entry["type"] = "loss";
            entry["id"] = id;
        }
        content["events"].append(entry);

    } else ExcAssert(false);

    HttpRequest::Content reqContent { content, "application/json" };
    httpClientAdserverEvents->post(adserverEventPath, callbacks, reqContent, {} /* queryParams */);
}

void HttpBidderInterface::sendWinLossMessage(
        const std::shared_ptr<const AgentConfig>& agentConfig,
        MatchedWinLoss const & event) {

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

    if (adserverWinFormat == FMT_STANDARD) {
        content["timestamp"] = event.timestamp.secondsSinceEpoch();
        content["bidRequestId"] = event.auctionId.toString();
        content["impid"] = event.impId.toString();
        content["userIds"] = event.uids.toJsonArray();
        // ratio cannot be casted to json::value ...
        content["price"] = (double) getAmountIn<CPM>(event.winPrice);
        content["account"] = event.response.account.toJson();
        content["creative-id"] = event.response.creativeId;

        auto& ext = content["ext"];
        ext["external-id"] = agentConfig->externalId;
        ext["creative-index"] = event.response.agentCreativeIndex;
        //requestStr["passback"];
    }
    else if (adserverWinFormat == FMT_DATACRATIC) {
        content["id"] = event.auctionId.toString();

        auto& ext = content["ext"]["datacratic"];
        ext["request"] = event.requestStr;
        ext["requestFormat"] = event.requestStrFormat;

        Json::Value entry;
        {
            entry["impid"] = event.impId.toString();
            entry["type"] = event.type == MatchedWinLoss::Loss ? "loss" : "win";
            entry["bidPrice"] = (double) getAmountIn<CPM>(event.response.price.maxPrice);
            entry["winPrice"] = event.type == MatchedWinLoss::Loss ? 0.0 : (double) getAmountIn<CPM>(event.winPrice);
            entry["cid"] = event.response.account[1];
            entry["ext"]["datacratic"]["meta"] = Json::parse(event.response.meta.rawString());

            Json::Value users;
            for (const auto& item : event.uids) {
                Json::Value user;
                user["id"] = item.second.toString();
                users.append(user);
            }
            entry["users"] = users;
        }
        content["events"].append(entry);
    }
    else ExcAssert(false);
    
    HttpRequest::Content reqContent { content, "application/json" };
    httpClientAdserverWins->post(adserverWinPath, callbacks, reqContent,
                         { } /* queryParams */);
}


void HttpBidderInterface::sendBidLostMessage(
        const std::shared_ptr<const AgentConfig>& agentConfig,
        std::string const & agent, std::shared_ptr<Auction> const & auction) {
}

void HttpBidderInterface::sendCampaignEventMessage(
        const std::shared_ptr<const AgentConfig>& agentConfig,
        std::string const & agent, MatchedCampaignEvent const & event)
{
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

    if (adserverEventFormat == FMT_STANDARD) {
        content["timestamp"] = event.timestamp.secondsSinceEpoch();
        content["bidRequestId"] = event.auctionId.toString();
        content["impid"] = event.impId.toString();
        content["type"] = event.label;
        content["account"] = event.response.account.toJson();
        content["creative-id"] = event.response.creativeId;

        auto& ext = content["ext"];
        ext["external-id"] = agentConfig->externalId;
        ext["creative-index"] = event.response.agentCreativeIndex;
    }
    else if (adserverEventFormat == FMT_DATACRATIC) {
        content["id"] = event.auctionId.toString();

        auto& ext = content["ext"]["datacratic"];
        ext["request"] = event.requestStr;
        ext["requestFormat"] = event.requestStrFormat;

        Json::Value entry;
        {
            entry["impid"] = event.impId.toString();
            entry["type"] = event.label;
            entry["cid"] = event.response.account[1];
            entry["ext"]["datacratic"]["meta"] = Json::parse(event.bid["meta"].asString());
        }
        content["events"].append(entry);
    }
    else ExcAssert(false);

    HttpRequest::Content reqContent { content, "application/json" };
    httpClientAdserverEvents->post(adserverEventPath, callbacks, reqContent,
                         { } /* queryParams */);
    
}

void HttpBidderInterface::sendBidErrorMessage(
        const std::shared_ptr<const AgentConfig>& agentConfig,
        std::string const & agent, std::shared_ptr<Auction> const & auction,
        std::string const & type, std::string const & reason)
{
    auto callbacks = std::make_shared<HttpClientSimpleCallbacks>(
        [=](const HttpRequest &, HttpClientError errorCode,
            int statusCode, const std::string &, std::string &&body)
        {
            if (errorCode != HttpClientError::None) {
                 LOG(error) << "Error requesting "
                            << adserverHost << ":" << adserverErrorPort
                            << " (" << httpErrorString(errorCode) << ")" << std::endl;
                 recordError("network");
              }
        });

    Json::Value content;
    content["id"] = auction->id.toString();
    content["cid"] = agent;
    content["type"] = type;
    if (!reason.empty()) content["reason"] = reason;

    HttpRequest::Content reqContent { content, "application/json" };
    httpClientAdserverErrors->post(adserverErrorPath, callbacks, reqContent, {});
}

void HttpBidderInterface::sendBidDroppedMessage(
        const std::shared_ptr<const AgentConfig>& agentConfig,
        std::string const & agent, std::shared_ptr<Auction> const & auction)
{
    if (adserverEventFormat == FMT_DATACRATIC)
        sendBidErrorMessage(agentConfig, agent, auction, "ERROR", "DROPPED");
}

void HttpBidderInterface::sendBidInvalidMessage(
        const std::shared_ptr<const AgentConfig>& agentConfig,
        std::string const & agent, std::string const & reason,
        std::shared_ptr<Auction> const & auction)
{
    if (adserverEventFormat == FMT_DATACRATIC)
        sendBidErrorMessage(agentConfig, agent, auction, "ERROR", "INVALID");
}

void HttpBidderInterface::sendNoBudgetMessage(
        const std::shared_ptr<const AgentConfig>& agentConfig,
        std::string const & agent, std::shared_ptr<Auction> const & auction)
{
    if (adserverEventFormat == FMT_DATACRATIC)
        sendBidErrorMessage(agentConfig, agent, auction, "ERROR", "NOBUDGET");
}

void HttpBidderInterface::sendTooLateMessage(
        const std::shared_ptr<const AgentConfig>& agentConfig,
        std::string const & agent, std::shared_ptr<Auction> const & auction)
{
    if (adserverEventFormat == FMT_DATACRATIC)
        sendBidErrorMessage(agentConfig, agent, auction, "ERROR", "TOOLATE");
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
    monitor->addMessageLoop(serviceName(), &loop);
}

bool HttpBidderInterface::prepareRequest(OpenRTB::BidRequest &request,
                                         const RTBKIT::BidRequest &originalRequest,
                                         const std::shared_ptr<Auction> &auction,
                                         const std::map<std::string, BidInfo> &bidders) const
{
    if (routerFormat == FMT_STANDARD)
        return prepareStandardRequest(request, originalRequest, auction, bidders);

    if (routerFormat == FMT_DATACRATIC)
        return prepareDatacraticRequest(request, originalRequest, auction, bidders);

    ExcAssert(false);
}

void HttpBidderInterface::tagRequest(OpenRTB::BidRequest &request,
                                     const std::map<std::string, BidInfo> &bidders) const
{
    static const Json::Value null(Json::nullValue);

    static constexpr const char *ExternalIdsFieldName = "external-ids";
    static constexpr const char *CreativeIdsFieldName = "creative-ids";

    // Make sure to tag every impression, even impressions that do not satisfy
    // filters
    for (auto& imp: request.imp)
        imp.ext[ExternalIdsFieldName] = imp.ext[CreativeIdsFieldName] = null;

    for (const auto &bidder: bidders) {
        const auto &agentConfig = bidder.second.agentConfig;
        const auto &spots = bidder.second.imp;

        for (const auto &spot: spots) {
            const int adSpotIndex = spot.first;
            const auto& creativeIndexes = spot.second;
            ExcCheck(adSpotIndex >= 0 && adSpotIndex < request.imp.size(),
                     "adSpotIndex out of range");
            auto &imp = request.imp[adSpotIndex];
            auto &externalIds = imp.ext[ExternalIdsFieldName];
            externalIds.append(agentConfig->externalId);

            auto& creativesExtField = imp.ext[CreativeIdsFieldName];


            auto &creativesList = creativesExtField[std::to_string(agentConfig->externalId)];
            const auto& creatives = agentConfig->creatives;
            for (int index: creativeIndexes) {
                ExcAssert(index < creatives.size());

                creativesList.append(creatives[index].id);
            }

        }

    }

}

bool HttpBidderInterface::prepareStandardRequest(OpenRTB::BidRequest &request,
                                         const RTBKIT::BidRequest &originalRequest,
                                         const std::shared_ptr<Auction> &auction,
                                         const std::map<std::string, BidInfo> &bidders) const {
    tagRequest(request, bidders);

     request.ext["exchange"] = originalRequest.exchange;

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

bool HttpBidderInterface::prepareDatacraticRequest(OpenRTB::BidRequest &request,
                                                   const RTBKIT::BidRequest &originalRequest,
                                                   const std::shared_ptr<Auction> &auction,
                                                   const std::map<std::string, BidInfo> &bidders) const
{
    // uses strategy slug as the bidder id.
    auto bidderId = [] (const AccountKey& account) { return account[1]; };

    for (const auto& bidder : bidders) {
        const AgentConfig& config = *bidder.second.agentConfig;
        std::string id = bidderId(config.account);

        for (const auto& imp : bidder.second.imp) {
            auto& ext = request.imp[imp.first].ext;

            auto& creatives = ext["datacratic"]["allowed"][id];
            for (size_t creativeIndex : imp.second) {
                int creativeId = config.creatives[creativeIndex].id;
                creatives.append(creativeId);
            }
        }
    }

    auto& dataExt = request.ext["rtbkit"]["data"];
    for (const auto& augmentation : auction->augmentations) {
        auto& augExt = dataExt[augmentation.first];

        for (const auto& bidder : bidders) {
            const AccountKey& account = bidder.second.agentConfig->account;

            Augmentation aug = augmentation.second.filterForAccount(account);
            if (!!aug.data) augExt[bidderId(account)] = aug.data;
        }
    }

    auto& ext = request.ext["datacratic"];
    ext["request"] = auction->requestStr;
    ext["requestFormat"] = auction->requestStrFormat;

    // We update the tmax value before sending the BidRequest to substract our
    // processing time
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

void HttpBidderInterface::submitBids(AgentBids &info) {
    for (auto &bidsInfo: info) {
        auto &bids = bidsInfo.second;
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
      PluginInterface<BidderInterface>::registerPlugin("http",
          [](std::string const &serviceName,
             std::shared_ptr<ServiceProxies> const &proxies,
             Json::Value const &json)
          {
              return new HttpBidderInterface(serviceName, proxies, json);
          });
    }
} atInit;

}

