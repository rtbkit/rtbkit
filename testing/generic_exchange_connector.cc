/* generic_exchange_connector.cc
   Jeremy Barnes, 28 November 2012
   Copyright (c) 2012 Datacratic.  All rights reserved.

   Exchange connector component.
*/

#include "generic_exchange_connector.h"
#include "rtbkit/common/bid_request.h"
#include "jml/utils/json_parsing.h"
#include "rtbkit/core/agent_configuration/agent_config.h"
#include <boost/algorithm/string/trim.hpp>
#include "rtbkit/core/router/router.h"


using namespace std;
using namespace ML;

namespace RTBKIT {

/*****************************************************************************/
/* GENERIC EXCHANGE CONNECTOR                                                */
/*****************************************************************************/

GenericExchangeConnector::
GenericExchangeConnector(Router * router, Json::Value config)
    : HttpExchangeConnector("GenericExchangeConnector", *router,
                      [=] (std::shared_ptr<Auction> auction)
                      {
                          router->injectAuction(auction, 0.1 /* seconds */);
                      },
                      [=] (std::shared_ptr<Auction> auctionDone)
                      {
                          router->onAuctionDone(auctionDone);
                      })
{
}

GenericExchangeConnector::
~GenericExchangeConnector()
{
    shutdown();
}

void
GenericExchangeConnector::
shutdown()
{
    PassiveEndpoint::shutdown();
    ExchangeConnector::shutdown();
}

void
GenericExchangeConnector::
configure(const Json::Value & parameters)
{
    numThreads = 2;
    listenPort = 9950;
    bindHost = "*";
    performNameLookup = true;
    backlog = DEF_BACKLOG;

    getParam(parameters, numThreads, "numThreads");
    getParam(parameters, listenPort, "listenPort");
    getParam(parameters, bindHost, "bindHost");
    getParam(parameters, performNameLookup, "performNameLookup");
    getParam(parameters, backlog, "connectionBacklog");
}

void
GenericExchangeConnector::
start()
{
    PassiveEndpoint::init(listenPort, bindHost, numThreads, true,
                          performNameLookup, backlog);
}

std::shared_ptr<BidRequest>
GenericExchangeConnector::
parseBidRequest(const HttpHeader & header,
                const std::string & payload)
{
    const char * start = payload.c_str();
    const char * end = start + payload.length();
    while (end > start && end[-1] == '\n') --end;
        
    string requestStr(start, end);
    Json::Value j = Json::parse(requestStr);
    return std::make_shared<BidRequest>
        (BidRequest::createFromJson(j));
}

double
GenericExchangeConnector::
getTimeAvailableMs(const HttpHeader & header,
                   const std::string & payload)
{
    double timeAvailableMs = 35;

    auto it = header.headers.find("x-timeleft");
    if (it == header.headers.end())
        it = header.headers.find("timeleft");

    if (it != header.headers.end()) {
        //cerr << "got X-TimeLeft of " << it->second << endl;

        const char * start = it->second.c_str();
        const char * end = start + it->second.length();
        char * end2;

        double tm = strtod(start, &end2);
        if (end2 == end)
            timeAvailableMs = tm;
        else cerr << "couldn't parse time " << it->second;
    }

    return timeAvailableMs;
}

double
GenericExchangeConnector::
getRoundTripTimeMs(const HttpHeader & header,
                   const HttpAuctionHandler & connection)
{
    return 5.0;
}

HttpResponse
GenericExchangeConnector::
getResponse(const Auction & auction) const
{
    std::string result;
    result.reserve(256);

    const Auction::Data * current = auction.getCurrentData();

    if (current->hasError()) {
        return getErrorResponse(auction,
                                current->error + ": " + current->details);
    }

    result = "{";
    result += "\"spots\":[";

    int numSpotsThere = 0;
    string passback;

    for (unsigned spotNum = 0;
         spotNum < current->responses.size();  ++spotNum) {
        
        if (!current->hasValidResponse(spotNum))
            continue;

        if (numSpotsThere > 0)
            result += ",";
        ++numSpotsThere;

        // TODO: sign the passback values and encrypt them... it will go in
        // clear text on the internet...

        auto & resp = current->winningResponse(spotNum);
        const AgentConfig * config
            = std::static_pointer_cast<const AgentConfig>
            (resp.agentConfig).get();

        result += format("{\"id\":\"%s\",\"max_price\":%d,\"tag_id\":%d,\"passback\":\"%s,%s,%.6f\"",
                         jsonEscape(auction.request->spots[spotNum].id.toString()).c_str(),
                         resp.price.maxPrice,
                         resp.tagId,
                         jsonEscape(resp.account.toString()),
                         Date::now().secondsSinceEpoch());
            
            // Copy any extra fields from the configuration into the bid
        if (config
            && config->providerConfig.isMember("datacratic")
            && config->providerConfig["datacratic"].isMember("bidExtraFields")) {
            auto & fields = config->providerConfig["datacratic"]["bidExtraFields"];
            for (auto it = fields.begin(), end = fields.end();
                 it != end;  ++it) {
                string field = it.memberName();
                result += ",\"" + field + "\":" + boost::trim_copy(it->toString());
            }
        }
        result += "}";
    }
    
    result += "]}";

    return HttpResponse(200, "application/json", result);
}

HttpResponse
GenericExchangeConnector::
getDroppedAuctionResponse(const Auction & auction,
                          const std::string & reason) const
{
    Json::Value response;
    if (reason != "")
        response["reason"] = "Early drop";
    response["spots"] = Json::Value(Json::arrayValue);
    
    return HttpResponse(200, response);
}

HttpResponse
GenericExchangeConnector::
getErrorResponse(const Auction & auction,
                 const std::string & errorMessage) const
{
    Json::Value response;
    response["error"] = errorMessage;

    return HttpResponse(400, response);
}

} // namespace RTBKIT


