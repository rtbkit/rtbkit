 /* mock_exchange_connector.cc
   Eric Robert, 9 April 2013
   Copyright (c) 2012 Datacratic Inc.  All rights reserved.

   Simple mock exchange connector
*/

#include "rtbkit/core/router/router_runner.h"
#include "rtbkit/plugins/exchange/http_exchange_connector.h"
#include "jml/utils/json_parsing.h"

namespace RTBKIT {

struct MockExchangeConnector : public HttpExchangeConnector {
    MockExchangeConnector(ServiceBase & owner, const std::string & name) :
        HttpExchangeConnector(name, owner)
    {
        this->auctionResource = "/bids";
        this->auctionVerb = "POST";
    }

    MockExchangeConnector(const std::string & name, std::shared_ptr<ServiceProxies> proxies) :
        HttpExchangeConnector(name, proxies)
    {
        this->auctionResource = "/bids";
        this->auctionVerb = "POST";
    }

    static std::string exchangeNameString() {
        return "mock";
    }

    std::string exchangeName() const {
        return exchangeNameString();
    } 

    void configure(Json::Value const & parameters) {
        numThreads = 1;
        listenPort = 12339;
        bindHost = "*";
        performNameLookup = true;
        backlog = 128;

        getParam(parameters, numThreads, "numThreads");
        getParam(parameters, listenPort, "listenPort");
        getParam(parameters, bindHost, "bindHost");
        getParam(parameters, performNameLookup, "performNameLookup");
        getParam(parameters, backlog, "connectionBacklog");
    }

    void start() {
        init(listenPort, bindHost, numThreads, true, performNameLookup, backlog);
    }

    std::shared_ptr<BidRequest> parseBidRequest(HttpAuctionHandler & handler,
                                                Datacratic::HttpHeader const & header,
                                                std::string const & payload) {
        std::shared_ptr<BidRequest> request;
        request.reset(BidRequest::parse("datacratic", payload));
        return request;
    }

    double getTimeAvailableMs(HttpAuctionHandler & handler,
                              const HttpHeader & header,
                              const std::string & payload) {
        return 35.0;
    }

    double getRoundTripTimeMs(HttpAuctionHandler & handler,
                              const HttpHeader & header) {
        return 5.0;
    }

    Datacratic::HttpResponse getResponse(HttpAuctionHandler const & handler,
                                         const HttpHeader & header,
                                         Auction const & auction) const {
        std::string result;
    
        const Auction::Data * current = auction.getCurrentData();
        if (current->hasError())
            return getErrorResponse(handler, current->error + ": " + current->details);

        result = "{\"imp\":[";

        bool first = true;
        for (unsigned spotNum = 0; spotNum < current->responses.size(); ++spotNum) {
            if (!current->hasValidResponse(spotNum))
                continue;

            if (!first) result += ",";
            first = false;

            auto & resp = current->winningResponse(spotNum);
            result += ML::format("{\"id\":\"%s\",\"max_price\":%ld,\"account\":\"%s\"}",
                             ML::jsonEscape(auction.request->imp.at(spotNum).id.toString()).c_str(),
                             (int64_t)(MicroUSD_CPM(resp.price.maxPrice)),
                             resp.account.toString('.'));
        }

        result += "]}";
        return HttpResponse(200, "application/json", result);
    }

    int numThreads;
    int listenPort;
    std::string bindHost;
    bool performNameLookup;
    int backlog;
};

}
