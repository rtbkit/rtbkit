/** standalone_bidder_ex.cc
    Jeremy Barnes, 17 February 2013
    Copyright (c) 2013 Datacratic Inc.  All rights reserved.

    Example program that starts up a standalone bidder.  This will receive bid requests
    from a set of exchanges (specified in a JSON file) and bid a fixed price for each
    impression seen.
*/

#include "rtbkit/plugins/exchange/http_exchange_connector.h"
#include <boost/any.hpp>

using namespace std;
using namespace RTBKIT;



struct DumpingExchangeConnector: public HttpExchangeConnector {

    DumpingExchangeConnector(const std::string & name,
                             std::shared_ptr<ServiceProxies> proxies)
        : HttpExchangeConnector(name, proxies)
    {
    }

    virtual std::string exchangeName() const
    {
        return "dumping";
    }

    virtual std::shared_ptr<BidRequest>
    parseBidRequest(HttpAuctionHandler & connection,
                    const HttpHeader & header,
                    const std::string & payload)
    {
        cerr << "got request" << endl << header << endl << payload << endl;
        return std::make_shared<BidRequest>();
    }

    virtual double
    getTimeAvailableMs(HttpAuctionHandler & connection,
                       const HttpHeader & header,
                       const std::string & payload)
    {
        return 10;
    }

    virtual double
    getRoundTripTimeMs(HttpAuctionHandler & connection,
                       const HttpHeader & header)
    {
        return 5;
    }

    virtual HttpResponse getResponse(const HttpAuctionHandler & connection,
                                     const HttpHeader & requestHeader,
                                     const Auction & auction) const
    {
        return HttpResponse(204, "", "");
    }

    virtual HttpResponse
    getDroppedAuctionResponse(const HttpAuctionHandler & connection,
                              const std::string & reason) const
    {
        return HttpResponse(204, "application/json", "{}");
    }

    virtual HttpResponse
    getErrorResponse(const HttpAuctionHandler & connection,
                     const std::string & errorMessage) const
    {
        return HttpResponse(400, "application/json", "{}");
    }
};

int main(int argc, char ** argv)
{
    std::shared_ptr<ServiceProxies> proxies(new ServiceProxies());

    DumpingExchangeConnector connector("connector", proxies);
    
    connector.configureHttp(1, 10002, "0.0.0.0");
    connector.start();
    connector.enableUntil(Date::positiveInfinity());

    for (;;) {
        sleep(10.0);
    }

    return 1;
}
