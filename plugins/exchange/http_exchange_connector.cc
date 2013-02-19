/* http_exchange_connector.cc
   Jeremy Barnes, 31 January 2011
   Copyright (c) 2011 Datacratic.  All rights reserved.

   Auction endpoint class.
*/

#include "http_exchange_connector.h"
#include "http_auction_handler.h"
#include "jml/arch/exception.h"
#include "jml/arch/format.h"
#include "jml/arch/backtrace.h"
#include "jml/utils/guard.h"
#include "jml/utils/set_utils.h"
#include "jml/utils/vector_utils.h"
#include "jml/arch/timers.h"
#include <set>

#include <boost/foreach.hpp>


using namespace std;
using namespace ML;


namespace RTBKIT {


/*****************************************************************************/
/* HTTP EXCHANGE CONNECTOR                                                   */
/*****************************************************************************/

HttpExchangeConnector::
HttpExchangeConnector(const std::string & name,
                      ServiceBase & parent,
                      OnAuction onNewAuction, OnAuction onAuctionDone)
    : ExchangeConnector(name, parent),
      HttpEndpoint(name),
      acceptAuctionProbability(1.0),
      onNewAuction(onNewAuction),
      onAuctionDone(onAuctionDone),
      numServingRequest_(0)
{
    // Link up events
    onTransportOpen = [=] (TransportBase *)
        {
            this->recordHit("auctionNewConnection");
        };

    onTransportClose = [=] (TransportBase *)
        {
            this->recordHit("auctionClosedConnection");
        };

    handlerFactory = [=] () { return new HttpAuctionHandler(); };
}

HttpExchangeConnector::
~HttpExchangeConnector()
{
    shutdown();
}

void
HttpExchangeConnector::
start()
{
}

void
HttpExchangeConnector::
shutdown()
{
    HttpEndpoint::shutdown();
}

std::shared_ptr<ConnectionHandler>
HttpExchangeConnector::
makeNewHandler()
{
    return makeNewHandlerShared();
}

std::shared_ptr<HttpAuctionHandler>
HttpExchangeConnector::
makeNewHandlerShared()
{
    if (!handlerFactory)
        throw ML::Exception("need to initialize handler factory");

    HttpAuctionHandler * handler = handlerFactory();
    std::shared_ptr<HttpAuctionHandler> handlerSp(handler);
    {
        Guard guard(handlersLock);
        handlers.insert(handlerSp);
    }

    return handlerSp;
}

void
HttpExchangeConnector::
finishedWithHandler(std::shared_ptr<HttpAuctionHandler> handler)
{
    Guard guard(handlersLock);
    handlers.erase(handler);
}

Json::Value
HttpExchangeConnector::
getServiceStatus() const
{
    Json::Value result;

    result["numConnections"] = numConnections();
    result["activeConnections"] = numServingRequest();
    result["connectionLoadFactor"]
        = xdiv<float>(numServingRequest(),
                      numConnections());
    
    map<string, int> peerCounts = numConnectionsByHost();
    
    BOOST_FOREACH(auto cnt, peerCounts)
        result["hostConnections"][cnt.first] = cnt.second;

    return result;
}

std::shared_ptr<BidRequest>
HttpExchangeConnector::
parseBidRequest(const HttpHeader & header,
                const std::string & payload)
{
    throw ML::Exception("need to override HttpExchangeConnector::parseBidRequest");
}

double
HttpExchangeConnector::
getTimeAvailableMs(const HttpHeader & header,
                   const std::string & payload)
{
    throw ML::Exception("need to override HttpExchangeConnector::getTimeAvailableMs");
}

double
HttpExchangeConnector::
getRoundTripTimeMs(const HttpHeader & header,
                   const HttpAuctionHandler & connection)
{
    throw ML::Exception("need to override HttpExchangeConnector::getRoundTripTimeMs");
}

HttpResponse
HttpExchangeConnector::
getResponse(const Auction & auction) const
{
    throw ML::Exception("need to override HttpExchangeConnector::getResponse");
}

HttpResponse
HttpExchangeConnector::
getDroppedAuctionResponse(const Auction & auction,
                          const std::string & reason) const
{
    // Default for when dropped auction == no bid
    return getResponse(auction);
}

HttpResponse
HttpExchangeConnector::
getErrorResponse(const Auction & auction,
                 const std::string & errorMessage) const
{
    // Default for when error == no bid
    return getResponse(auction);
}


} // namespace RTBKIT
