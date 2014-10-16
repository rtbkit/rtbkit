/* generic_exchange_connector.h                                    -*- C++ -*-
   Jeremy Barnes, 1 Febrary 2012
   Copyright (c) 2012 Datacratic.  All rights reserved.

   Base class for all Exchange connectors
*/

#pragma once

#include "rtbkit/plugins/exchange/http_exchange_connector.h"
#include "rtbkit/common/exchange_connector.h"

namespace RTBKIT {


/*****************************************************************************/
/* GENERIC EXCHANGE CONNECTOR                                                */
/*****************************************************************************/

/** Exchange connector that takes bid requests in our generic bid request
    format over HTTP.
*/

struct GenericExchangeConnector
    : public RTBKIT::HttpExchangeConnector {
    
    GenericExchangeConnector(ServiceBase & owner);
    GenericExchangeConnector(
            std::shared_ptr<ServiceProxies> proxies = std::shared_ptr<ServiceProxies>());

    ~GenericExchangeConnector();

    virtual void shutdown();

    int numThreads;
    int listenPort;
    std::string bindHost;
    bool performNameLookup;
    int backlog;

    virtual std::string exchangeName() const
    {
        return "rtbkit";
    }

    virtual std::shared_ptr<RTBKIT::BidRequest>
    parseBidRequest(HttpAuctionHandler & connection,
                    const HttpHeader & header,
                    const std::string & payload);

    virtual double
    getTimeAvailableMs(HttpAuctionHandler & connection,
                       const HttpHeader & header,
                       const std::string & payload);

    virtual double
    getRoundTripTimeMs(HttpAuctionHandler & connection,
                       const HttpHeader & header);

    virtual HttpResponse
    getResponse(const HttpAuctionHandler & connection,
                const HttpHeader & requestHeader,
                const RTBKIT::Auction & auction) const;

    virtual HttpResponse
    getDroppedAuctionResponse(const HttpAuctionHandler & connection,
                              const std::string & reason) const;

    virtual HttpResponse
    getErrorResponse(const HttpAuctionHandler & connection,
                     const std::string & errorMessage) const;

};

} // namespace RTBKIT
