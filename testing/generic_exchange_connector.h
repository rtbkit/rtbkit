/* generic_exchange_connector.h                                    -*- C++ -*-
   Jeremy Barnes, 1 Febrary 2012
   Copyright (c) 2012 Datacratic.  All rights reserved.

   Base class for all Exchange connectors
*/

#pragma once

#include "rtbkit/plugins/exchange/http_exchange_connector.h"
#include "rtbkit/plugins/exchange/exchange_connector.h"

namespace RTBKIT {


/*****************************************************************************/
/* GENERIC EXCHANGE CONNECTOR                                                */
/*****************************************************************************/

/** Exchange connector that takes bid requests in our generic bid request
    format over HTTP.
*/

struct GenericExchangeConnector
    : public RTBKIT::HttpExchangeConnector {
    
    GenericExchangeConnector(RTBKIT::Router * router,
                             Json::Value config);

    ~GenericExchangeConnector();

    virtual void shutdown();

    int numThreads;
    int listenPort;
    std::string bindHost;
    bool performNameLookup;
    int backlog;

    virtual void configure(const Json::Value & parameters);

    virtual void start();

    virtual std::shared_ptr<RTBKIT::BidRequest>
    parseBidRequest(const HttpHeader & header,
                    const std::string & payload);

    virtual double
    getTimeAvailableMs(const HttpHeader & header,
                       const std::string & payload);

    virtual double
    getRoundTripTimeMs(const HttpHeader & header,
                       const RTBKIT::HttpAuctionHandler & connection);

    virtual HttpResponse getResponse(const RTBKIT::Auction & auction) const;

    virtual HttpResponse
    getDroppedAuctionResponse(const RTBKIT::Auction & auction,
                              const std::string & reason) const;

    virtual HttpResponse
    getErrorResponse(const RTBKIT::Auction & auction,
                     const std::string & errorMessage) const;

};

} // namespace RTBKIT
