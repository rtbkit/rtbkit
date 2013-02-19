/* http_exchange_connector.h                                       -*- C++ -*-
   Jeremy Barnes, 31 January 2011
   Copyright (c) 2011 Datacratic.  All rights reserved.

   Endpoint for bidding system.
*/

#pragma once

#include <boost/scoped_ptr.hpp>
#include <boost/shared_ptr.hpp>
#include "jml/arch/atomic_ops.h"
#include "soa/service/json_endpoint.h"
#include "soa/service/stats_events.h"
#include "rtbkit/common/auction.h"
#include <limits>
#include "exchange_connector.h"
#include <boost/algorithm/string.hpp>


namespace RTBKIT {

struct HttpExchangeConnector;
struct HttpAuctionHandler;


/*****************************************************************************/
/* HTTP EXCHANGE CONNECTOR                                                   */
/*****************************************************************************/

/** Class that manages incoming auctions. */

struct HttpExchangeConnector
    : public ExchangeConnector,
      public HttpEndpoint {

    /** Function that will be called to notify of a new auction. */
    typedef boost::function<void (std::shared_ptr<Auction> Auction)>
        OnAuction;
    
    HttpExchangeConnector(const std::string & name,
                          ServiceBase & parent,
                          OnAuction onNewAuction, OnAuction onAuctionDone);
    ~HttpExchangeConnector();

    /** How many connections are serving a request at the moment? */
    int numServingRequest() const
    {
        return numServingRequest_;
    }
    
    virtual Json::Value getServiceStatus() const;


    /*************************************************************************/
    /* METHODS CALLED BY THE ROUTER TO CONTROL THE EXCHANGE CONNECTOR        */
    /*************************************************************************/

    /** Configure the exchange connector.  The JSON provided is entirely
        interpreted by the exchange connector itself.
    */
    virtual void configure(const Json::Value & parameters) = 0;

    /** Start the exchange connector running */
    virtual void start();

    /** Shutdown the exchange connector ready to be destroyed. */
    virtual void shutdown();

    /** Set the time until which the exchange is enabled.  Normally this will
        be pushed forward a few seconds periodically so that everything will
        shut down if there is nothing controlling the exchange connector.
    */
    virtual void enableUntil(Date date)
    {
        this->enabledUntil = date;
    }

    /** Are we currently authorized to bid? */
    bool isEnabled(Date now = Date::now()) const
    {
        return now <= enabledUntil;
    }

    /** Set which percentage of bid requests will be accepted by the
        exchange connector.
    */
    virtual void setAcceptBidRequestProbability(double prob)
    {
        if (prob < 0 || prob > 1)
            throw ML::Exception("invalid probability for "
                                "setBidRequestProbability: "
                                "%f is not between 0 and 1");
        this->acceptAuctionProbability = prob;
    }


    /*************************************************************************/
    /* METHODS TO OVERRIDE FOR A GIVEN EXCHANGE                              */
    /*************************************************************************/

    /** Parse the given payload into a bid request. */
    virtual std::shared_ptr<BidRequest>
    parseBidRequest(const HttpHeader & header,
                    const std::string & payload);

    /** Return the available time for the bid request in milliseconds.  This
        method should not parse the bid request, as when shedding load
        we want to do as little work as possible.

        Most exchanges include this information in the HTTP headers.
    */
    virtual double
    getTimeAvailableMs(const HttpHeader & header,
                       const std::string & payload);

    /** Return an estimate of how long a round trip with the connected
        server takes, in milliseconds at the exchange's latency percentile,
        including all hops (load balancers, reverse proxies, etc).

        The default implementation returns 5ms, which is only valid for when
        a service is in the same datacenter.  Many exchanges implement
        as part of their protocol a way to measure the round trip time
        between a given exchange host and a given bidder.
    */
    virtual double
    getRoundTripTimeMs(const HttpHeader & header,
                       const HttpAuctionHandler & connection);

    /** Return the HTTP response for our auction.  Default
        implementation calls getResponse() and stringifies the result.

        This version is provided as it may be more efficient in terms of
        memory allocations.

        The first element returned is the HTTP body, the second is the
        content type.
    */
    virtual HttpResponse getResponse(const Auction & auction) const;

    /** Return a stringified JSON of the response for when we drop an
        auction.
    */
    virtual HttpResponse
    getDroppedAuctionResponse(const Auction & auction,
                              const std::string & reason) const;

    /** Return a stringified JSON of the response for our auction.  Default
        implementation calls getResponse() and stringifies the result.

        This version is provided as it may be more efficient in terms of
        memory allocations.

        The first element returned is the HTTP body, the second is the
        content type.
    */
    virtual HttpResponse
    getErrorResponse(const Auction & auction,
                     const std::string & errorMessage) const;

protected:
    virtual std::shared_ptr<ConnectionHandler> makeNewHandler();
    virtual std::shared_ptr<HttpAuctionHandler> makeNewHandlerShared();

    /** Probability that we will accept a given auction. */
    double acceptAuctionProbability;

    /** Time until which the exchange is enabled.  Normally this will be
        pushed forward a few seconds periodically so that everything will
        shut down if there is nothing controlling the exchange connector.
    */
    Date enabledUntil;
    
    /** Callback for a) when there is a new auction, and b) when an auction
        is finished.

        These are used to hook the exchange connector into the router.
    */
    OnAuction onNewAuction, onAuctionDone;

    /** Function to be called back when there is an auction timeout.  It is
        rare that you would want to do anything with this as the router will
        automatically know that an auction has timed out.
    */
    typedef boost::function<void (std::shared_ptr<Auction>, Date)> OnTimeout;
    OnTimeout onTimeout;

    /** Connection factory used to create HTTP connections.  Unless your
        application is very specialized, the default will work fine.
    */
    typedef boost::function<HttpAuctionHandler * ()> HandlerFactory;
    HandlerFactory handlerFactory;

    int numServingRequest_;  ///< How many connections are serving a request
    
private:
    friend class HttpAuctionHandler;

    Lock handlersLock;
    std::set<std::shared_ptr<HttpAuctionHandler> > handlers;

    void finishedWithHandler(std::shared_ptr<HttpAuctionHandler> handler);
};

} // namespace RTBKIT

