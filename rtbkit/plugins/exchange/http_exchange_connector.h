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
#include "rtbkit/common/exchange_connector.h"
#include <boost/algorithm/string.hpp>


namespace RTBKIT {

struct HttpExchangeConnector;
struct HttpAuctionLogger;
struct HttpAuctionHandler;

/*****************************************************************************/
/* HTTP EXCHANGE CONNECTOR                                                   */
/*****************************************************************************/

/** Class that manages incoming auctions. */

struct HttpExchangeConnector
    : public ExchangeConnector,
      public HttpEndpoint {

    HttpExchangeConnector(const std::string & name,
                          ServiceBase & parent);

    HttpExchangeConnector(const std::string & name,
                          std::shared_ptr<ServiceProxies> proxies);

    ~HttpExchangeConnector();

    virtual Json::Value getServiceStatus() const;

    /** Start logging requests */
    void startRequestLogging(std::string const & filename, int count = 1000);

    /** Stop logging */
    void stopRequestLogging();

    /*************************************************************************/
    /* METHODS CALLED BY THE ROUTER TO CONTROL THE EXCHANGE CONNECTOR        */
    /*************************************************************************/

    /** Configure the exchange connector.  The JSON provided is entirely
        interpreted by the exchange connector itself.
    */
    virtual void configure(const Json::Value & parameters);

    /** Configure just the HTTP part of the server. */
    void configureHttp(int numThreads,
                       const PortRange & listenPort,
                       const std::string & bindHost = "*",
                       bool performNameLookup = false,
                       int backlog = DEF_BACKLOG,
                       const std::string & auctionResource = "/auctions",
                       const std::string & auctionVerb = "POST",
                       int realTimePriority = -1,
                       bool realTimePolling = false,
                       double absoluteTimeMax = 50.0);

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

    /** Returns a function that can be used to sample the load of the exchange
        connector. See LoopMonitor documentation for more details.
     */
    virtual std::function<double(double)> getLoadSampleFn() const
    {
        // This gets captured by value and effectively becomes a state variable
        // for the mutable lambda. Not very clean but it works.
        std::vector<rusage> lastSample;
        auto lastTime = Date::now();

        return [=] (double elapsed) mutable -> double {
            auto sample = getResourceUsage();

            // get how much time elapsed since last time
            auto now = Date::now();
            auto dt = now.secondsSince(lastTime);
            lastTime = now;

            //first time?
            if(sample.size() != lastSample.size()) {
                lastSample = std::move(sample);
                return 0.0;
            }

            double sum = 0.0;
            for (auto i = 0; i < sample.size(); i++) {
                auto sec = double(sample[i].ru_utime.tv_sec - lastSample[i].ru_utime.tv_sec);
                auto usec = double(sample[i].ru_utime.tv_usec - lastSample[i].ru_utime.tv_usec);

                auto load = sec + usec * 0.000001;
                if (load >= dt) {
                    sum += 1.0;
                } else {
                    sum += load/dt;
                }
            }

            double value = sum / sample.size();
            lastSample = std::move(sample);
            return value;
        };
    }


    /*************************************************************************/
    /* METHODS TO OVERRIDE FOR A GIVEN EXCHANGE                              */
    /*************************************************************************/

    /** Return the name of the exchange, as it would be written as an
        identifier.
    */
    virtual std::string exchangeName() const = 0;

    /** Parse the given payload into a bid request. */
    virtual std::shared_ptr<BidRequest>
    parseBidRequest(HttpAuctionHandler & connection,
                    const HttpHeader & header,
                    const std::string & payload);

    /** This method is called right after the bid request has been parsed and
     *  the Auction object has been created. This method should be reimplemented
     *  if you want to modify the auction before it is injected into the router
     *
     *  The default implementation of this function does nothing
     *
     *  @Postcondition: the auction must not be null
     */
    virtual void
    adjustAuction(std::shared_ptr<Auction>& auction) const;


    /** Return the available time for the bid request in milliseconds.  This
        method should not parse the bid request, as when shedding load
        we want to do as little work as possible.

        Most exchanges include this information in the HTTP headers.
    */
    virtual double
    getTimeAvailableMs(HttpAuctionHandler & connection,
                       const HttpHeader & header,
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
    getRoundTripTimeMs(HttpAuctionHandler & connection,
                       const HttpHeader & header);

    /** Return the HTTP response for our auction.  Default
        implementation calls getResponse() and stringifies the result.

        This version is provided as it may be more efficient in terms of
        memory allocations.

        The first element returned is the HTTP body, the second is the
        content type.
    */
    virtual HttpResponse
    getResponse(const HttpAuctionHandler & connection,
                const HttpHeader & requestHeader,
                const Auction & auction) const;

    /** Return a stringified JSON of the response for when we drop an
        auction.
    */
    virtual HttpResponse
    getDroppedAuctionResponse(const HttpAuctionHandler & connection,
                              const std::string & reason) const;

    /** Return a stringified JSON of the response for our auction.  Default
        implementation calls getResponse() and stringifies the result.

        This version is provided as it may be more efficient in terms of
        memory allocations.

        The first element returned is the HTTP body, the second is the
        content type.
    */
    virtual HttpResponse
    getErrorResponse(const HttpAuctionHandler & connection,
                     const std::string & errorMessage) const;

    /** Handles a request to a resource other than the auctionResource that
        is specified in the configuration.  This can be used by exchange
        connectors to handle things like ready requests.

        This method should always write a response on the connection.
        
        Default will return a 404.
    */
    virtual void
    handleUnknownRequest(HttpAuctionHandler & connection,
                         const HttpHeader & header,
                         const std::string & payload) const;

    /** Given an agent configuration, return a structure that describes
        the compatibility of each campaign and creative with the
        exchange.

        If includeReasons is true, then the reasons structure should be
        filled in with a list of reasons for which the exchange rejected
        the creative or campaign.  If includeReasons is false, the reasons
        should be all empty to save memory allocations.  Note that it
        doesn't make much sense to have the reasons non-empty for creatives
        or campaigns that are approved.

        The default implementation assumes that all campaigns and
        creatives are compatible with the exchange.
    */
    virtual ExchangeCompatibility
    getCampaignCompatibility(const AgentConfig & config,
                             bool includeReasons) const;
    
    /** Tell if a given creative is compatible with the given exchange.
        See getCampaignCompatibility().
    */
    virtual ExchangeCompatibility
    getCreativeCompatibility(const Creative & creative,
                             bool includeReasons) const;


    /** Method invoked every second for accounting */
    virtual void periodicCallback(uint64_t numWakeups) const;

protected:
    virtual std::shared_ptr<ConnectionHandler> makeNewHandler();
    virtual std::shared_ptr<HttpAuctionHandler> makeNewHandlerShared();

    /** Time until which the exchange is enabled.  Normally this will be
        pushed forward a few seconds periodically so that everything will
        shut down if there is nothing controlling the exchange connector.
    */
    Date enabledUntil;
    
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

    int numServingRequest;  ///< How many connections are serving a request

    /// Configuration parameters
    int numThreads;
    int realTimePriority;
    PortRange listenPort;
    std::string bindHost;
    bool performNameLookup;
    int backlog;
    std::string auctionResource;
    std::string auctionVerb;
    double absoluteTimeMax;
    bool disableAcceptProbability;
    bool disableExceptionPrinting;

    /// The ping time to known hosts in milliseconds
    std::unordered_map<std::string, float> pingTimesByHostMs;

    /// The ping time to assume for unknown hosts
    float pingTimeUnknownHostsMs;
    
private:
    friend class HttpAuctionHandler;

    std::shared_ptr<HttpAuctionLogger> logger;

    Lock handlersLock;
    std::set<std::shared_ptr<HttpAuctionHandler> > handlers;
    void finishedWithHandler(std::shared_ptr<HttpAuctionHandler> handler);

    /** Common code from all constructors. */
    void postConstructorInit();
};

} // namespace RTBKIT

