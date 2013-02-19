/* exchange_connector.h                                            -*- C++ -*-
   Jeremy Barnes, 13 December 2012
   Copyright (c) 2012 Datacratic.  All rights reserved.

   Exchange connector.
*/

#pragma once

#include "soa/service/service_base.h"
#include "rtbkit/common/auction.h"

namespace RTBKIT {

class Router;


/*****************************************************************************/
/* EXCHANGE CONNECTOR                                                        */
/*****************************************************************************/

/** Base class to connect to exchanges.  This class is owned by a router.

    This provides:
    1.  Callbacks that can be used to inject an auction and a win into the
        router;
    2.  Interfaces for the router to control the exchange connector, such as
        cut off or throttle bids.
*/

struct ExchangeConnector: public ServiceBase {

    ExchangeConnector(const std::string & name,
                      ServiceBase & parent);
    ExchangeConnector(const std::string & name,
                      std::shared_ptr<ServiceProxies>
                          = std::shared_ptr<ServiceProxies>());

    virtual ~ExchangeConnector();

    /** Set the router used by the exchange connector. */
    void setRouter(Router * router);

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
    virtual void enableUntil(Date date) = 0;

    /** Set which percentage of bid requests will be accepted by the
        exchange connector.
    */
    virtual void setAcceptBidRequestProbability(double prob) = 0;

    /*************************************************************************/
    /* FACTORY INTERFACE                                                     */
    /*************************************************************************/

    /** Type of a callback which is registered as an exchange factory. */
    typedef std::function<ExchangeConnector * (Router * owner,std::string name)>
        Factory;
    
    /** Register the given exchange factory. */
    static void registerFactory(const std::string & exchange, Factory factory);

    /** Create a new exchange connector from a factory. */
    static std::unique_ptr<ExchangeConnector>
    create(const std::string & exchangeType,
           std::shared_ptr<Router> owner,
           const std::string & name);

    /** Start up a new exchange and connect it to the router.  The exchange
        will read its configuration from the given JSON blob.
    */
    static void startExchange(std::shared_ptr<Router> router,
                              const std::string & exchangeType,
                              const Json::Value & exchangeConfig);

    /** Start up a new exchange and connect it to the router.  The exchange
        will read its configuration and type from the given JSON blob.
    */
    static void startExchange(std::shared_ptr<Router> router,
                              const Json::Value & exchangeConfig);

protected:
    /** The router that the exchange is connected to. */
    Router * router;
};


} // namespace RTBKIT
    
