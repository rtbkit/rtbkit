/* router_base.h                                                   -*- C++ -*-
   Jeremy Barnes, 29 May 2012
   Copyright (c) 2012 Datacratic.  All rights reserved.

   Base class for the router containing basic functionality.
*/

#pragma once

#include "soa/service/service_base.h"
#include "rtbkit/core/agent_configuration/agent_config.h"
#include "router_types.h"
#include "soa/gc/gc_lock.h"
#include "soa/logger/logger.h"
#include "soa/service/timeout_map.h"
#include "soa/service/zmq_named_pub_sub.h"
#include <unordered_map>


namespace RTBKIT {


/*****************************************************************************/
/* DEBUG INFO                                                                */
/*****************************************************************************/

struct AuctionDebugInfo {
    void addAuctionEvent(Date date, std::string type,
                         const std::vector<std::string> & args);
    void addSpotEvent(const Id & spot, Date date, std::string type,
                      const std::vector<std::string> & args);
    void dumpAuction() const;
    void dumpSpot(Id spot) const;

    struct Message {
        Date timestamp;
        Id spot;
        std::string type;
        std::vector<std::string> args;
    };

    std::vector<Message> messages;
};

/*****************************************************************************/
/* ROUTER SHARED                                                             */
/*****************************************************************************/

struct RouterShared {

    RouterShared(std::shared_ptr<zmq::context_t> context);

    ~RouterShared()
    {
        shutdown();
    }

    void shutdown()
    {
        logger.shutdown();
    }

    ZmqNamedPublisher logger;

    /** Debug only */
    bool doDebug;

    mutable ML::Spinlock debugLock;
    TimeoutMap<Id, AuctionDebugInfo> debugInfo;

    uint64_t numAuctions;
    uint64_t numBids;
    uint64_t numNonEmptyBids;
    uint64_t numAuctionsWithBid;
    uint64_t numWins;
    uint64_t numLosses;
    uint64_t numImpressions;
    uint64_t numClicks;
    uint64_t numNoPotentialBidders;
    uint64_t numNoBidders;

    bool simulationMode_;
    Date simulatedTime_;
};


/*****************************************************************************/
/* ROUTER BASE                                                               */
/*****************************************************************************/

struct RouterServiceBase : public ServiceBase {

    RouterServiceBase(const std::string & serviceName,
                      ServiceBase & parent)
        : ServiceBase(serviceName, parent)
    {
        shared.reset(new RouterShared(parent.getZmqContext()));
    }

    RouterServiceBase(const std::string & serviceName,
                      std::shared_ptr<ServiceProxies> services)
        : ServiceBase(serviceName, services)
    {
        shared.reset(new RouterShared(getZmqContext()));
    }

    std::shared_ptr<RouterShared> shared;

    /*************************************************************************/
    /* EXCEPTIONS                                                            */
    /*************************************************************************/

    /** Throw an exception and log the error in Graphite and in the router
        log file.
    */
    void throwException(const std::string & key, const std::string & fmt,
                        ...) __attribute__((__noreturn__));


    /*************************************************************************/
    /* SYSTEM LOGGING                                                        */
    /*************************************************************************/

    /** Log a router error. */
    template<typename... Args>
    void logRouterError(const std::string & function,
                        const std::string & exception,
                        Args... args)
    {
        shared->logger.publish("ROUTERERROR", Date::now().print(5),
                               function, exception, args...);
        recordHit("error.%s", function);
    }


    /*************************************************************************/
    /* DATA LOGGING                                                          */
    /*************************************************************************/

    /** Log a given message to the given channel. */
    template<typename... Args>
    void logMessage(const std::string & channel, Args... args)
    {
        using namespace std;
        //cerr << "********* logging message to " << channel << endl;
        shared->logger.publish(channel, Date::now().print(5), args...);
    }

    /** Log a given message to the given channel. */
    template<typename... Args>
    void logMessageNoTimestamp(const std::string & channel, Args... args)
    {
        using namespace std;
        //cerr << "********* logging message to " << channel << endl;
        shared->logger.publish(channel, args...);
    }

    /*************************************************************************/
    /* DEBUGGING                                                             */
    /*************************************************************************/

    void debugAuction(const Id & auction, const std::string & type,
                      const std::vector<std::string> & args
                      = std::vector<std::string>())
    {
        if (JML_LIKELY(!shared->doDebug)) return;
        debugAuctionImpl(auction, type, args);
    }

    void debugAuctionImpl(const Id & auction, const std::string & type,
                          const std::vector<std::string> & args);

    void debugSpot(const Id & auction,
                   const Id & spot,
                   const std::string & type,
                   const std::vector<std::string> & args
                       = std::vector<std::string>())
    {
        if (JML_LIKELY(!shared->doDebug)) return;
        debugSpotImpl(auction, spot, type, args);
    }

    void debugSpotImpl(const Id & auction,
                       const Id & spot,
                       const std::string & type,
                       const std::vector<std::string> & args);

    void expireDebugInfo();

    void dumpAuction(const Id & auction) const;
    void dumpSpot(const Id & auction, const Id & spot) const;


    /*************************************************************************/
    /* TIME                                                                  */
    /*************************************************************************/

    Date getCurrentTime() const;
    void setSimulatedTime(Date newTime);

};


} // namespace RTBKIT
