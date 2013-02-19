/* augmentation_loop.h                                              -*- C++ -*-
   Jeremy Barnes, 1 March 2012
   Copyright (c) 2012 Datacratic.  All rights reserved.

   Augmentation of bid requests.
*/

#ifndef __rtb_router__augmentation_loop_h__
#define __rtb_router__augmentation_loop_h__

#include "rtbkit/common/augmentation.h"
#include "soa/service/timeout_map.h"
#include "soa/service/zmq_endpoint.h"
#include "soa/service/typed_message_channel.h"
#include "router_types.h"
#include <boost/scoped_ptr.hpp>
#include <boost/thread/thread.hpp>
#include "soa/service/zmq.hpp"
#include "soa/service/socket_per_thread.h"
#include "soa/service/stats_events.h"
#include "jml/arch/spinlock.h"
#include <boost/thread/locks.hpp>
#include "soa/gc/gc_lock.h"


namespace RTBKIT {


/*****************************************************************************/
/* AUGMENTOR CONFIG                                                          */
/*****************************************************************************/

/** Information about a given augmentor. */
struct AugmentorInfo {
    AugmentorInfo()
        : numInFlight(0)
    {
    }

    std::string augmentorAddr;             ///< zmq socket name for it
    std::string name;                   ///< What the augmentation is called
    std::map<Id, Date> inFlight;
    int numInFlight;
};

// Information about an auction being augmented
struct AugmentationInfo {
    AugmentationInfo()
    {
    }

    AugmentationInfo(const std::shared_ptr<Auction> & auction,
                     Date lossTimeout)
        : auction(auction), lossTimeout(lossTimeout)
    {
    }

    std::shared_ptr<Auction> auction;   ///< Our copy of the auction
    Date lossTimeout;                     ///< When we send a loss if
    std::vector<GroupPotentialBidders> potentialGroups; ///< One per group
};


/*****************************************************************************/
/* AUGMENTATION LOOP                                                         */
/*****************************************************************************/

struct AugmentationLoop : public ServiceBase, public MessageLoop {

    AugmentationLoop(ServiceBase & parent,
                     const std::string & name = "augmentationLoop");
    AugmentationLoop(std::shared_ptr<ServiceProxies> proxies,
                     const std::string & name = "augmentationLoop");
    ~AugmentationLoop();
    
    typedef boost::function<void (const std::shared_ptr<AugmentationInfo> &)>
        OnFinished;

    void init();

    void start();
    void sleepUntilIdle();
    void shutdown();
    size_t numAugmenting() const;
    bool currentlyAugmenting(const Id & auctionId) const;

    void bindAugmentors(const std::string & uri);

    /** Push an auction into the augmentor.  Can be called from any thread. */
    void augment(const std::shared_ptr<AugmentationInfo> & info,
                 Date timeout,
                 const OnFinished & onFinished);

    struct Entry {
        std::shared_ptr<AugmentationInfo> info;
        std::set<std::string> outstanding;
        OnFinished onFinished;
        Date timeout;
    };

    /** List of auctions we're currently augmenting.  Once the augmentation
        process is finished the auction will be passed on.
    */
    typedef TimeoutMap<Id, std::shared_ptr<Entry> > Augmenting;
    Augmenting augmenting;

    /** Currently configured augmentors.  Indexed by the augmentor name. */
    std::map<std::string, std::shared_ptr<AugmentorInfo> > augmentors;

    /** A single entry in the augmentor info structure. */
    struct AugmentorInfoEntry {
        std::string name;
        std::shared_ptr<AugmentorInfo> info;
        //std::shared_ptr<const AugmentorConfig> config;
    };

    /** A read-only structure in which the augmentors are periodically published.
        Protected by RCU.
    */
    struct AllAugmentorInfo : public std::vector<AugmentorInfoEntry> {
        std::set<std::string> index;
    };

    /** Pointer to current version.  Protected by allAgentsGc. */
    AllAugmentorInfo * allAugmentors;

    /** RCU protection for allAgents. */
    mutable GcLock allAugmentorsGc;

    int idle_;

    /// We pick up augmentations to be done from here
    TypedMessageSink<std::shared_ptr<Entry> > inbox;

    /// Connection to all of our augmentors
    ZmqNamedClientBus toAugmentors;

    typedef ML::Spinlock Lock;
    typedef boost::unique_lock<Lock> Guard;
    mutable ML::Spinlock lock;

    /** Update the augmentors from the configuration settings. */
    void updateAllAugmentors();

    void handleAugmentorMessage(const std::vector<std::string> & message);

    void checkExpiries();

    /** Handle a configuration message from an augmentor. */
    void doConfig(const std::vector<std::string> & message);

    /** Handle a response from an augmentation. */
    void doResponse(const std::vector<std::string> & message);

    /** Handle a message asking for augmentation. */
    void doAugment(const std::vector<std::string> & message);

    void augmentationExpired(const Id & id, const Entry & entry);
};

} // namespace RTBKIT

#endif /* __rtb_router__augmentation_loop_h__ */
