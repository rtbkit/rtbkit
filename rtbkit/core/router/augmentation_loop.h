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

/** Information about a specific augmentor which belongs to an augmentor class.
 */
struct AugmentorInstanceInfo {
    AugmentorInstanceInfo(const std::string& addr = "", int maxInFlight = 0) :
        addr(addr), numInFlight(0), maxInFlight(maxInFlight)
    {}

    std::string addr;
    int numInFlight;
    int maxInFlight;
};

/** Information about a given class of augmentor. */
struct AugmentorInfo {
    AugmentorInfo(const std::string& name = "") : name(name) {}

    std::string name;                   ///< What the augmentation is called
    std::vector<std::shared_ptr<AugmentorInstanceInfo>> instances;

    std::shared_ptr<AugmentorInstanceInfo> findInstance(const std::string& addr)
    {
        for (auto it = instances.begin(), end = instances.end();
             it != end; ++it)
        {
            auto & info = *it;
            if (info->addr == addr) return info;
        }
        return nullptr;
    }
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

    void bindAugmentors(const std::string & uri);

    /** Push an auction into the augmentor.  Can be called from any thread. */
    void augment(const std::shared_ptr<AugmentationInfo> & info,
                 Date timeout,
                 const OnFinished & onFinished);

private:

    struct Entry {
        std::shared_ptr<AugmentationInfo> info;
        std::set<std::string> outstanding;
        // We need to keep a list of current outstanding instances in our entry
        // to be able to decrement the inFlight count when expiring an entry
        // (after a timeout)
        //
        // Note that we are keeping a weak_ptr in the case where the instance
        // either disconnects or crashes. Keeping a weak_ptr prevents us from
        // possibly keeping a dangling pointer
        std::map<std::string, std::weak_ptr<AugmentorInstanceInfo>> instances;
        std::map<std::string, std::set<std::string> > augmentorAgents;
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
    };

    /** A read-only structure in which the augmentors are periodically published.
        Protected by RCU.
    */
    typedef std::vector<AugmentorInfoEntry> AllAugmentorInfo;

    /** Pointer to current version.  Protected by allAgentsGc. */
    AllAugmentorInfo * allAugmentors;

    /** RCU protection for allAgents. */
    mutable GcLock allAugmentorsGc;

    int idle_;

    /// We pick up augmentations to be done from here
    TypedMessageSink<std::shared_ptr<Entry> > inbox;
    TypedMessageSink<std::string> disconnections;

    /// Connection to all of our augmentors
    ZmqNamedClientBus toAugmentors;

    /** Update the augmentors from the configuration settings. */
    void updateAllAugmentors();


    void handleAugmentorMessage(const std::vector<std::string> & message);

    std::shared_ptr<AugmentorInstanceInfo> pickInstance(AugmentorInfo& aug);
    void doAugmentation(std::shared_ptr<Entry>&& entry);

    void recordStats();

    void checkExpiries();

    /** Handle a configuration message from an augmentor. */
    void doConfig(const std::vector<std::string> & message);

    /** Disconnect the instance at addr for type aug. */
    void doDisconnection(const std::string & addr, const std::string & aug = "");

    /** Handle a response from an augmentation. */
    void doResponse(const std::vector<std::string> & message);

    /** Handle a message asking for augmentation. */
    void doAugment(const std::vector<std::string> & message);

    void augmentationExpired(const Id & id, const Entry & entry);
};

} // namespace RTBKIT

#endif /* __rtb_router__augmentation_loop_h__ */
