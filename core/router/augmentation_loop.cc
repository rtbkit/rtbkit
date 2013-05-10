/* augmentation.cc
   Jeremy Barnes, 1 March 2012
   Copyright (c) 2012 Datacratic.  All rights reserved.

   How we do auction augmentation.
*/

#include "augmentation_loop.h"
#include "jml/arch/timers.h"
#include "jml/arch/futex.h"
#include "jml/utils/vector_utils.h"
#include "jml/utils/set_utils.h"
#include "jml/arch/exception_handler.h"
#include "soa/service/zmq_utils.h"
#include <iostream>
#include <boost/make_shared.hpp>
#include "rtbkit/core/agent_configuration/agent_config.h"


using namespace std;
using namespace ML;



namespace RTBKIT {


/*****************************************************************************/
/* AUGMENTATION LOOP                                                         */
/*****************************************************************************/

AugmentationLoop::
AugmentationLoop(ServiceBase & parent,
                 const std::string & name)
    : ServiceBase(name, parent),
      allAugmentors(0),
      idle_(1),
      inbox(65536),
      toAugmentors(getZmqContext())
{
    updateAllAugmentors();
}

AugmentationLoop::
AugmentationLoop(std::shared_ptr<ServiceProxies> proxies,
                 const std::string & name)
    : ServiceBase(name, proxies),
      allAugmentors(0),
      idle_(1),
      inbox(65536),
      toAugmentors(getZmqContext())
{
    updateAllAugmentors();
}

AugmentationLoop::
~AugmentationLoop()
{
}

void
AugmentationLoop::
init()
{
    registerServiceProvider(serviceName(), { "rtbRouterAugmentation" });

    toAugmentors.init(getServices()->config, serviceName() + "/augmentors");

    toAugmentors.clientMessageHandler
        = [&] (const std::vector<std::string> & message)
        {
            //cerr << "got augmentor message " << message << endl;
            handleAugmentorMessage(message);
        };

    toAugmentors.bindTcp(getServices()->ports->getRange("augmentors"));

    toAugmentors.onConnection = [=] (const std::string & client)
        {
            cerr << "augmentor " << client << " has connected" << endl;
        };

    toAugmentors.onDisconnection = [=] (const std::string & client)
        {
            cerr << "augmentor " << client << " has disconnected" << endl;
        };

    inbox.onEvent = [&] (const std::shared_ptr<Entry> & entry)
        {
            //cerr << "got event on inbox" << endl;

            Guard guard(lock);
            Date now = Date::now();

            //cerr << "got lock on inbox" << endl;

            // TODO: wake up loop if slower...
            // TODO: DRY with other function...
            augmenting.insert(entry->info->auction->id, entry,
                              entry->timeout);

            for (auto it = entry->outstanding.begin(),
                     end = entry->outstanding.end();
                 it != end;  ++it) {

                auto & aug = *augmentors[*it];

                //cerr << "sending to " << *it << " at "
                //     << aug.agentAddr << endl;

                set<string> agents;
                const auto& bidderGroups = entry->info->potentialGroups;

                for (auto jt = bidderGroups.begin(), end = bidderGroups.end();
                     jt != end; ++jt)
                {
                    for (auto kt = jt->begin(), end = jt->end();
                         kt != end; ++kt)
                    {
                        agents.insert(kt->agent);
                    }
                }

                std::ostringstream availableAgentsStr;
                ML::DB::Store_Writer writer(availableAgentsStr);
                writer.save(agents);
                        
                // Send the message to the augmentor
                toAugmentors.sendMessage(aug.augmentorAddr,
                                         "AUGMENT", "1.0", *it,
                                         entry->info->auction->id.toString(),
                                         entry->info->auction->requestStrFormat,
                                         entry->info->auction->requestStr,
                                         availableAgentsStr.str(),
                                         Date::now());

                if (!aug.inFlight.insert
                    (make_pair(entry->info->auction->id, now))
                    .second) {
                    cerr << "warning: double augment for auction "
                         << entry->info->auction->id << endl;
                }
                else aug.numInFlight = aug.inFlight.size();
            }

            recordLevel(Date::now().secondsSince(now), "requestTimeMs");
                    
            idle_ = 0;
        };

    addSource("AugmentationLoop::inbox", inbox);
    addSource("AugmentationLoop::toAugmentors", toAugmentors);
    addPeriodic("AugmentationLoop::checkExpiries", 0.977,
                [=] (int) { checkExpiries(); });
}

void
AugmentationLoop::
start()
{
    //toAugmentors.start();
    MessageLoop::start();
}

void
AugmentationLoop::
sleepUntilIdle()
{
    while (!idle_)
        futex_wait(idle_, 0);
}

void
AugmentationLoop::
shutdown()
{
    MessageLoop::shutdown();
    toAugmentors.shutdown();
}

size_t
AugmentationLoop::
numAugmenting() const
{
    // TODO: can we get away without a lock here?
    Guard guard(lock);
    return augmenting.size();
}

bool
AugmentationLoop::
currentlyAugmenting(const Id & auctionId) const
{
    Guard guard(lock);
    return augmenting.count(auctionId);
}

void
AugmentationLoop::
bindAugmentors(const std::string & uri)
{
    try {
        toAugmentors.bind(uri.c_str());
    } catch (const std::exception & exc) {
        throw Exception("error while binding augmentation URI %s: %s",
                        uri.c_str(), exc.what());
    }
}

void
AugmentationLoop::
handleAugmentorMessage(const std::vector<std::string> & message)
{
    Guard guard(lock);

    Date now = Date::now();

    const std::string & type = message.at(1); 
    if (type == "CONFIG") {
        doConfig(message);
    }
    else if (type == "RESPONSE") {
        doResponse(message);
    }
    else throw ML::Exception("error handling unknown "
                             "augmentor message of type "
                             + type);
}

void
AugmentationLoop::
checkExpiries()
{
    //cerr << "checking expiries" << endl;

    Guard guard(lock);

    Date now = Date::now();

    for (auto it = augmentors.begin(), end = augmentors.end();
         it != end;  ++it) {

        AugmentorInfo & aug = *it->second;

        vector<Id> lostAuctions;

        for (auto jt = aug.inFlight.begin(),
                 jend = aug.inFlight.end();
             jt != jend;  ++jt) {
            if (now.secondsSince(jt->second) > 5.0) {
                cerr << "warning: augmentor " << it->first
                     << " lost auction " << jt->first
                     << endl;

                string eventName = "augmentor."
                    + it->first + ".lostAuction";
                recordEvent(eventName.c_str());

                lostAuctions.push_back(jt->first);
            }
        }
                
        // Delete all in flight that appear to be lost
        for (unsigned i = 0;  i < lostAuctions.size();  ++i)
            aug.inFlight.erase(lostAuctions[i]);
                
        string eventName = "augmentor." + it->first + ".numInFlight";
        recordEvent(eventName.c_str(), ET_LEVEL,
                    aug.inFlight.size());
    }
    
#if 0
    vector<string> deadAugmentors;

    for (unsigned i = 0;  i < deadAugmentors.size();  ++i) {
        string aug = deadAugmentors[i];
        augmentors.erase(aug);
                
        string eventName = "augmentor." + aug + ".dead";
        recordEvent(eventName.c_str());
    }

    if (!deadAugmentors.empty())
        updateAllAugmentors();
    
#endif

    auto onExpired = [&] (const Id & id,
                          const std::shared_ptr<Entry> & entry) -> Date
        {
            //++numAugmented;
            //cerr << "augmented " << ++numAugmented << " bids" << endl;

            for (auto it = entry->outstanding.begin(),
                     end = entry->outstanding.end();
                 it != end;  ++it) {
                string eventName = "augmentor." + *it
                    + ".expiredTooLate";
                recordEvent(eventName.c_str(), ET_COUNT);
            }
                
            this->augmentationExpired(id, *entry);
            return Date();
        };

    if (augmenting.earliest <= now) {
        //Guard guard(lock);
        augmenting.expire(onExpired, now);
    }

    if (augmenting.empty() && !idle_) {
        idle_ = 1;
        futex_wake(idle_);
    }

}

void
AugmentationLoop::
updateAllAugmentors()
{
    for (;;) {
        auto_ptr<AllAugmentorInfo> newInfo(new AllAugmentorInfo);

        AllAugmentorInfo * current = allAugmentors;

        for (auto it = augmentors.begin(), end = augmentors.end();
             it != end;  ++it) {
            AugmentorInfo & aug = *it->second;

            //if (!it->second.configured) continue;
            if (!it->second) continue;
            if (aug.name == "") continue;
            AugmentorInfoEntry entry;
            entry.name = aug.name;
            entry.info = it->second;
            //entry.config = aug.config;
            newInfo->push_back(entry);
        }

        // Sort they by their name
        std::sort(newInfo->begin(), newInfo->end(),
                  [] (const AugmentorInfoEntry & entry1,
                      const AugmentorInfoEntry & entry2)
                  {
                      return entry1.name < entry2.name;
                  });
        
        // Add the index
        //for (unsigned i = 0;  i < newInfo->size();  ++i) {
        //    newInfo->index[(*newInfo.get())[i].name] = i;
        //}

        if (ML::cmp_xchg(allAugmentors, current, newInfo.get())) {
            newInfo.release();
            if (current)
                allAugmentorsGc.defer([=] () { delete current; });
            break;
        }
    }
}

void
AugmentationLoop::
augment(const std::shared_ptr<AugmentationInfo> & info,
        Date timeout,
        const OnFinished & onFinished)
{
    Date now = Date::now();

    auto entry = std::make_shared<Entry>();
    entry->onFinished = onFinished;
    entry->info = info;
    entry->timeout = timeout;

    // Get a set of all augmentors
    std::set<std::string> augmentors;

    // Now go through and find all of the bidders
    for (unsigned i = 0;  i < info->potentialGroups.size();  ++i) {
        const GroupPotentialBidders & group = info->potentialGroups[i];
        for (unsigned j = 0;  j < group.size();  ++j) {
            const PotentialBidder & bidder = group[j];
            const AgentConfig & config = *bidder.config;
            for (unsigned k = 0;  k < config.augmentations.size();  ++k) {
                const std::string & name = config.augmentations[k].name;
                augmentors.insert(name);
            }
        }
    }

    //cerr << "need augmentors " << augmentors << endl;

    // Find which ones are actually available...
    GcLock::SharedGuard guard(allAugmentorsGc);
    const AllAugmentorInfo * ai = allAugmentors;
    
    ExcAssert(ai);

    auto it1 = augmentors.begin(), end1 = augmentors.end();
    auto it2 = ai->begin(), end2 = ai->end();

    while (it1 != end1 && it2 != end2) {
        if (*it1 == it2->name) {
            // Augmentor we need to run

            //cerr << "augmenting with " << it2->name << endl;

            recordEvent("augmentation.request");
            string eventName = "augmentor." + it2->name + ".request";
            recordEvent(eventName.c_str());
            
            if (it2->info->numInFlight > 3000) {
                string eventName = "augmentor." + it2->name
                    + ".skippedTooManyInFlight";
                recordEvent(eventName.c_str());
#if 0
            } else if (it2->info->lastHeartbeat.secondsUntil(now) > 2.0) {
                string eventName = "augmentor." + it2->name
                    + ".skippedNoHeartbeat";
                recordEvent(eventName.c_str());
#endif
            }
            else {
                entry->outstanding.insert(*it1);
            }

            ++it1;
            ++it2;
        }
        else if (*it1 < it2->name) {
            // Augmentor is not available
            //cerr << "augmentor " << *it1 << " is not available" << endl;
            ++it1;
        }
        else if (it2->name < *it1) {
            // Augmentor is not required
            //cerr << "augmentor " << it2->name << " is not required" << endl;
            ++it2;
        }
        else throw ML::Exception("logic error traversing augmentors");
    }

#if 0
    while (it1 != end1) {
        cerr << "augmentor " << *it1 << " is not available" << endl;
        ++it1;
    }
    
    while (it2 != end2) {
        cerr << "augmentor " << it2->name << " is not required" << endl;
        ++it2;
    }
#endif

    if (entry->outstanding.empty()) {
        // No augmentors required... run the auction straight away
        onFinished(info);
    }
    else {
        //cerr << "putting in inbox" << endl;
        inbox.push(entry);

#if 0 // optimization
        // Set up to run the augmentors
        Guard guard(lock, boost::try_to_lock_t());
        if (guard) {
            // Got the lock... put it straight in
            augmenting.insert(info->auction->id, entry, timeout);

            for (auto it = entry->outstanding.begin(),
                     end = entry->outstanding.end();
                 it != end;  ++it) {
                auto & aug = *this->augmentors[*it];

                if (!aug.inFlight.insert
                    (make_pair(info->auction->id, now))
                    .second) {
                    cerr << "warning: double augment for auction "
                         << info->auction->id << endl;
                }
                else aug.numInFlight = aug.inFlight.size();
            }

            idle_ = 0;
        }
        else {
            // Couldn't get the lock... put it on the queue
            sendMessage(toEndpoint_(), "QUEUE", entry);
        }
#endif // optimization

    }
}

void
AugmentationLoop::
doConfig(const std::vector<std::string> & message)
{
    if (message.size() != 4)
        throw ML::Exception("config message has wrong size: %zd vs 4",
                            message.size());

    const string & augmentorAddr = message[0];
    const string & version = message[2];
    const string & name = message[3];

    if (version != "1.0")
        throw ML::Exception("unknown version for config message");

    //cerr << "configuring augmentor " << name << " on " << connectTo
    //     << endl;

    string eventName = "augmentor." + name + ".configured";
    recordEvent(eventName.c_str());

    auto newInfo = std::make_shared<AugmentorInfo>();
    newInfo->name = name;
    newInfo->augmentorAddr = augmentorAddr;

    //cerr << "connecting on " << connectTo << endl;
    //info->connection();

    if (augmentors.count(name)) {
        // Grab the old version
        auto oldInfo = augmentors[name];

        // There was an old entry... wait until nobody is using it

        // First unpublish the entry
        updateAllAugmentors();

        // Now wait until nothing else can see it
        allAugmentorsGc.deferBarrier();

        augmentors.erase(name);

        //cerr << "  done removing old version" << endl;
    }

    augmentors[name] = newInfo;

    updateAllAugmentors();

    toAugmentors.sendMessage(augmentorAddr, "CONFIGOK");

    //cerr << "done updating" << endl;
}

void
AugmentationLoop::
doResponse(const std::vector<std::string> & message)
{
    recordEvent("augmentation.response");
    //cerr << "doResponse " << message << endl;
    if (message.size() != 7)
        throw ML::Exception("response message has wrong size: %zd",
                            message.size());
    const string & version = message[2];
    if (version != "1.0")
        throw ML::Exception("unknown response version");
    Date startTime = Date::parseSecondsSinceEpoch(message[3]);

    Id id(message[4]);
    const std::string & augmentor = message[5];
    const std::string & augmentation = message[6];

    ML::Timer timer;

    AugmentationList augmentationList;
    if (augmentation != "" && augmentation != "null") {
        try {
            Json::Value augmentationJson;

            JML_TRACE_EXCEPTIONS(false);
            augmentationJson = Json::parse(augmentation);
            augmentationList = AugmentationList::fromJson(augmentationJson);
        } catch (const std::exception & exc) {
            string eventName = "augmentor." + augmentor
                + ".responseParsingExceptions";
            recordEvent(eventName.c_str(), ET_COUNT);
        }
    }

    recordLevel(timer.elapsed_wall(), "responseParseTimeMs");

    {
        double timeTakenMs = startTime.secondsUntil(Date::now()) * 1000.0;
        string eventName = "augmentor." + augmentor + ".timeTakenMs";
        recordEvent(eventName.c_str(), ET_OUTCOME, timeTakenMs);
    }

    {
        double responseLength = augmentation.size();
        string eventName = "augmentor." + augmentor + ".responseLengthBytes";
        recordEvent(eventName.c_str(), ET_OUTCOME, responseLength);
    }

    string eventName = ML::format("augmentor.%s.%s",
                                  augmentor.c_str(),
                                  (augmentation == "" || augmentation == "null"
                                   ? "nullResponse" : "validResponse"));
    recordEvent(eventName.c_str());

    // Modify the augmentor data structures
    //Guard guard(lock);

    if (augmentors.count(augmentor)) {
        auto & entry = *augmentors[augmentor];
        entry.inFlight.erase(id);
        entry.numInFlight = entry.inFlight.size();
    }

    auto it = augmenting.find(id);
    if (it == augmenting.end()) {
        recordEvent("augmentation.unknown");
        string eventName = "augmentor." + augmentor + ".unknown";
        recordEvent(eventName.c_str());
        //cerr << "warning: handled response for unknown auction" << endl;
        return;
    }

    it->second->info->auction->augmentations.mergeWith(augmentationList);

    it->second->outstanding.erase(augmentor);
    if (it->second->outstanding.empty()) {
        it->second->onFinished(it->second->info);
        augmenting.erase(it);
    }
}

void
AugmentationLoop::
augmentationExpired(const Id & id, const Entry & entry)
{
    entry.onFinished(entry.info);
}                     

} // namespace RTBKIT
