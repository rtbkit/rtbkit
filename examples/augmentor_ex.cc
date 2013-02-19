/** augmentor_ex.cc                                 -*- C++ -*-
    Rémi Attab, 14 Feb 2013
    Copyright (c) 2013 Datacratic.  All rights reserved.

    Augmentor example that can be used to do extremely simplistic frequency
    capping.

*/

#include "rtbkit/core/agent_configuration/agent_configuration_listener.h"
#include "rtbkit/core/agent_configuration/agent_config.h"
#include "rtbkit/plugins/augmentor/augmentor_base.h"
#include "rtbkit/common/bid_request.h"
#include "soa/service/zmq_named_pub_sub.h"
#include "jml/utils/exc_assert.h"

#include <boost/program_options/cmdline.hpp>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/positional_options.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>

#include <unordered_map>
#include <iostream>
#include <mutex>
#include <thread>
#include <chrono>


using namespace std;


/******************************************************************************/
/* FREQUENCY CAP STORAGE                                                      */
/******************************************************************************/

/** A very primitive storage for frequency cap events.

    Ideally this should be replaced by some kind of low latency persistent
    storage (eg. Redis).

    Note that the locking scheme for this class is unlikely to scale well. A
    better scheme is left as an exercise to the reader.
 */
struct FrequencyCapStorage
{
    /** Returns the number of times an ad for the given account has been shown
        to the given user.
     */
    size_t get(const RTBKIT::AccountKey& account, const RTBKIT::UserIds& uids)
    {
        lock_guard<mutex> guard(lock);
        return counters[uids.exchangeId][account[0]];
    }

    /** Increments the number of times an ad for the given account has been
        shown to the given user.
     */
    void inc(const RTBKIT::AccountKey& account, const RTBKIT::UserIds& uids)
    {
        lock_guard<mutex> guard(lock);
        counters[uids.exchangeId][account[0]]++;
    }

private:

    mutex lock;
    unordered_map<Datacratic::Id, unordered_map<string, size_t> > counters;

};

/******************************************************************************/
/* FREQUENCY CAP AUGMENTOR                                                    */
/******************************************************************************/

/** A Simple frequency cap augmentor which limits the number of times an ad can
    be shown to a specific user. It's multithreaded and connects to the
    following services:

    - The augmentation loop for its bid request stream.
    - The post auction loop for its win notification
    - The agent configuration listener to retrieve agent configuration for the
      augmentor.
    - FrequencyCapStorage for its simplistic data repository.

 */
struct FrequencyCapAugmentor :
    public RTBKIT::SyncAugmentorBase
{

    /** Note that the serviceName and augmentorName are distinct because you may
        have multiple instances of the service that provide the same
        augmentation.
     */
    FrequencyCapAugmentor(
            std::shared_ptr<Datacratic::ServiceProxies> services,
            const string& serviceName,
            const string& augmentorName = "frequency-cap-ex") :
        SyncAugmentorBase(augmentorName, serviceName, services),
        agentConfig(getZmqContext()),
        palEvents(getZmqContext())
    {
        recordHit("up");
    }


    /** Sets up the internal components of the augmentor.

        Note that SyncAugmentorBase is a MessageLoop so we can attach all our
        other service providers to our message loop to cut down on the number of
        polling threads which in turns reduces the number of context switches.
     */
    void init()
    {
        SyncAugmentorBase::init(2 /* numThreads */);

        /* Manages all the communications with the AgentConfigurationService. */
        agentConfig.init(getServices()->config);
        addSource("FrequencyCapAugmentor::agentConfig", agentConfig);

        /* This lambda will get called when the post auction loop receives a win
           on an auction.
         */
        palEvents.messageHandler = [&] (const vector<zmq::message_t>& msg)
            {
                RTBKIT::AccountKey account(msg[19].toString());
                RTBKIT::UserIds uids =
                    RTBKIT::UserIds::createFromJson(msg[15].toString());

                storage.inc(account, uids);
                recordHit("wins");
            };

        palEvents.connectAllServiceProviders(
            "rtbPostAuctionService", "logger", {"MATCHEDWIN"});

        addSource("FrequencyCapAugmentor::palEvents", palEvents);
    }


private:

    /** Augments the bid request with our frequency cap information.

        This function has a 5ms window to respond (including network latency).
        Note that the augmentation is in charge of ensuring that the time
        constraints are respected and any late responses will be ignored.
     */
    virtual RTBKIT::AugmentationList
    onRequest(const RTBKIT::AugmentationRequest& request)
    {
        recordHit("requests");

        RTBKIT::AugmentationList result;

        const RTBKIT::UserIds& uids = request.bidRequest->userIds;

        for (const string& agent : request.agents) {

            RTBKIT::AgentConfigEntry config = agentConfig.getAgentEntry(agent);

            /* When a new agent comes online there's a race condition where the
               router may send us a bid request for that agent before we receive
               its configuration. This check keeps us safe in that scenario.
             */
            if (!config.valid()) {
                recordHit("unknownConfig");
                continue;
            }

            const RTBKIT::AccountKey& account = config.config->account;

            size_t count = storage.get(account, uids);

            /* The number of times a user has been seen by a given agent can be
               useful to make bid decisions so attach this data to the bid
               request.

               It's also recomended to place your data in an object labeled
               after the augmentor from which it originated.
             */
            result[account[0]].data[request.augmentor] = count;

            /* We tag bid requests that pass the frequency cap filtering because
               if the augmentation doesn't terminate in time or if an error
               occured, then the bid request will not receive any tags and will
               therefor be filtered out for agents that require the frequency
               capping.
             */
            if (count < getCap(request.augmentor, agent, config)) {
                result[account].tags.insert("pass-frequency-cap-ex");
                recordHit("accounts." + account[0] + ".passed");
            }
            else recordHit("accounts." + account[0] + ".capped");
        }

        return result;
    }


    /** Returns the frequency cap configured by the agent.

        This function is a bit brittle because it makes no attempt to validate
        the configuration.
     */
    size_t getCap(
            const string& augmentor,
            const string& agent,
            const RTBKIT::AgentConfigEntry& config) const
    {
        for (const auto& augConfig : config.config->augmentations) {
            if (augConfig.name != augmentor) continue;
            return augConfig.config.asInt();
        }

        /* There's a race condition here where if an agent removes our augmentor
           from its configuration while there are bid requests being augmented
           for that agent then we may not find its config. A sane default is
           good to have in this scenario.
         */

        return 0;
    }


    FrequencyCapStorage storage;

    RTBKIT::AgentConfigurationListener agentConfig;
    Datacratic::ZmqNamedMultipleSubscriber palEvents;
};


/******************************************************************************/
/* MAIN                                                                       */
/******************************************************************************/

int main(int argc, char** argv)
{
    string zookeeperUri;
    string zookeeperPrefix;

    string carbonConn;
    string carbonPrefix;

    using namespace boost::program_options;

    options_description options;
    options.add_options()
        ("zookeeper-uri,z", value<string>(&zookeeperUri),
                "URI of the zookeeper instance.")

        ("zookeeper-prefix", value<string>(&zookeeperPrefix),
                "Path prefix for zookeeper.")

        ("carbon-uri,c", value<string>(&carbonConn),
                "URI of connection to carbon daemon")

        ("carbon-prefix", value<string>(&carbonPrefix),
                "Path prefix for the carbon logging")

        ("help,h", "Print this message");


    variables_map vm;
    store(command_line_parser(argc, argv).options(options).run(), vm);
    notify(vm);

    if (vm.count("help")) {
        cerr << options << endl;
        return 1;
    }

    auto proxies = std::make_shared<Datacratic::ServiceProxies>();
    if (!carbonPrefix.empty())
        proxies->logToCarbon(carbonConn, carbonPrefix);
    if (!zookeeperPrefix.empty())
        proxies->useZookeeper(zookeeperUri, zookeeperPrefix);


    FrequencyCapAugmentor augmentor(proxies, "frequency-cap-ex");
    augmentor.init();
    augmentor.start();

    while (true) this_thread::sleep_for(chrono::seconds(10));

    // Won't ever reach this point but this is how you shutdown the augmentor.
    augmentor.shutdown();

    return 0;
}
