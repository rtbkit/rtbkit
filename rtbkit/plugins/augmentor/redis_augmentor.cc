/*
 * redis_augmentor.cpp
 *
 *  Created on: Aug 7, 2013
 *      Author: Jan Sulmont
 *      Copyright (c) 2013 Datacratic.  All rights reserved.
 */

#include <iterator> // std::back_inserter
#include <algorithm>// std::copy_if
#include <boost/range/irange.hpp>
#include "redis_augmentor.h"
#include "jml/utils/exc_assert.h"
using namespace std;

namespace RTBKIT {


RedisAugmentor::
~RedisAugmentor()
{
}

/** Sets up the internal components of the augmentor.

    Note that AsyncAugmentorBase is a MessageLoop so we can attach all our
    other service providers to our message loop to cut down on the number of
    polling threads which in turns reduces the number of context switches.
*/
void
RedisAugmentor::
init(int nthreads)
{
    AsyncAugmentor::init(nthreads);
    /* Manages all the communications with the AgentConfigurationService. */
    agent_config_.init(getServices()->config);
    addSource("RedisAugmentor::agentConfig", agent_config_);
}


void
RedisAugmentor::
onRequest(const AugmentationRequest & request, SendResponseCB sendResponse)
{
    ML::Timer tm;

    recordHit("requests");

    // we build an *ordered* map indexed by Redis keys, pointing
    // at set of account keys. It will be captured by copy by the
    // Redis lambda call back, and used in order to build the
    // augmentation list.
    map<string,set<RTBKIT::AccountKey>> jobs;
    auto br = request.bidRequest->toJson();
    for (const string& agent : request.agents)
    {
        RTBKIT::AgentConfigEntry c  = agent_config_.getAgentEntry(agent);

        /* When a new agent comes online there's a race condition where the
           router may send us a bid request for that agent before we receive
           its configuration. This check keeps us safe in that scenario. */
        if (!c.valid())
        {
            recordHit("unknownConfig");
            continue;
        }

        // FIXME avoid converting the thing in Json::Value
        const auto& aug_c = c.config->toJson(false);
        const auto& aug_l = aug_c.atStr("augmentations").atStr("redis").atStr("config").atStr("aug-list");
        if (!aug_l || aug_l.type() != Json::arrayValue || !aug_l.size())
        {
            recordHit ("noRedisAugAgentConfig");
            continue ;
        }
        int n = aug_l.size();
        static const string prefix = "RTBkit:aug" ;
        for (auto i: boost::irange (0,n))
        {
            auto key = aug_l.atIndex(i).asString();
            if (key.empty()) continue;
            // prefix root path (.) if absent.
            auto root_key = key[0] == '.' ? key : "."+key;
            Json::Value v = Json::Path(root_key).make(br);
            if (!v) continue;
            auto v_str = v.toString();
            string vv_str ;
            copy_if(v_str.begin(), v_str.end(),  back_inserter(vv_str), [](const char& c) {
                return c!='\n'&&c!='"';
            });
            jobs[prefix+":"+key+":"+vv_str].insert (c.config->account);
        }
    }

    if (jobs.empty())
    {
        recordHit ("noRedisKeys");
        sendResponse(AugmentationList());
        return;
    }

    auto doResponse = [=](const Redis::Results& results) {
        ExcAssertEqual (results.size(), jobs.size());
        AugmentationList auglret;
        if (results)
        {
            auto i=0;
            for (const auto& ii: jobs)
            {
                const auto& res = results.at(i).reply().asString();
                if (!res.empty())
                    for (const auto& jj: ii.second)
                        auglret[jj].data.atStr(ii.first) = res;
                ++i;
            }
        }
        else
        {
            cerr << "RedisAugmentor::onRequest::lambda(doResponse) error: " << results.error() << endl ;
            recordHit("redisError."+results.error());
        }
        recordOutcome(tm.elapsed_wall() * 1000.0, "redisResponseMs");
        sendResponse(auglret);
    };

    // build a vector of commands
    vector<Redis::Command> cmds;
    for (auto& ii: jobs)
        cmds.emplace_back (Redis::GET(ii.first));

    // and post it
    redis_->queueMulti(cmds, doResponse, 0.004);

}
} /* namespace RTBKIT */
