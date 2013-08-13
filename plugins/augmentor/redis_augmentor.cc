/*
 * redis_augmentor.cpp
 *
 *  Created on: Aug 7, 2013
 *      Author: Jan Sulmont
 *      Copyright (c) 2013 Datacratic.  All rights reserved.
 */

#include <atomic>
#include <boost/range/irange.hpp>
#include <boost/algorithm/string.hpp>
#include "redis_augmentor.h"

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

    // const RTBKIT::UserIds& uids = request.bidRequest->userIds;

    for (const string& agent : request.agents)
    {
        RTBKIT::AgentConfigEntry c  = agent_config_.getAgentEntry(agent);

        /* When a new agent comes online there's a race condition where the
           router may send us a bid request for that agent before we receive
           its configuration. This check keeps us safe in that scenario.
        */
        if (!c.valid())
        {
            recordHit("unknownConfig");
            continue;
        }

        // needed in the lambda below
        //const RTBKIT::AccountKey& account = c.config->account;

        // Will be called later on.
        auto doResponse = [=](const Redis::Results& results) {
            static std::atomic<uint64_t> counter;
            if (results)
            {
                AugmentationList alresp ;
//                auto reply = result.reply();
//                alresp[account].data = reply.asString();
//                sendResponse(alresp);
                recordOutcome(tm.elapsed_wall() * 1000.0, "redisResponseMs");
            }
            else
            {
                cerr << "RedisAugmentor::onRequest::lambda(doResponse) error: " << results.error() << endl ;
                recordHit("redisError."+results.error());
            }
        };

        // FIXME avoid converting the thing in Json::Value
        const auto& aug_c = c.config->toJson(false);
        const auto& aug_l = aug_c.atStr("augmentations").atStr("redis").atStr("config").atStr("aug-list");
        if (!aug_l || aug_l.type() != Json::arrayValue || !aug_l.size())
        {
        	recordHit ("noRedisAugAgentConfig");
        	continue ;
        }
        // now try to build an array of redis commands
        auto br = request.bidRequest->toJson();
        vector<Redis::Command> cmds ;
        int n = aug_l.size();
        static const string prefix = "rtbkit:redis" ;

        for (auto i: boost::irange (0,n))
        {
        	auto key = aug_l.atIndex(i).asString();
        	if (key.empty()) continue;
        	if (key[0] != '.') key = "."+key ;
        	cerr << "key = " << key << endl ;
        	Json::Value v = Json::Path(key).make(br);
        	if (!v) continue;
        	string v_str;
        	switch (v.type())
        	{
        	case Json::stringValue: v_str = v.asString(); break;
        	case Json::intValue: v_str = to_string(v.asInt()); break;
        	case Json::uintValue: v_str = to_string(v.asUInt()); break;
        	case Json::realValue: v_str = to_string(v.asDouble()); break;
        	default: v_str = v.toString();
        	}
        	cmds.emplace_back( Redis::GET(prefix+":"+key+":"+v_str));
        }
        if (cmds.empty())
        {
        	recordHit ("noRedisAugKeyFound");
        	continue;
        }
        redis_.queueMulti(cmds, doResponse, 0.004);
    }
}
} /* namespace RTBKIT */
