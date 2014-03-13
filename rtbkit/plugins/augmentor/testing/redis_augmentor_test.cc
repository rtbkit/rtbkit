/** augmentor_stress_test.cc                                 -*- C++ -*-
    Rémi Attab, 10 May 2013
    Copyright (c) 2013 Datacratic.  All rights reserved.

    Stress test to worm out the bottleneck in augmentor base.

*/

#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include "rtbkit/testing/test_agent.h"
#include "rtbkit/plugins/augmentor/redis_augmentor.h"
#include "rtbkit/core/agent_configuration/agent_configuration_service.h"
#include "soa/service/testing/redis_temporary_server.h"
#include "jml/db/persistent.h"
#include "jml/utils/exc_assert.h"
#include "soa/service/redis.h"

#include <boost/test/unit_test.hpp>
#include <mutex>
#include <atomic>
#include <set>

using namespace std;
using namespace ML;
using namespace RTBKIT;


static const string sampleBr =
    "{\"id\":\"85885bb0-b91b-11e2-c4cf-7fba90171555\",\"timestamp\":1368153863.008756,\"isTest\":false,\"url\":\"http://myonlinearcade.com/\",\"ipAddress\":\"166.13.20.21\",\"userAgent\":\"Mozilla/5.0 (Windows NT 6.1; WOW64; rv:19.0) Gecko/20100101 Firefox/20.0\",\"language\":\"fr\",\"protocolVersion\":\"0.3\",\"exchange\":\"appnexus\",\"provider\":\"appnexus\",\"winSurcharges\":{\"surcharge\":{\"USD/1M\":50}},\"winSurchageMicros\":{\"surcharge\":{\"USD/1M\":50}},\"location\":{\"countryCode\":\"CA\",\"regionCode\":\"QC\",\"cityName\":\"Laval\",\"postalCode\":\"0\",\"dma\":0,\"timezoneOffsetMinutes\":-1},\"segments\":{\"appnexus\":[\"memberId1357\"],\"browser\":[\"Mozilla Firefox\"],\"device_type\":[\"Computer\"],\"os\":[\"Microsoft Windows 7\"]},\"userIds\":{\"an\":\"5273283952213481305\",\"xchg\":\"5273283952213481305\"},\"imp\":[{\"id\":\"156331815539876686\",\"banner\":{\"w\":728,\"h\":90},\"formats\":[\"728x90\"]}],\"spots\":[{\"id\":\"156331815539876686\",\"banner\":{\"w\":728,\"h\":90},\"formats\":[\"728x90\"]}]}";

static std::atomic<size_t> instances(0);
std::string instancedName(const std::string& prefix)
{
    return prefix + to_string(instances.fetch_add(1));
}

static mutex aug_mtx ;
static const string aug_str =
    "[{\"account\":[\"aliceCampaign\",\"aliceStrategy\"],\"augmentation\":{\"data\":{\"RTBkit:aug:id:85885bb0-b91b-11e2-c4cf-7fba90171555\":\"9876\",\"RTBkit:aug:url:http://myonlinearcade.com/\":\"JSCRIPT\"}}},{\"account\":[\"testCampaign\",\"testStrategy\"],\"augmentation\":{\"data\":{\"RTBkit:aug:id:85885bb0-b91b-11e2-c4cf-7fba90171555\":\"9876\",\"RTBkit:aug:winSurcharges.surcharge.USD/1M:50\":\"123\"}}}]";
static vector<string> aug_vec ;
struct MockAugmentationLoop : public ServiceBase, public MessageLoop
{
    MockAugmentationLoop(const std::shared_ptr<ServiceProxies>& proxies) :
        ServiceBase(instancedName("mock-aug-loop-"), proxies),
        toAug(proxies->zmqContext),
        sent(0), recv(0)
    {}

    void start()
    {
        registerServiceProvider(serviceName(), { "rtbRouterAugmentation" });

        toAug.init(getServices()->config, serviceName() + "/augmentors");
        toAug.bindTcp(getServices()->ports->getRange("augmentors"));

        toAug.clientMessageHandler = [&] (const vector<string> & message) {
            ExcAssertEqual (message[0], "redis-augmentation");
            if (message[1] == "RESPONSE")
            {
                ExcAssertEqual (message.size(), 7);
                ExcAssertEqual (message[5], "redis-augmentation");
                recordHit("recv");
                lock_guard<mutex> l(aug_mtx);
                aug_vec.emplace_back (message[6]);
                recv++;
            }
        };

        toAug.onConnection = [=] (const std::string & client) {
            cerr << "augmentor " << client << " has connected" << endl;
        };
        toAug.onDisconnection = [=] (const std::string & client) {
            cerr << "augmentor " << client << " has disconnected" << endl;
        };

        addSource("MockAugLoop::toAug", toAug);


        {
            // FIXME: should use Agent Config
            set<string> agents {
                "bob-the-agent_" + to_string(getpid()),
                "alice-the-agent_" + to_string(getpid())
            };
            std::ostringstream agentStr;
            ML::DB::Store_Writer writer(agentStr);
            writer.save(agents);
            this->agents = agentStr.str();
        }

        addPeriodic("MockAugLoop::send", 0.9, [=] (uint64_t) {
            toAug.sendMessage(
                "redis-augmentation", "AUGMENT", "1.0", "redis-augmentation",
                to_string(random()), "datacratic", sampleBr,
                agents, Date::now());

            sent++;
        });

        MessageLoop::start();
    }

    ZmqNamedClientBus toAug;
    string agents;
    size_t sent, recv;
};



BOOST_AUTO_TEST_CASE( redisAugmentorTest )
{
    enum {
        FeederThreads = 1,
        TestLength = 5,
        RedisThreads = 2
    };

    Redis::RedisTemporaryServer redis;
    {
        using namespace Redis;
        AsyncConnection async_redis(redis);
        // set a few keys
        Command mset(MSET);
        mset.addArg("RTBkit:aug:winSurcharges.surcharge.USD/1M:50");
        mset.addArg(123.45);
        mset.addArg("RTBkit:aug:id:85885bb0-b91b-11e2-c4cf-7fba90171555");
        mset.addArg(9876);
        mset.addArg("RTBkit:aug:url:http://myonlinearcade.com/");
        mset.addArg("JSCRIPT");

        Result result = async_redis.exec(mset);
        BOOST_CHECK_EQUAL(result.ok(), true);
    }

    auto proxies = make_shared<ServiceProxies>();

    AgentConfigurationService agentConfig(proxies, "config");
    agentConfig.unsafeDisableMonitor();
    agentConfig.init();
    agentConfig.bindTcp();
    agentConfig.start();

    TestAgent agent1(proxies, "bob-the-agent");
    agent1.init();
    agent1.start();
    agent1.configure();
    {
        AugmentationConfig aug_conf;
        aug_conf.name = "redis";
        aug_conf.required = true;
        auto& v =  aug_conf.config;
        Json::Value av(Json::arrayValue);
        av.append("winSurcharges.surcharge.USD/1M");
        av.append("id");
        av.append("exchange");
        av.append("foo.bar"); // not found
        v["aug-list"] = av;
        v["aug-prefix"] = "RTBkit:aug";
        agent1.config.addAugmentation(aug_conf);
    }
    agent1.doConfig (agent1.config);

    TestAgent agent2(proxies, "alice-the-agent");
    agent2.config.account = {"aliceCampaign", "aliceStrategy"};
    agent2.init();
    agent2.start();
    agent2.doConfig(agent2.config);
    {
        AugmentationConfig aug_conf;
        aug_conf.name = "redis";
        aug_conf.required = true;
        auto& v =  aug_conf.config;
        Json::Value av(Json::arrayValue);
        av.append("id");
        av.append("url"); // not found
        v["aug-list"] = av;
        v["aug-prefix"] = "RTBkit:aug";
        agent2.config.addAugmentation(aug_conf);
    }
    agent2.doConfig (agent2.config);

    cerr << "init feeders\n";

    vector< std::shared_ptr<MockAugmentationLoop> > feederThreads;
    for (size_t i = 0; i < FeederThreads; ++i)
    {
        feederThreads.emplace_back(new MockAugmentationLoop(proxies));
        feederThreads.back()->start();
    }

    cerr << "init aug\n";

    RedisAugmentor aug("redis-augmentation", "redis-augmentation", proxies, redis);
    aug.init(RedisThreads);
    aug.start();

    this_thread::sleep_for(chrono::milliseconds(100));

    cerr << "sleeping..." << endl;

    for (size_t i = 0; i < TestLength; ++i)
    {
        this_thread::sleep_for(chrono::seconds(1));
        cerr << "[ " << i << " / " << TestLength << " ]: "
             << "load=" << aug.sampleLoad()
             << ", prob=" << aug.shedProbability()
             << endl;
    }

    cerr << "WOKE UP!" << endl;

    size_t sent = 0, recv = 0;
    for (auto& th : feederThreads)
    {
        th->shutdown();
        sent += th->sent;
        recv += th->recv;
    }
    aug.shutdown();

    BOOST_CHECK_EQUAL(aug_vec.size(),sent);
    BOOST_CHECK_EQUAL(sent,recv);
    for (const auto& str: aug_vec)
        BOOST_CHECK_EQUAL(str, aug_str);

    cerr << "sent: " << sent << endl
         << "recv: " << recv << endl;

    proxies->events->dump(cerr);
}
