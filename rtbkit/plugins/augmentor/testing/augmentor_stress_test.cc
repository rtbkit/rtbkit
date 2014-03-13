/** augmentor_stress_test.cc                                 -*- C++ -*-
    Rémi Attab, 10 May 2013
    Copyright (c) 2013 Datacratic.  All rights reserved.

    Stress test to worm out the bottleneck in augmentor base.

*/

#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include "rtbkit/plugins/augmentor/augmentor_base.h"
#include "jml/db/persistent.h"

#include <boost/test/unit_test.hpp>
#include <thread>
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
            recordHit("recv");
            recv++;
        };

        toAug.onConnection = [=] (const std::string & client) {
            cerr << "augmentor " << client << " has connected" << endl;
        };
        toAug.onDisconnection = [=] (const std::string & client) {
            cerr << "augmentor " << client << " has disconnected" << endl;
        };

        addSource("MockAugLoop::toAug", toAug);


        {
            set<string> agents { "bob-the-agent", "thingy" };
            std::ostringstream agentStr;
            ML::DB::Store_Writer writer(agentStr);
            writer.save(agents);
            this->agents = agentStr.str();
        }

        addPeriodic("MockAugLoop::send", 0.0001, [=] (uint64_t) {
                    toAug.sendMessage(
                            "test-aug", "AUGMENT", "1.0", "test-aug",
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

BOOST_AUTO_TEST_CASE( stressTest )
{
    enum {
        FeederThreads = 6,
        TestLength = 100
    };

    auto proxies = make_shared<ServiceProxies>();

    atomic<size_t> processed(0);

    cerr << "init feeders\n";

    vector< std::shared_ptr<MockAugmentationLoop> > feederThreads;
    for (size_t i = 0; i < FeederThreads; ++i) {
        feederThreads.emplace_back(new MockAugmentationLoop(proxies));
        feederThreads.back()->start();
    }

    cerr << "init aug\n";

    SyncAugmentor aug("test-aug", "test-aug", proxies);
    aug.init();
    aug.doRequest = [&] (const AugmentationRequest& req) {
        processed++;
        return AugmentationList();
    };
    aug.start();

    this_thread::sleep_for(chrono::milliseconds(100));

    cerr << "sleeping..." << endl;

    for (size_t i = 0; i < TestLength; ++i) {
        this_thread::sleep_for(chrono::seconds(1));
        cerr << "[ " << i << " / " << TestLength << " ]: "
            << "load=" << aug.sampleLoad()
            << ", prob=" << aug.shedProbability()
            << endl;
    }

    cerr << "WOKE UP!" << endl;

    size_t sent = 0, recv = 0;
    for (auto& th : feederThreads) {
        th->shutdown();
        sent += th->sent;
        recv += th->recv;
    }
    aug.shutdown();

    cerr << "sent: " << sent << endl
        << "proc: " << processed << endl
        << "recv: " << recv << endl;

    proxies->events->dump(cerr);
}
