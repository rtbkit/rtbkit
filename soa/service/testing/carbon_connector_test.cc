/* graphite_connector_test.cc
   Jeremy Barnes, 3 August 2011
   Copyright (c) 2011 Datacratic.  All rights reserved.

   Test for the carbon connector.
*/

#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include "soa/service/carbon_connector.h"
#include "jml/arch/atomic_ops.h"
#include "jml/arch/timers.h"
#include "soa/service/passive_endpoint.h"
#include <boost/make_shared.hpp>


using namespace std;
using namespace Datacratic;
using namespace ML;


BOOST_AUTO_TEST_CASE( test_counter_aggregator )
{
    // We record events to aggregate from multiple threads with simultaneous
    // occasional resets of the counter and we make sure that no events are
    // lost.

    cerr << "counter aggregator" << endl;

    CounterAggregator aggregator;

    uint64_t nthreads = 8, iter = 100000;
    boost::barrier barrier(nthreads);
    uint64_t total = 0;
    boost::thread_group tg;
    for (unsigned i = 0;  i < nthreads;  ++i) {
        auto doThread = [&] ()
            {
                barrier.wait();

                for (unsigned i = 0;  i < iter;  ++i) {
                    aggregator.record(1.0);

                    if (random() % 100 == 0) {
                        double val = aggregator.reset().first;
                        uint64_t val2 = val;

                        //cerr << "val2 = " << val2 << endl;

                        //BOOST_CHECK_EQUAL(val, val2);

                        atomic_add(total, val2);
                    }
                }
            };
        
        tg.create_thread(doThread);
    }

    tg.join_all();

    double val = aggregator.reset().first;
    uint64_t val2 = val;

    //cerr << "val2 = " << val2 << endl;

    BOOST_CHECK_EQUAL(val, val2);
    atomic_add(total, val2);

    BOOST_CHECK_EQUAL(total, iter * nthreads);
}

BOOST_AUTO_TEST_CASE( test_gauge_aggregator )
{
    // We record events to aggregate from multiple threads with simultaneous
    // occasional resets of the counter and we make sure that no events are
    // lost.

    cerr << "gauge aggregator" << endl;

    GaugeAggregator aggregator;

    uint64_t nthreads = 8, iter = 100000;
    boost::barrier barrier(nthreads);
    boost::thread_group tg;

    boost::mutex mutex;

    ML::distribution<float> allValues;

    for (unsigned i = 0;  i < nthreads;  ++i) {
        auto doThread = [&] ()
            {
                ML::distribution<float> threadValues;

                barrier.wait();

                for (unsigned i = 0;  i < iter;  ++i) {
                    aggregator.record(1.0 + (i % 2));

                    if (random() % 1000 == 0) {
                        ML::distribution<float> * values
                            = aggregator.reset().first;
                       
                        threadValues.insert(threadValues.end(),
                                            values->begin(), values->end());

                        delete values;
                    }
                }
                
                boost::lock_guard<boost::mutex> lock(mutex);
                allValues.insert(allValues.end(),
                                 threadValues.begin(), threadValues.end());
            };
        
        tg.create_thread(doThread);
    }

    tg.join_all();

    ML::distribution<float> * values
        = aggregator.reset().first;
    
    allValues.insert(allValues.end(),
                     values->begin(), values->end());
    std::sort(allValues.begin(), allValues.end());

    BOOST_CHECK_EQUAL(allValues.size(), iter * nthreads);
    BOOST_CHECK_EQUAL(allValues.mean(), 1.5);
    BOOST_CHECK_EQUAL(allValues.std(), 0.5);
}

BOOST_AUTO_TEST_CASE( test_multi_aggregator )
{
    std::vector<StatReading> readings;

    boost::mutex m;
    m.lock();

    auto recordReading = [&] (const std::vector<StatReading> & stats)
        {
            cerr << "got reading of " << stats.size() << " counters"
                 << endl;
            for (unsigned i = 0;  i < stats.size();  ++i)
                cerr << stats[i].name << " " << stats[i].value << endl;
            readings.insert(readings.end(), stats.begin(), stats.end());

            m.unlock();
        };

    MultiAggregator agg("hello", recordReading, 0.0);

    for (unsigned i = 0;  i <= 100;  ++i)
        agg.recordLevel("bonus", i);

    agg.dump();

    // agg.dump() will eventually call recordReading from another thread,
    // which will unlock m when it's finished, allowing this code to
    // proceed.

    m.lock();

    BOOST_REQUIRE_EQUAL(readings.size(), 8);
    BOOST_CHECK_EQUAL(readings[0].name, "bonus.mean");
    BOOST_CHECK_EQUAL(readings[0].value, 50.0);
}

struct FakeCarbon : public PassiveEndpointT<SocketTransport> {

    FakeCarbon()
        : PassiveEndpointT<SocketTransport>("Carbon"),
          numConnections(0), numDisconnections(0),
          numDataMessages(0), numErrorMessages(0)
    {
    }

    struct CarbonConnection: public PassiveConnectionHandler {

        CarbonConnection(FakeCarbon * owner)
            : owner(owner)
        {
        }

        FakeCarbon * owner;

        /** Function called out to when we got some data */
        virtual void handleData(const std::string & data)
        {
            cerr << "got data " << data << endl;
            ML::atomic_inc(owner->numDataMessages);
        }
    
        /** Function called out to when we got an error from the socket. */
        virtual void handleError(const std::string & message)
        {
            cerr << "got error " << message << endl;
            ML::atomic_inc(owner->numErrorMessages);
        }

        virtual void onGotTransport()
        {
            cerr << "on got transport" << endl;
            startReading();
        }

        virtual void handleDisconnect()
        {
            cerr << "got disconnection" << endl;
            ML::atomic_inc(owner->numDisconnections);
            closeWhenHandlerFinished();
        }
    };

    virtual std::shared_ptr<ConnectionHandler>
    makeNewHandler()
    {
        ML::atomic_inc(numConnections);
        cerr << "got new connection" << endl;
        return std::make_shared<CarbonConnection>(this);
    }

    int numConnections, numDisconnections, numDataMessages, numErrorMessages;
};

BOOST_AUTO_TEST_CASE( test_multiple_carbon_connectors )
{
    FakeCarbon carbon1, carbon2;
    int port1 = carbon1.init();
    int port2 = carbon2.init();

    string addr1 = ML::format("localhost:%d", port1);
    string addr2 = ML::format("localhost:%d", port2);

    cerr << "fake carbons are on " << port1 << " and " << port2 << endl;

    vector<string> addrs({ addr1, addr2 });

    CarbonConnector x(addrs, "test");

    for (unsigned i = 0;  i < 1000;  ++i) {
        x.recordHit("hit");
    }

    x.dump();

    ML::sleep(0.5);

    BOOST_CHECK_EQUAL(carbon1.numConnections, 1);
    BOOST_CHECK_EQUAL(carbon2.numConnections, 1);
    BOOST_CHECK_EQUAL(carbon1.numDisconnections, 0);
    BOOST_CHECK_EQUAL(carbon2.numDisconnections, 0);
    BOOST_CHECK_EQUAL(carbon1.numDataMessages, 1);
    BOOST_CHECK_EQUAL(carbon2.numDataMessages, 1);
    BOOST_CHECK_EQUAL(carbon1.numErrorMessages, 0);
    BOOST_CHECK_EQUAL(carbon2.numErrorMessages, 0);
    
    carbon1.shutdown();

    for (unsigned i = 0;  i < 1000;  ++i) {
        x.recordHit("hit");
    }
    
    x.dump();

    ML::sleep(1.0);

    carbon1.init(port1);

    ML::sleep(10.0);

    cerr << "done carbon connectors" << endl;
}
