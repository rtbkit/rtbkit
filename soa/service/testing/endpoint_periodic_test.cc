/* endpoint_periodic_test.cc
   Wolfgang Sourdeau, May 2013
   Copyright (c) 2013 Datacratic.  All rights reserved.

   Tests for the handle of periodic events in endpoints.cc
*/

#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include "soa/service/endpoint.h"
#include <sys/socket.h>
#include "jml/utils/testing/watchdog.h"
#include "jml/arch/timers.h"
#include "jml/arch/futex.h"
#include <mutex>
#include <condition_variable>


using namespace std;
using namespace ML;
using namespace Datacratic;



namespace {

struct MockEndpoint : public Datacratic::EndpointBase {
    MockEndpoint (const std::string & name)
        : EndpointBase(name)
    {
    }

    ~MockEndpoint()
    {
        shutdown();
    }

    virtual std::string hostname()
        const
    {
        return "mock-ep";
    }
    
    virtual int port()
        const
    {
        return -1;
    }

    virtual void closePeer()
    {}
};

}

BOOST_AUTO_TEST_CASE( test_periodic )
{
    MockEndpoint anEndpoint("myEndpoint");

    int invocations(0);
    uint64_t lastNumWakeUps(0);
    double cbSleep(0.0);
    int processing(0);
    auto timerCallback = [&] (uint64_t numWakeUps) {
        processing++;
        futex_wake(processing);
        if (cbSleep) {
            ML::sleep(cbSleep);
        }
        invocations++;
        lastNumWakeUps = numWakeUps;
        processing--;
        futex_wake(processing);
    };

    anEndpoint.init(10);
    anEndpoint.spinup(10, true);

    anEndpoint.addPeriodic("test", 1.0, timerCallback);

    ML::sleep(2);

    /* ensure the timer has been triggered */
    BOOST_CHECK_NE(invocations, 0);

    /* ensure that shutdown is properly handled */
    while (processing) {
        int oldValue = processing;
        futex_wait(processing, oldValue);
    }
    cbSleep = 5.0;
    while (!processing) {
        int oldValue = processing;
        futex_wait(processing, oldValue);
    }
    anEndpoint.shutdown();
}

BOOST_AUTO_TEST_CASE( test_add_remove_periodic )
{
    MockEndpoint anEndpoint("myEndpoint");

    std::mutex m;
    std::condition_variable cv;
    int invocations(0);
    auto timerCallback = [&] (uint64_t numWakeUps) {
        invocations++;
        cv.notify_one();
    };

    anEndpoint.init(10);
    anEndpoint.spinup(10, true);

    anEndpoint.addPeriodic("test", 1.0, timerCallback);
    {
        std::unique_lock<std::mutex> lock(m);
        auto status = cv.wait_for(lock, std::chrono::seconds(2));
        if (status == cv_status::timeout) BOOST_FAIL("should not timeout");
    }
    BOOST_CHECK_EQUAL(invocations, 1);

    anEndpoint.removePeriodic("test");
    {
        std::unique_lock<std::mutex> lock(m);
        auto status = cv.wait_for(lock, std::chrono::seconds(2));
        if (status != cv_status::timeout) BOOST_FAIL("should timeout");
    }
    BOOST_CHECK_EQUAL(invocations, 1);
}
