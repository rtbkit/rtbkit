/* rtb_router_leak_test.cc
   
   Check that the router et al don't leak.
*/

#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include "jml/arch/format.h"
#include "rtb/simulation/simulation.h"
#include "rtbkit/core/router/router.h"
#include <boost/thread/thread.hpp>
#include <ace/Signal.h>


using namespace std;
using namespace ML;
using namespace Datacratic;
using namespace RTBKIT;

BOOST_AUTO_TEST_CASE( test_router_simulation_dont_leak )
{
    int numAuctions = 200;
    int queriesPerSecond = 100;
    int maxTimeMs = 100;

    Router router;
    router.initEndpoints();
    string host = "localhost";
    int bidPort = router.bidPort();
    int backchannelPort = router.backchannelPort();

    Simulation sim(queriesPerSecond, maxTimeMs);

    EventLog log(some_sqlite_file);
    EventFilter query = log.query().type(some_request_type).limit(numAuctions);


    sim.init(host, bidPort, backchannelPort);

    sim.run(query);

    //client->sleepUntilIdle();

    sim.shutdown();
    router.shutdown();

    sim.stats();

    BOOST_CHECK_EQUAL(sim.counters["auctions_started"], numAuctions);
    BOOST_CHECK_EQUAL(sim.counters["auctions_finished"], numAuctions);
    BOOST_CHECK_GT(sim.counters["errors"], 0);
}
