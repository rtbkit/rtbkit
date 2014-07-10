/* agent_context_switch_test.cc
   Jeremy Barnes, 15 June 2014
   Copyright (c) 2014 Datacratic Inc.  All rights reserved.

   Test for the context switches for the bidding agent.
*/

#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>

#include "rtbkit/core/router/router.h"
#include "rtbkit/core/agent_configuration/agent_configuration_service.h"
#include "rtbkit/core/agent_configuration/agent_configuration_listener.h"
#include "soa/service/process_stats.h"
#include "jml/utils/pair_utils.h"
#include "jml/arch/timers.h"
#include "jml/arch/futex.h"
#include "test_agent.h"

using namespace std;
using namespace ML;
using namespace Datacratic;
using namespace RTBKIT;


BOOST_AUTO_TEST_CASE( test_agent_configuration )
{
    std::shared_ptr<ServiceProxies> proxies(new ServiceProxies());

    TestAgent agent(proxies, "bidding_agent");
    agent.init();
    agent.start();

    ProcessStats stats0(false);

    ML::sleep(1.0);

    ProcessStats stats(false);

    agent.shutdown();

    cerr << "did " << stats.totalContextSwitches() - stats0.totalContextSwitches()
         << " context switches per second" << endl;

    cerr << "used " << stats.totalTime() - stats0.totalTime() << "s CPU time"
         << endl;
}

