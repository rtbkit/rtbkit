/* configuration_agent_test.cc
   Jeremy Barnes, 27 November 2012
   Copyright (c) 2012 Datacratic Inc.  All rights reserved.

   Test for the configuration agent.
*/

#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>

#include "rtbkit/core/router/router.h"
#include "rtbkit/core/agent_configuration/agent_configuration_service.h"
#include "rtbkit/core/agent_configuration/agent_configuration_listener.h"
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

    AgentConfigurationService config(proxies);
    
    config.init();
    config.bindTcp();
    config.start();

    proxies->config->dump(cerr);

    TestAgent agent(proxies, "bidding_agent");

    AgentConfigurationListener listener(proxies->zmqContext);

    int numConfigurations = 0;
    std::shared_ptr<const AgentConfig> currentConfig;
    std::string currentName;

    listener.onConfigChange = [&] (std::string agent,
                                   std::shared_ptr<const AgentConfig> config)
        {
            currentName = agent;
            //ExcAssertEqual(agent, "bidding_agent");
            currentConfig = config;
            if (config) {
                cerr << "got new configuration for agent " << agent << endl;
                ++numConfigurations;
                ML::futex_wake(numConfigurations);
            }
            else {
            	cerr << "agent " << agent << " removed\n";
                --numConfigurations;
                ML::futex_wake(numConfigurations);
            }

        };

    listener.init(proxies->config);
    listener.start();

    agent.agentName = "bidding_agent";
    agent.init();
    agent.start();
    agent.configure();
    
    while (numConfigurations == 0)
        futex_wait(numConfigurations, 0);
    
    BOOST_CHECK_EQUAL(numConfigurations, 1);
    BOOST_CHECK_EQUAL(currentConfig->toJson(), agent.config.toJson());

    BOOST_CHECK_EQUAL(listener.getAgentEntry("bidding_agent").config,
                      currentConfig);
    BOOST_CHECK_EQUAL(currentName, "bidding_agent");

    // Check that a second configuration with the same parameters doesn't cause a new
    // message to be broadcast
    agent.configure();

    ML::sleep(0.1);

    BOOST_CHECK_EQUAL(numConfigurations, 1);

    // Now modify the configuration
    agent.config.maxInFlight = 10000;

    agent.configure();

    while (numConfigurations == 1)
        futex_wait(numConfigurations, 1);

    BOOST_CHECK_EQUAL(numConfigurations, 2);
    BOOST_CHECK_EQUAL(currentConfig->toJson(), agent.config.toJson());

    BOOST_CHECK_EQUAL(listener.getAgentEntry("bidding_agent").config,
                      currentConfig);
    BOOST_CHECK_EQUAL(currentName, "bidding_agent");

    TestAgent agent2(proxies, "bidding_agent2");
    agent2.agentName = "bidding_agent2";
    agent2.init();
    agent2.start();
    agent2.configure();

    while (numConfigurations == 2)
        futex_wait(numConfigurations, 2);
    
    BOOST_CHECK_EQUAL(numConfigurations, 3);
    BOOST_CHECK_EQUAL(currentConfig->toJson(), agent2.config.toJson());

    BOOST_CHECK_EQUAL(listener.getAgentEntry("bidding_agent2").config,
                      currentConfig);
    BOOST_CHECK_EQUAL(currentName, "bidding_agent2");

    // now we shutdown agent2 and verify that we've received a void config
    // for it.
    agent2.shutdown();

    while (numConfigurations == 3)
        futex_wait(numConfigurations, 3);
    BOOST_CHECK_EQUAL(numConfigurations, 2);


    cerr << "tests done" << endl;
}

