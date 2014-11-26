/* analytics_test.cc

    Analytics plugin test
*/


#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include "jml/arch/timers.h"

#include "rtbkit/plugins/analytics/analytics_endpoint.h"
#include "rtbkit/common/analytics_publisher.h"

using namespace std;
using namespace ML;
using namespace Datacratic;

void setUpEndpoint(shared_ptr<AnalyticsRestEndpoint> & analyticsEndpoint)
{
    auto proxies = std::make_shared<ServiceProxies> ();
    analyticsEndpoint = make_shared<AnalyticsRestEndpoint> (proxies, "analytics");
    analyticsEndpoint->init();
    analyticsEndpoint->bindTcp(40000);
    analyticsEndpoint->start();
}

void setUpClient(shared_ptr<AnalyticsPublisher> & analyticsClient)
{
    analyticsClient = make_shared<AnalyticsPublisher> ();
    analyticsClient->init("http://127.0.0.1:40000", 1);
    analyticsClient->start();
}

BOOST_AUTO_TEST_CASE( analytics_simple_message_test )
{
    shared_ptr<AnalyticsRestEndpoint> analyticsEndpoint;
    setUpEndpoint(analyticsEndpoint);

    shared_ptr<AnalyticsPublisher> analyticsClient;
    setUpClient(analyticsClient);

    // must sleep to let the heartbeat make a connection,
    // else it will not send message because the connection
    // is not established.
    ML::sleep(2.0);

    analyticsClient->publish("Test", "message", "channel is ignored");

    ML::sleep(0.5);

    analyticsEndpoint->enableChannel("Test");

    ML::sleep(0.5);

    analyticsClient->syncChannelFilters();

    ML::sleep(0.5);

    analyticsClient->publish("Test", "message", "channel is enabled");

    ML::sleep(0.5);

    analyticsEndpoint->enableAllChannels();

    ML::sleep(0.5);

    analyticsClient->syncChannelFilters();

    ML::sleep(0.5);

    // will not be sent because the Test2 is not yet in channel filter,
    // so it will be ignored.
    analyticsClient->publish("Test2", "all", "channels disbled");

    ML::sleep(0.5);

    analyticsEndpoint->disableAllChannels();

    ML::sleep(0.5);

    analyticsClient->syncChannelFilters();

    ML::sleep(0.5);

    analyticsClient->shutdown();
    analyticsEndpoint->shutdown();
}

BOOST_AUTO_TEST_CASE( analytics_filter_sync_test )
{
    shared_ptr<AnalyticsRestEndpoint> analyticsEndpoint;
    setUpEndpoint(analyticsEndpoint);

    shared_ptr<AnalyticsPublisher> analyticsClient;
    setUpClient(analyticsClient);

    analyticsEndpoint->enableChannel("Test");
    analyticsEndpoint->enableChannel("Test2");

    ML::sleep(1.0);

    analyticsClient->syncChannelFilters();

    ML::sleep(1.0);

    analyticsClient->shutdown();
    analyticsEndpoint->shutdown();

}
