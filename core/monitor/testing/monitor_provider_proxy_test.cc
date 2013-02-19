/* monitor_provider_proxy_test.cc
   Wolfgang Sourdeau, January 2013
   Copyright (c) 2013 Datacratic.  All rights reserved.
   
   Unit tests for the MonitorProviderProxy class
*/

#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include "soa/types/date.h"

#include "rtbkit/core/monitor/monitor_provider_proxy.h"

using namespace Datacratic;
using namespace RTBKIT;

struct MockMonitorProviderSubscriber : public MonitorProvidersSubscriber
{
    MonitorProviderResponses lastResponses;

    void onProvidersStatusLoaded(const MonitorProviderResponses & responses)
    {
        lastResponses = responses;
    }
};

BOOST_AUTO_TEST_CASE( test_monitor_provider_proxy_onResponseReceived )
{
    auto proxies = std::make_shared<ServiceProxies>();

    MockMonitorProviderSubscriber subscriber;

    MonitorProviderProxy proxy(proxies->zmqContext, subscriber);

    proxy.pendingRequests = 2;

    /* first request response:
       pendingRequests -> 1
       proxy.responses.size() == 1
       subscriber.lastResponses.size() = 0 */
    proxy.onResponseReceived("service1", nullptr, 123, "response1");
    BOOST_CHECK_EQUAL(proxy.pendingRequests, 1);
    BOOST_CHECK_EQUAL(proxy.responses.size(), 1);
    BOOST_CHECK_EQUAL(subscriber.lastResponses.size(), 0);
    const MonitorProviderResponse *response = &proxy.responses[0];
    BOOST_CHECK_EQUAL(response->serviceName, "service1");
    BOOST_CHECK_EQUAL(response->code, 123);
    BOOST_CHECK_EQUAL(response->body, "response1");

    /* second request response:
       pendingRequests -> 0
       proxy.responses.size() == 2
       proxy.responses transferred in order to subscriber.lastResponses
    */
    proxy.onResponseReceived("service2", nullptr, 456, "response2");
    BOOST_CHECK_EQUAL(proxy.pendingRequests, 0);
    BOOST_CHECK_EQUAL(proxy.responses.size(), 2);
    BOOST_CHECK_EQUAL(subscriber.lastResponses.size(), 2);
    response = &subscriber.lastResponses[0];
    BOOST_CHECK_EQUAL(response->serviceName, "service1");
    BOOST_CHECK_EQUAL(response->code, 123);
    BOOST_CHECK_EQUAL(response->body, "response1");
    response = &subscriber.lastResponses[1];
    BOOST_CHECK_EQUAL(response->serviceName, "service2");
    BOOST_CHECK_EQUAL(response->code, 456);
    BOOST_CHECK_EQUAL(response->body, "response2");
}
