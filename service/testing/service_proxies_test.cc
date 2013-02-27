#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <string>

#include <boost/test/unit_test.hpp>
#include <boost/make_shared.hpp>
#include <boost/thread/thread.hpp>

#include <jml/utils/set_utils.h>

#include "soa/service/service_base.h"
#include "soa/service/rest_service_endpoint.h"
#include "soa/service/testing/zookeeper_temporary_server.h"

using namespace std;
using namespace ML;

using namespace Datacratic;

struct MockRestService : public ServiceBase,
                         public RestServiceEndpoint
{
    MockRestService(std::shared_ptr<ServiceProxies> proxies,
                    const string & serviceName)
        : ServiceBase(serviceName, proxies),
          RestServiceEndpoint(proxies->zmqContext)
    {}

    ~MockRestService()
    {
        unregisterServiceProvider(serviceName(), classes_);
    }

    void init(const vector<string> & classes)
    {
        classes_ = classes;
        registerServiceProvider(serviceName(), classes);
        getServices()->config->removePath(serviceName());
        RestServiceEndpoint::init(getServices()->config, serviceName());
    }

    vector<string> classes_;
};

BOOST_AUTO_TEST_CASE( test_service_proxies_getServiceClassInstances )
{
    ZooKeeper::TemporaryServer server;
    server.start();

    auto proxies = std::make_shared<ServiceProxies>();
    proxies->useZookeeper(ML::format("localhost:%d", server.getPort()));

    set<string> expectedUris;

    MockRestService testService1(proxies, "testServiceName1");
    testService1.init({"testServiceClass", "otherServiceClass"});
    testService1.bindTcp();
    vector<string> uris(testService1.zmqEndpoint.getPublishedUris());
    expectedUris.insert(uris.begin(), uris.end());
    uris = testService1.httpEndpoint.getPublishedUris();
    expectedUris.insert(uris.begin(), uris.end());

    MockRestService testService2(proxies, "testServiceName2");
    testService2.init({"testServiceClass"});
    testService2.bindTcp();
    uris = testService2.zmqEndpoint.getPublishedUris();
    expectedUris.insert(uris.begin(), uris.end());
    uris = testService2.httpEndpoint.getPublishedUris();
    expectedUris.insert(uris.begin(), uris.end());
    
    set<string> instanceUris;
    uris = proxies->getServiceClassInstances("testServiceClass", "http");
    instanceUris.insert(uris.begin(), uris.end());
    uris = proxies->getServiceClassInstances("testServiceClass", "zeromq");
    instanceUris.insert(uris.begin(), uris.end());

    BOOST_CHECK_EQUAL(instanceUris, expectedUris);
}
