/* multiple_service_test.cc
   Jeremy Barnes, 10 December 2012
   Copyright (c) 2012 Datacratic Inc.  All rights reserved.

*/

#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include <boost/make_shared.hpp>
#include "soa/service/named_endpoint.h"
#include "soa/service/message_loop.h"
#include "soa/service/zmq_endpoint.h"
#include "soa/service/testing/zookeeper_temporary_server.h"
#include "jml/utils/guard.h"
#include "jml/arch/exception_handler.h"
#include "jml/utils/testing/watchdog.h"
#include "jml/utils/testing/fd_exhauster.h"
#include "jml/utils/vector_utils.h"
#include "jml/arch/timers.h"
#include <thread>
#include "soa/service/zmq_utils.h"
#include "soa/service/zookeeper.h"

using namespace std;
using namespace ML;
using namespace Datacratic;


/*****************************************************************************/
/* ECHO SERVICE                                                              */
/*****************************************************************************/

/** Simple test service that listens on zeromq and simply echos everything
    that it gets back.
*/

struct EchoService : public ServiceBase {

    EchoService(std::shared_ptr<ServiceProxies> proxies,
                const std::string & name)
        : ServiceBase(name, proxies),
          toClients(getZmqContext())
    {
        proxies->config->removePath(serviceName());
        registerServiceProvider(serviceName(), { "echo" });

        auto handler = [=] (vector<string> message)
            {
                //cerr << "got message " << message << endl;
                ExcAssertEqual(message.size(), 3);
                ExcAssertEqual(message[1], "ECHO");
                message[1] = "REPLY";
                return message;
            };

        toClients.clientMessageHandler = handler;
    }

    ~EchoService()
    {
        shutdown();
    }

    void init()
    {
        toClients.init(getServices()->config, serviceName() + "/echo");
    }

    void start()
    {
        toClients.start();
    }

    void shutdown()
    {
        toClients.shutdown();
    }

    std::string bindTcp()
    {
        return toClients.bindTcp();
    }

    ZmqNamedClientBus toClients;
};
BOOST_AUTO_TEST_CASE( test_service_zk_disconnect )
{
    ZooKeeper::TemporaryServer zookeeper;
    std::cerr <<"starting zookeeper..." << std::endl;
    zookeeper.start();
    ML::sleep(2);

    cerr << "Starting multiple service zk disconnect " << endl;
    auto proxies = std::make_shared<ServiceProxies>();
    proxies->useZookeeper(ML::format("localhost:%d", zookeeper.getPort()));
    //proxies->useZookeeper(ML::format("ag3.datacratic.com:2181"));

    ZmqMultipleNamedClientBusProxy connection(proxies->zmqContext);
    connection.init(proxies->config, "client1");

    connection.connectHandler = [&] (const std::string & svc)
        {
            cerr << "connected to " << svc << endl;
        };

    connection.disconnectHandler = [&] (const std::string  & svc)
        {
            cerr << "disconnected from " << svc << endl;
        };

    connection.start();

    BOOST_CHECK_EQUAL(connection.connectionCount(), 0);

    connection.connectAllServiceProviders("echo", "echo");

    BOOST_CHECK_EQUAL(connection.connectionCount(), 0);

    std::vector<unique_ptr<EchoService> > services;

    auto startService = [&] ()
        {
            services.emplace_back(new EchoService(proxies, "echo" + to_string(services.size())));
            EchoService & service = *services.back();
            service.init();
            auto addr = service.bindTcp();
            cerr << "echo service is listening on " << addr << endl;
            service.start();
        };

    startService();

//    proxies->config->dump(cerr);
//    std::cerr <<"going to sleep for 5 seconds " << std::endl;
    ML::sleep(5);

    std::cerr << "About to suspend zookeeper..." ;
    // Make sure that the latest callback id is correct
    BOOST_CHECK_EQUAL(ZookeeperCallbackManager::instance().getId(),6);
    zookeeper.suspend();
    std::cerr << "zookeeper suspended " << std::endl;
    ML::sleep(10);
    std::cerr <<"resuming zookeeper " << std::endl;
    zookeeper.resume() ;
    ML::sleep(10);
    // When we resume zookeeper we will reconnect and a new callback
    // should be installed if the watch is reinstalled correctly
    // @todo better mechanism for checking that watches are handled 
    // correctly.
    BOOST_CHECK_EQUAL(ZookeeperCallbackManager::instance().getId(),8);
    zookeeper.suspend();
    std::cerr << "zookeeper suspended again" << std::endl;
    ML::sleep(10);
    std::cerr <<"resuming zookeeper again " << std::endl;
    zookeeper.resume() ;
    cerr << "going to sleep for 30 seconds.." << endl;
    ML::sleep(10);
    BOOST_CHECK_EQUAL(ZookeeperCallbackManager::instance().getId(), 10);
    cerr << "shutting down" << endl;

    connection.shutdown();

    for (unsigned i = 0;  i < services.size();  ++i)
        services[i]->shutdown();
}

#if 1
BOOST_AUTO_TEST_CASE( test_early_connection )
{
    /** Test that we can do a "connect", then start the service, and
        have the connection come up once the service appears.
    */
    cerr << "Testing early connection..." << endl;

    ZooKeeper::TemporaryServer zookeeper;
    zookeeper.start();

    auto proxies = std::make_shared<ServiceProxies>();
    proxies->useZookeeper(ML::format("localhost:%d", zookeeper.getPort()));

    ZmqNamedClientBusProxy connection(proxies->zmqContext);
    connection.init(proxies->config, "client1");
    connection.connectHandler = [&] (const std::string & svc) {
        cerr << "connected to " << svc << endl;
    };

    connection.disconnectHandler = [&] (const std::string  & svc) {
        cerr << "disconnected from " << svc << endl;
    };

    connection.start();
    connection.connectToServiceClass("echo", "echo");

    //for(int i = 0; i != 2; ++i)
    {
        while(connection.isConnected()) {
            ML::sleep(0.1);
        }

        BOOST_CHECK_EQUAL(connection.isConnected(), false);

        proxies->config->removePath("");

        EchoService service(proxies, "echo");
        service.init();
        auto addr = service.bindTcp();
        cerr << "echo service is listening on " << addr << endl;
        service.start();

        //proxies->config->dump(cerr);

        while(!connection.isConnected()) {
            ML::sleep(0.1);
        }

        cerr << "Checking that we are connected " << endl;
        BOOST_CHECK_EQUAL(connection.isConnected(), true);
    }

    std::cerr << "done." << std::endl;
}

BOOST_AUTO_TEST_CASE( test_multiple_services )
{
    ZooKeeper::TemporaryServer zookeeper;
    zookeeper.start();

    auto proxies = std::make_shared<ServiceProxies>();
    proxies->useZookeeper(ML::format("localhost:%d", zookeeper.getPort()));

    cerr << "Starting multiple services test " << endl;

    ZmqMultipleNamedClientBusProxy connection(proxies->zmqContext);
    connection.init(proxies->config, "client1");

    connection.connectHandler = [&] (const std::string & svc)
        {
            cerr << "connected to " << svc << endl;
        };

    connection.disconnectHandler = [&] (const std::string  & svc)
        {
            cerr << "disconnected from " << svc << endl;
        };

    connection.start();

    BOOST_CHECK_EQUAL(connection.connectionCount(), 0);

    connection.connectAllServiceProviders("echo", "echo");

    BOOST_CHECK_EQUAL(connection.connectionCount(), 0);

    std::vector<unique_ptr<EchoService> > services;

    auto startService = [&] ()
        {
            services.emplace_back(new EchoService(proxies, "echo" + to_string(services.size())));
            EchoService & service = *services.back();
            service.init();
            auto addr = service.bindTcp();
            cerr << "echo service is listening on " << addr << endl;
            service.start();
        };

    startService();

    proxies->config->dump(cerr);

    ML::sleep(0.1);

    BOOST_CHECK_EQUAL(connection.connectionCount(), 1);

    cerr << "shutting down" << endl;

    connection.shutdown();

    for (unsigned i = 0;  i < services.size();  ++i)
        services[i]->shutdown();

}
#endif
