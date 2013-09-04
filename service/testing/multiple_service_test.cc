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
#include "soa/service/zookeeper_configuration_service.h"
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

class ServiceDiscoveryScenario {
public:
    friend class ServiceDiscoveryScenarioTest;

    ServiceDiscoveryScenario(const std::string &name) :
        name { name }
    { }

    std::shared_ptr<ServiceProxies>
    createProxies(const std::string &host, 
                  const std::string &name = "DEFAULT") 
    {
        if (getFromMap(proxiesMap, name))
            throw ML::Exception(ML::format("Proxies with name '%s' already exist",
                                name.c_str()));

        auto proxies = std::make_shared<ServiceProxies>();
        proxies->useZookeeper(host);

        auto proxiesEntry = std::make_shared<ProxiesEntry>(proxies, host);
        proxiesMap.insert(std::make_pair(name, proxiesEntry));

        return proxies;
    }

    int
    startTemporaryServer() 
    {
        zooServer.reset(new ZooKeeper::TemporaryServer);
        zooServer->start();

        return zooServer->getPort();
    }

    void
    reset() {
        zooServer.reset(nullptr);
        proxiesMap.clear();
        clientsMap.clear();
        servicesMap.clear();
    }

    std::shared_ptr<ZmqMultipleNamedClientBusProxy> 
    createConnection(const std::string &name,
                     const std::string &proxiesName = "DEFAULT")
    {
        auto proxiesEntry = getFromMap(proxiesMap, proxiesName);
        if (!proxiesEntry)
            throw ML::Exception(ML::format("Proxies with name '%s' does not "
                                           "exist", proxiesName.c_str()));

        if (getFromMap(clientsMap, name))
            throw ML::Exception("createConnection: client with the same name already "
                                 "exists");

        auto proxies = proxiesEntry->proxies;
        ExcAssert(proxies);
        auto conn = std::make_shared<ZmqMultipleNamedClientBusProxy>(proxies->zmqContext);
        conn->init(proxies->config);
        clientsMap.insert(std::make_pair(name, conn));

        return conn;

    }

    std::shared_ptr<ZmqMultipleNamedClientBusProxy>
    createConnectionAndStart(const std::string &name,
                             const std::string &proxiesName = "DEFAULT") 
    {
        auto conn = createConnection(name, proxiesName);
        conn->start();

        return conn;
    }    

    std::shared_ptr<ZmqMultipleNamedClientBusProxy> 
    connectServiceProviders(const std::string &clientName,
                            const std::string &serviceClass,
                            const std::string &endpointName)
    {
        auto connection = getFromMap(clientsMap, clientName);
        if (!connection)
            throw ML::Exception(ML::format("connection with name '%s' does "
                                           "not exist", clientName.c_str()));

        connection->connectAllServiceProviders(serviceClass, endpointName);
        return connection;
    }

    std::shared_ptr<EchoService>
    createService(const std::string &name, 
                  const std::string &proxiesName = "DEFAULT")
    {
        auto proxiesEntry = getFromMap(proxiesMap, proxiesName);
        if (!proxiesEntry)
            throw ML::Exception(ML::format("Proxies with name '%s' does not "
                                           "exist", proxiesName.c_str()));
        
        auto proxies = proxiesEntry->proxies;
        ExcAssert(proxies);
        auto service = std::make_shared<EchoService>(proxies, name);
        service->init();

        servicesMap.insert(std::make_pair(name, service));

        return service;
    }

    std::shared_ptr<EchoService>
    createServiceAndStart(const std::string &name,
                          const std::string &proxiesName = "DEFAULT")
    {
        auto service = createService(name, proxiesName);
            
        service->bindTcp();
        service->start();

        return service;
    }

    void
    expireSession(const std::string &proxiesName)
    {
        auto proxiesEntry = getFromMap(proxiesMap, proxiesName);
        if (!proxiesEntry) 
            throw ML::Exception(ML::format("Proxies with name '%s' does not exist",
                                           proxiesName.c_str()));

        if (!proxiesEntry->hasCredentials)
            throw ML::Exception("No credentials for given proxy");

        auto proxies = proxiesEntry->proxies;
        ExcAssert(proxies);
        
        std::unique_ptr<ZookeeperConnection> connection(new ZookeeperConnection);
        connection->connectWithCredentials(proxiesEntry->host, proxiesEntry->sessionId,
                                           proxiesEntry->password);
        connection->close();
    }
        

private:
    std::string name;
    std::unique_ptr<ZooKeeper::TemporaryServer> zooServer;

    struct ProxiesEntry {
        ProxiesEntry(const std::shared_ptr<ServiceProxies> &proxies,
                     const std::string &host) :
            proxies { proxies },
            host { host },
            hasCredentials { false }
        { }

        ProxiesEntry(const std::shared_ptr<ServiceProxies> &proxies,
                     const std::string &host,
                     int64_t sessionId,
                     const std::string &password) :
            proxies { proxies },
            host { host },
            hasCredentials { true },
            sessionId { sessionId },
            password { password }
        { }

        std::shared_ptr<ServiceProxies> proxies;
        std::string host;
        bool hasCredentials;

        int64_t sessionId;
        std::string password;

    };

    typedef std::map<std::string, std::shared_ptr<ZmqMultipleNamedClientBusProxy>>
    ClientsMap;
    typedef std::map<std::string, std::shared_ptr<EchoService>>
    ServicesMap;
    typedef std::map<std::string, std::shared_ptr<ProxiesEntry>>
    ProxiesMap;

    ClientsMap clientsMap;
    ServicesMap servicesMap;
    ProxiesMap proxiesMap;

    template<typename Map>
    typename Map::mapped_type 
    getFromMap(const Map &map, const std::string &name)
    {
        auto it = map.find(name);
        if (it == end(map)) 
            return nullptr;

        return it->second;
    } 

};

class ServiceDiscoveryScenarioTest {
public:
    ServiceDiscoveryScenarioTest(ServiceDiscoveryScenario &scenario) :
        scenario(scenario)
    { }

    void assertConnectionCount(const std::string &connectionName,
                               int count)
    {
        auto connection = scenario.getFromMap(scenario.clientsMap,
                                              connectionName);
        ExcAssert(connection);
        BOOST_CHECK_EQUAL(connection->connectionCount(), count);
    }

private:
    ServiceDiscoveryScenario &scenario;
};

#if 0
BOOST_AUTO_TEST_CASE( test_service_zk_disconnect )
{
    ZooKeeper::TemporaryServer zookeeper;
    std::cerr <<"starting zookeeper..." << std::endl;
    zookeeper.start();
    ML::sleep(2);

    cerr << "Starting multiple service zk disconnect " << endl;
    auto proxies = std::make_shared<ServiceProxies>();
    proxies->useZookeeper(ML::format("localhost:%d", zookeeper.getPort()));

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
#endif

std::string formatHost(const std::string &host, int port) {
    return ML::format("%s:%d", host.c_str(), port);
}

BOOST_AUTO_TEST_CASE( zk_simple_watch_test )
{
    ZooKeeper::TemporaryServer zkServer;
    zkServer.start();

    const std::string host { formatHost("localhost", zkServer.getPort()) };

    auto proxies = std::make_shared<ServiceProxies>();
    proxies->useZookeeper(host);

    ZmqNamedClientBusProxy connection(proxies->zmqContext);
    connection.init(proxies->config, "client");
    connection.start();
    connection.connectToServiceClass("echo", "echo");

    std::unique_ptr<EchoService> service { new EchoService(proxies, "echo0") };
    service->init();
    auto addr = service->bindTcp();
    std::cout << "Service listening on " << addr << std::endl; 

    service->start();

    while (!connection.isConnected())
        ML::sleep(0.1);

    ML::sleep(5);

    connection.shutdown();
    service->shutdown();
}


#if 0
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

BOOST_AUTO_TEST_CASE( test_multiple_services )
{
    ServiceDiscoveryScenario scenario("test_multiple_services");
    ServiceDiscoveryScenarioTest test(scenario);

    int port = scenario.startTemporaryServer();
    scenario.createProxies(formatHost("localhost", port));

    auto connection = scenario.createConnectionAndStart("client");
    connection->connectHandler = [](const std::string &) {
        std::cout << "connected" << std::endl;
    };

    test.assertConnectionCount("client", 0);

    scenario.connectServiceProviders("client", "echo", "echo");
    test.assertConnectionCount("client", 0);
    scenario.createServiceAndStart("echo0");

    ML::sleep(0.5);

    test.assertConnectionCount("client", 1);
}

BOOST_AUTO_TEST_CASE( test_simple_disconnect )
{
    ServiceDiscoveryScenario scenario("test_simple_disconnect");
    ServiceDiscoveryScenarioTest test(scenario);

    int port = scenario.startTemporaryServer();
    scenario.createProxies(formatHost("localhost", port), "connectionProxy");
    scenario.createProxies(formatHost("localhost", port), "endpointProxy");

    auto connection = scenario.createConnectionAndStart("client", "connectionProxy");
    connection->connectHandler = [](const std::string &) {
        std::cout << "connected" << std::endl;
    };

    scenario.connectServiceProviders("client", "echo", "echo");
    scenario.createServiceAndStart("echo0", "endpointProxy");

    ML::sleep(0.5);

    test.assertConnectionCount("client", 1);
}
