#ifndef SERVICE_DISCOVERY_SCENARIO_H
#define SERVICE_DISCOVERY_SCENARIO_H

#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <map>
#include <string>
#include <memory>
#include <utility>
#include <boost/test/unit_test.hpp>
#include "zookeeper_temporary_server.h"
#include "jml/arch/exception.h"
#include "jml/arch/format.h"
#include "soa/service/zmq_endpoint.h"
#include "soa/service/service_base.h"
#include "echo_service.h"
#include <sys/stat.h> 
#include <fcntl.h>
#include <cstdint>

namespace Datacratic {

struct _zhandle {
    int fd;
};

inline std::string formatHost(const std::string &host, int port) {
    return ML::format("%s:%d", host.c_str(), port);
}

class ServiceDiscoveryScenario {
public:
    friend class ServiceDiscoveryScenarioTest;

    ServiceDiscoveryScenario(const std::string &name) :
        name { name },
        serverStatus { Down }
    { }

    ~ServiceDiscoveryScenario()
    {
       // reset();
    }

    std::shared_ptr<ServiceProxies>
    createProxies(const std::string &host, 
                  const std::string &name = "DEFAULT") 
    {
        if (getFromMap(proxiesMap, name))
            throw ML::Exception(ML::format("Proxies with name '%s' already exist",
                                name.c_str()));

        auto proxies = std::make_shared<ServiceProxies>();
        proxies->useZookeeper(host);

        proxiesMap.insert(std::make_pair(name, proxies));

        return proxies;
    }

    int
    startTemporaryServer() 
    {
        zooServer.reset(new ZooKeeper::TemporaryServer);
        zooServer->start();

        serverStatus = Running;
        return zooServer->getPort();
    }

    void
    suspendServer()
    {
        if (!zooServer)
            throw ML::Exception("NULL server can not be suspended");

        if (serverStatus != Running)
            throw ML::Exception("Server must be running to be suspended");

        zooServer->suspend();
        serverStatus = Suspended;
    }

    void
    resumeServer()
    {
        if (!zooServer)
            throw ML::Exception("NULL server can not be resumed");

        if (serverStatus != Suspended)
            throw ML::Exception("Server is not suspended");

        zooServer->resume();
        serverStatus = Running;
    }

    void
    reset()
    {
        for (auto &client: clientsMap)
            client.second->shutdown();

        for (auto &service: servicesMap)
            service.second->shutdown();

        proxiesMap.clear();
        clientsMap.clear();
        servicesMap.clear();

        if (zooServer)
            zooServer->shutdown();
            zooServer.reset(nullptr);

    }

    std::shared_ptr<ZmqMultipleNamedClientBusProxy> 
    createConnection(const std::string &name,
                     const std::string &proxiesName = "DEFAULT")
    {
        auto proxies = getFromMap(proxiesMap, proxiesName);
        if (!proxies)
            throw ML::Exception(ML::format("Proxies with name '%s' does not "
                                           "exist", proxiesName.c_str()));

        if (getFromMap(clientsMap, name))
            throw ML::Exception("createConnection: client with the same name already "
                                 "exists");

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

    void
    waitForClientConnected(const std::string &clientName,
                           double sleepTime = 0.1)
    {
        auto client = getFromMap(clientsMap, clientName);
        if (!client)
            throw ML::Exception(ML::format("connection with hame '%s' does "
                                           "not exist", clientName.c_str()));

        while (!client->connected)
            ML::sleep(sleepTime);
    }

    std::shared_ptr<EchoService>
    createService(const std::string &name, 
                  const std::string &proxiesName = "DEFAULT")
    {
        auto proxies = getFromMap(proxiesMap, proxiesName);
        if (!proxies)
            throw ML::Exception(ML::format("Proxies with name '%s' does not "
                                           "exist", proxiesName.c_str()));
        
        auto service = std::make_shared<EchoService>(proxies, name);
        service->init();
        service->bindTcp();

        servicesMap.insert(std::make_pair(name, service));

        return service;
    }

    std::shared_ptr<EchoService>
    createServiceAndStart(const std::string &name,
                          const std::string &proxiesName = "DEFAULT")
    {
        auto service = createService(name, proxiesName);
        service->start();

        return service;
    }

    void
    expireSession(const std::string &proxiesName = "DEFAULT")
    {
        auto proxies = getFromMap(proxiesMap, proxiesName);
        if (!proxies) 
            throw ML::Exception(ML::format("Proxies with name '%s' does not exist",
                                           proxiesName.c_str()));

        auto config = proxies->configAs<ZookeeperConfigurationService>();
// Closing the fd does not seem to trigger a session expired
#if 0
        int oldFd = config->zoo->handle->fd;
        int newFd = ::open("/dev/null", O_RDWR);
        config->zoo->handle->fd = newFd;
        std::cerr << ::close(oldFd) << std::endl;
#else
        auto credentials = config->zoo->sessionCredentials();

        std::unique_ptr<ZookeeperConnection> connection(new ZookeeperConnection);

        connection->connectWithCredentials(config->zoo->host,
                                           credentials.first, credentials.second);
        connection->close();
#endif
    }

    void
    reconnectSession(const std::string &proxiesName = "DEFAULT")
    {
        auto proxies = getFromMap(proxiesMap, proxiesName);
        if (!proxies) 
            throw ML::Exception(ML::format("Proxies with name '%s' does not exist",
                                           proxiesName.c_str()));

        auto config = proxies->configAs<ZookeeperConfigurationService>();
        config->zoo->reconnect();
    }



private:
    std::string name;

    enum ServerStatus {
        Down,
        Suspended,
        Running 
    } serverStatus;

    typedef std::map<std::string, std::shared_ptr<ZmqMultipleNamedClientBusProxy>>
    ClientsMap;
    typedef std::map<std::string, std::shared_ptr<EchoService>>
    ServicesMap;
    typedef std::map<std::string, std::shared_ptr<ServiceProxies>>
    ProxiesMap;

    ProxiesMap proxiesMap;
    ClientsMap clientsMap;
    ServicesMap servicesMap;

    std::unique_ptr<ZooKeeper::TemporaryServer> zooServer;

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

// TODO: Figure out a better name for this class
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

    void assertTriggeredWatches(const std::string &clientName,
                                uint32_t count)
    {
        auto client = scenario.getFromMap(scenario.clientsMap,
                                          clientName);

        BOOST_CHECK_EQUAL(client->changesCount[ConfigurationService::CREATED], 
                          count);
    }

private:
    ServiceDiscoveryScenario &scenario;
};

} // namespace Datacratic

#endif
