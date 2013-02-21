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
#include "jml/utils/guard.h"
#include "jml/arch/exception_handler.h"
#include "jml/utils/testing/watchdog.h"
#include "jml/utils/testing/fd_exhauster.h"
#include "jml/utils/vector_utils.h"
#include "jml/arch/timers.h"
#include <thread>
#include "soa/service/zmq_utils.h"


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
                const std::string & serviceName)
        : ServiceBase(serviceName, proxies),
          toClients(getZmqContext())
    {
        proxies->config->removePath(serviceName);
        registerServiceProvider(serviceName, { "echo" });

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

    std::string bindTcp(const std::string & host = "*")
    {
        return toClients.bindTcp(host);
    }

    ZmqNamedClientBus toClients;
};

BOOST_AUTO_TEST_CASE( test_early_connection )
{
    /** Test that we can do a "connect", then start the service, and
        have the connection come up once the service appears.
    */
    cerr << "Testing early connection..." << endl;
    auto proxies = std::make_shared<ServiceProxies>();
    proxies->useZookeeper();

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

#if 1
BOOST_AUTO_TEST_CASE( test_multiple_services )
{
    auto proxies = std::make_shared<ServiceProxies>();
    proxies->useZookeeper();
    proxies->config->removePath("");
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
