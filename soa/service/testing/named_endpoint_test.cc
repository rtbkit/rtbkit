/* named_endpoint_test.cc                                          -*- C++ -*-
   Jeremy Barnes, 24 September 2012
   Copyright (c) 2012 Datacratic Inc.  All rights reserved.

   Test for named endpoint.
*/

#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include <boost/make_shared.hpp>
#include "soa/service/named_endpoint.h"
#include "soa/service/message_loop.h"
#include "soa/service/zmq_endpoint.h"
#include <sys/socket.h>
#include "jml/utils/guard.h"
#include "jml/arch/exception_handler.h"
#include "jml/utils/testing/watchdog.h"
#include "jml/utils/testing/fd_exhauster.h"
#include "jml/utils/vector_utils.h"
#include "jml/arch/timers.h"
#include <thread>
#include "soa/service/zmq_utils.h"
#include "soa/service/testing/zookeeper_temporary_server.h"


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
          context(new zmq::context_t(1)),
          endpoint(context),
          loop(1 /* num threads */, 0.0001 /* maxAddedLatency */)
    {
        proxies->config->removePath(serviceName);
        //registerService();
        endpoint.init(proxies->config, ZMQ_XREP, serviceName + "/echo");

        auto handler = [=] (vector<string> message)
            {
                //cerr << "got message " << message << endl;
                ExcAssertEqual(message.size(), 3);
                ExcAssertEqual(message[1], "ECHO");
                message[1] = "REPLY";

                endpoint.sendMessage(message);
            };

        endpoint.messageHandler = handler;

        loop.addSource("EchoService::endpoint", endpoint);
    }

    void start()
    {
        loop.start();
    }

    void shutdown()
    {
        loop.shutdown();
    }

    std::string bindTcp()
    {
        return endpoint.bindTcp();
    }

    std::shared_ptr<zmq::context_t> context;
    ZmqNamedEndpoint endpoint;
    MessageLoop loop;
};

BOOST_AUTO_TEST_CASE( test_named_endpoint )
{
    ZooKeeper::TemporaryServer zookeeper;
    zookeeper.start();

    auto proxies = std::make_shared<ServiceProxies>();
    proxies->useZookeeper(ML::format("localhost:%d", zookeeper.getPort()));

    EchoService service(proxies, "echo");
    auto addr = service.bindTcp();
    cerr << "echo service is listening on " << addr << endl;

    service.start();

    proxies->config->dump(cerr);


    volatile int numPings = 0;

    auto runThread = [&] ()
        {
            ZmqNamedProxy proxy;
            proxy.init(proxies->config, ZMQ_XREQ);
            proxy.connect("echo/echo");

            ML::sleep(0.1);
    
            cerr << "connected" << endl;

            while (numPings < 100000) {
                int i = __sync_add_and_fetch(&numPings, 1);

                if (i && i % 1000 == 0)
                    cerr << i << endl;

                vector<string> request;
                request.push_back("ECHO");
                request.push_back(to_string(i));

                sendAll(proxy.socket(), request);

                vector<string> res = recvAll(proxy.socket());

                ExcAssertEqual(res.size(), 2);
                ExcAssertEqual(res[0], "REPLY");
                ExcAssertEqual(res[1], to_string(i));
            }
        };

    boost::thread_group threads;
    for (unsigned i = 0;  i < 10;  ++i) {
        threads.create_thread(runThread);
    }

    threads.join_all();

    cerr << "finished requests" << endl;

    service.shutdown();
}
