/* json_service_endpoint_test.cc
   Jeremy Barnes, 9 November 2012
   Copyright (c) 2012 Datacratic Inc.  All rights reserved.

   Test for the JSON service endpoint.
*/

#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include <boost/make_shared.hpp>
#include "soa/service/named_endpoint.h"
#include "soa/service/message_loop.h"
#include "soa/service/rest_service_endpoint.h"
#include "soa/service/rest_proxy.h"
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

struct EchoService : public ServiceBase, public RestServiceEndpoint {

    EchoService(std::shared_ptr<ServiceProxies> proxies,
                const std::string & serviceName)
        : ServiceBase(serviceName, proxies),
          RestServiceEndpoint(proxies->zmqContext)
    {
        proxies->config->removePath(serviceName);
        RestServiceEndpoint::init(proxies->config,
                                  serviceName, 0.0005 /* maxAddedLatency */);
    }

    ~EchoService()
    {
        shutdown();
    }

    virtual void handleRequest(const ConnectionId & connection,
                               const RestRequest & request) const
    {
        //cerr << "handling request " << request << endl;
        if (request.verb != "POST")
            throw ML::Exception("echo service needs POST");
        if (request.resource != "/echo")
            throw ML::Exception("echo service only responds to /echo");
        connection.sendResponse(200, request.payload, "text/plain");
    }
};

BOOST_AUTO_TEST_CASE( test_named_endpoint )
{
    ZooKeeper::TemporaryServer zookeeper;
    zookeeper.start();

    auto proxies = std::make_shared<ServiceProxies>();
    proxies->useZookeeper(ML::format("localhost:%d", zookeeper.getPort()));

    int totalPings = 1000;

    EchoService service(proxies, "echo");
    auto addr = service.bindTcp();
    cerr << "echo service is listening on " << addr.first << " and "
         << addr.second << endl;

    service.start();

    proxies->config->dump(cerr);


    volatile int numPings = 0;

    auto runZmqThread = [=, &numPings] ()
        {
            RestProxy proxy(proxies->zmqContext);
            proxy.init(proxies->config, "echo");
            proxy.start();
            cerr << "connected" << endl;

            volatile int numOutstanding = 0;

            while (numPings < totalPings) {
                int i = __sync_add_and_fetch(&numPings, 1);

                if (i && i % 1000 == 0)
                    cerr << i << " with " << numOutstanding << " outstanding"
                         << endl;

                auto onResponse = [=, &numOutstanding]
                    (std::exception_ptr ptr,
                     int responseCode,
                     std::string body)
                    {
                        //cerr << "got response " << responseCode
                        //     << endl;
                        ML::atomic_dec(numOutstanding);

                        if (ptr)
                            throw ML::Exception("response returned exception");
                        ExcAssertEqual(responseCode, 200);
                        ExcAssertEqual(body, to_string(i));

                        futex_wake(numOutstanding);
                    };
                
                proxy.push(onResponse,
                           "POST", "/echo", {}, to_string(i));
                ML::atomic_inc(numOutstanding);
            }

            proxy.sleepUntilIdle();

            //ML::sleep(1.0);

            cerr << "shutting down proxy " << this << endl;
            proxy.shutdown();
            cerr << "done proxy shutdown" << endl;
        };

#if 0
    auto runHttpThread = [&] ()
        {
            HttpNamedRestProxy proxy;
            proxy.init(proxies->config);
            proxy.connect("echo/http");

            while (numPings < totalPings) {
                int i = __sync_add_and_fetch(&numPings, 1);

                if (i && i % 1000 == 0)
                    cerr << i << endl;

                auto response = proxy.post("/echo", to_string(i));
                
                ExcAssertEqual(response.code_, 200);
                ExcAssertEqual(response.body_, to_string(i));
            }

        };
#endif

    boost::thread_group threads;

    for (unsigned i = 0;  i < 8;  ++i) {
        threads.create_thread(runZmqThread);
    }

    //for (unsigned i = 0;  i < 5;  ++i) {
    //    threads.create_thread(runHttpThread);
    //}

    threads.join_all();

    cerr << "finished requests" << endl;

    service.shutdown();
}
