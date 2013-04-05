/* zmq_tcp_bench.cc
   Wolfgang Sourdeau - April 2013 */

#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <fcntl.h>
#include <unistd.h>
#include <sys/epoll.h>
#include <sys/types.h>
#include <sys/socket.h>

#include <string>

#include <boost/test/unit_test.hpp>

#include "jml/arch/exception.h"
#include "jml/arch/futex.h"

#include "soa/service/service_base.h"
#include "soa/service/rest_service_endpoint.h"

const int NbrMsgs = 1000000;

using namespace std;
using namespace ML;

using namespace Datacratic;

#if 1
BOOST_AUTO_TEST_CASE( test_zmq )
{
    MessageLoop mainLoop;
    
    auto proxies = make_shared<ServiceProxies>();
    int recvMsgs(0), sendMsgs(0);
    struct timeval start, end;

    ZmqNamedEndpoint server(proxies->zmqContext);
    server.init(proxies->config, ZMQ_XREP, "server");
    auto onServerMessage = [&] (vector<zmq::message_t> && messages) {
        recvMsgs++;
        if (recvMsgs == sendMsgs) {
            futex_wake(recvMsgs);
        }
    };
    server.rawMessageHandler = onServerMessage;
    server.bindTcp();
    mainLoop.addSource("server", server);

    proxies->config->dump(cerr);

    ZmqNamedProxy client(proxies->zmqContext);
    client.init(proxies->config, ZMQ_XREQ, "client");
    mainLoop.addSource("client", client);

    client.connect("server");

    cerr << "awaiting connection\n";
    while (!client.isConnected()) {
        ML::sleep(1);
    }

    mainLoop.start();
    cerr << "connected and sending\n";

    gettimeofday(&start, NULL);

    for (int i = 0; i < NbrMsgs; i++) {
        client.sendMessage("test");
        sendMsgs++;
    }

    while (recvMsgs < sendMsgs) {
        // cerr << "awaiting end of messages: " << recvMsgs << "\n";
        ML::futex_wait(recvMsgs, recvMsgs);
    }
    cerr << "zmq test: received messages: " << recvMsgs << "\n";

    gettimeofday(&end, NULL);

    int delta_sec = (end.tv_sec - start.tv_sec);
    if (start.tv_usec > end.tv_usec) {
        delta_sec--;
        end.tv_usec += 1000000;
    }
    printf ("delta: %d.%.6ld\n", delta_sec, (end.tv_usec - start.tv_usec));
}
#endif

#if 1
#include "tcpsockets.h"

BOOST_AUTO_TEST_CASE( test_unix_tcp )
{
    auto proxies = make_shared<ServiceProxies>();
    int recvMsgs(0), sendMsgs(0);
    struct timeval start, end;

    TcpNamedEndpoint server;
    server.init(proxies->config, "server");
    auto onServerMessage = [&] (const string & message) {
        // cerr << "received tcp message: " << message << endl;
        // if (message != "test") {
        //     throw ML::Exception("error");
        // }
        recvMsgs++;
        if (recvMsgs == sendMsgs) {
            futex_wake(recvMsgs);
        }
    };
    server.onMessage_ = onServerMessage;
    server.bindTcp();

    TcpNamedProxy client;
    client.init(proxies->config);

    MessageLoop mainLoop;
    mainLoop.addSource("server", server);
    mainLoop.addSource("client", client);

    client.connectTo("127.0.0.1", 9876);

    mainLoop.start();
    cerr << "awaiting connection\n";
    while (!client.isConnected()) {
        ML::sleep(1);
    }
    ML::sleep(2);
    cerr << "connected and sending\n";

    gettimeofday(&start, NULL);

    for (int i = 0; i < NbrMsgs; i++) {
        client.sendMessage("test");
        sendMsgs++;
    }

    while (recvMsgs < sendMsgs) {
        // cerr << "awaiting end of messages: " << recvMsgs << "\n";
        ML::futex_wait(recvMsgs, recvMsgs);
    }
    cerr << "tcp test: received messages: " << recvMsgs << "\n";

    gettimeofday(&end, NULL);

    int delta_sec = (end.tv_sec - start.tv_sec);
    if (start.tv_usec > end.tv_usec) {
        delta_sec--;
        end.tv_usec += 1000000;
    }
    printf ("delta: %d.%.6ld\n", delta_sec, (end.tv_usec - start.tv_usec));

    mainLoop.shutdown();
}
#endif
