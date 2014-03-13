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
        const zmq::message_t & msg = messages[1];
        string message((const char *)msg.data(),
                       ((const char *)msg.data()) + msg.size());
        string expected("test" + to_string(recvMsgs));
        ExcAssertEqual(message, expected);
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
        ML::sleep(0.1);
    }

    mainLoop.start();
    cerr << "connected and sending\n";

    gettimeofday(&start, NULL);

    for (int i = 0; i < NbrMsgs; i++) {
        client.sendMessage("test" + to_string(sendMsgs));
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
    printf ("delta: %d.%.6ld secs\n", delta_sec, (end.tv_usec - start.tv_usec));
}
#endif

#if 1
#include "tcpsockets.h"

BOOST_AUTO_TEST_CASE( test_unix_tcp )
{
    auto proxies = make_shared<ServiceProxies>();
    MessageLoop mainLoop;
    int recvMsgs(0), sendMsgs(0);
    struct timeval start, end;

    TcpNamedEndpoint server;
    server.init(proxies->config, "server");
    auto onServerMessage = [&] (const string & message) {
        // cerr << "received tcp message: " << message << endl;
        string expected("test" + to_string(recvMsgs));
        ExcAssertEqual(message, expected);
        recvMsgs++;
        if (recvMsgs == sendMsgs) {
            futex_wake(recvMsgs);
        }
    };
    server.onMessage_ = onServerMessage;
    server.bindTcp(9876);
    ML::sleep(1);

    TcpNamedProxy client;
    client.init(proxies->config);

    mainLoop.addSource("server", server);
    mainLoop.addSource("client", client);
    mainLoop.start();

    client.connectTo("127.0.0.1", 9876);

    cerr << "awaiting connection\n";
    while (!client.isConnected()) {
        ML::sleep(0.1);
    }
    cerr << "connected and sending\n";

    gettimeofday(&start, NULL);

    for (sendMsgs = 0; sendMsgs < NbrMsgs;) {
        if (client.sendMessage("test" + to_string(sendMsgs))) {
            sendMsgs++;
        }
        // else {
        //     ML::sleep(0.1);
        // }
    }

    cerr << "sent " << sendMsgs << " messages\n";
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
    printf ("delta: %d.%.6ld secs\n", delta_sec, (end.tv_usec - start.tv_usec));
}
#endif
