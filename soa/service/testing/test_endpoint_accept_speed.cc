/* endpoint_test.cc
   Jeremy Barnes, 31 January 2011
   Copyright (c) 2011 Datacratic.  All rights reserved.

   Tests for the endpoints.
*/

#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include "soa/service/http_endpoint.h"
#include "soa/service/active_endpoint.h"
#include "soa/service/passive_endpoint.h"
#include <sys/socket.h>
#include "jml/utils/guard.h"
#include "jml/arch/exception_handler.h"
#include "jml/utils/testing/watchdog.h"
#include "jml/utils/testing/fd_exhauster.h"
#include "test_connection_error.h"
#include "ping_pong.h"
#include <poll.h>
#include "jml/utils/exc_assert.h"


using namespace std;
using namespace ML;
using namespace Datacratic;

void runAcceptSpeedTest()
{
    string connectionError;

    PassiveEndpointT<SocketTransport> acceptor("acceptor");
    
    acceptor.onMakeNewHandler = [&] ()
        {
            return ML::make_std_sp(new PongConnectionHandler(connectionError));
        };
    
    int port = acceptor.init();

    cerr << "port = " << port << endl;

    BOOST_CHECK_EQUAL(acceptor.numConnections(), 0);

    int nconnections = 100;

    Date before = Date::now();

    vector<int> sockets;

    /* Open all the connections */
    for (unsigned i = 0;  i < nconnections;  ++i) {
        int s = socket(AF_INET, SOCK_STREAM, 0);
        if (s == -1)
            throw Exception("socket");

        //cerr << "i = " << i << " s = " << s << " sockets.size() = "
        //     << sockets.size() << endl;

        struct sockaddr_in addr = { AF_INET, htons(port), { INADDR_ANY } }; 
        //cerr << "before connect on " << s << endl;
        int res = connect(s, reinterpret_cast<const sockaddr *>(&addr),
                          sizeof(addr));
        //cerr << "after connect on " << s << endl;

        if (res == -1) {
            cerr << "connect error: " << strerror(errno) << endl;
            close(s);
        }
        else {
            sockets.push_back(s);
        }
    }

    /* Write to each and get a response back.  This makes sure that all are open. */
    for (unsigned i = 0;  i < sockets.size();  ++i) {
        int s = sockets[i];
        int res = write(s, "hello", 5);
        ExcAssertEqual(res, 5);
        
        char buf[16];
        
        res = read(s, buf, 16);
        ExcAssertEqual(res, 4);
        if (res > 0) {
            ExcAssertEqual(string(buf, buf + res), "Hi!!");
        }
    }

    Date after = Date::now();

    BOOST_CHECK_LT(after.secondsSince(before), 1);

    BOOST_CHECK_EQUAL(sockets.size(), nconnections);



    BOOST_CHECK_EQUAL(acceptor.numConnections(), nconnections);

    acceptor.closePeer();

    for (unsigned i = 0;  i < sockets.size();  ++i) {
        close(sockets[i]);
    }

    acceptor.shutdown();
}


BOOST_AUTO_TEST_CASE( test_accept_speed )
{
    BOOST_REQUIRE_EQUAL(TransportBase::created, TransportBase::destroyed);
    BOOST_REQUIRE_EQUAL(ConnectionHandler::created,
                        ConnectionHandler::destroyed);

    Watchdog watchdog(50.0);

    int ntests = 1;
    //ntests = 1000;  // stress test

    for (unsigned i = 0;  i < ntests;  ++i) {
        runAcceptSpeedTest();
    }

    BOOST_CHECK_EQUAL(TransportBase::created, TransportBase::destroyed);
    BOOST_CHECK_EQUAL(ConnectionHandler::created,
                      ConnectionHandler::destroyed);
}
