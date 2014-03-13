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

using namespace std;
using namespace ML;
using namespace Datacratic;


BOOST_AUTO_TEST_CASE( test_active_endpoint_not_responding )
{
    BOOST_REQUIRE_EQUAL(TransportBase::created, TransportBase::destroyed);
    BOOST_REQUIRE_EQUAL(ConnectionHandler::created,
                        ConnectionHandler::destroyed);

    Watchdog watchdog(5.0);
    
    for (unsigned i = 0;  i < 1;  ++i) {
        // Listen but don't respond to connections.  When the backlog is
        // full, connections will hang and we can test the timeout
        // behaviour.
        int port = 12942;
        int backlog = 10;

        int s = socket(AF_INET, SOCK_STREAM, 0);
        BOOST_REQUIRE(s > 0);

        for (int j = 0;  j < 100;  ++j, ++port) {
            struct sockaddr_in addr = { AF_INET, htons(port), { INADDR_ANY } }; 
            errno = 0;
            int b = ::bind(s, reinterpret_cast<sockaddr *>(&addr), sizeof(addr));
            if (b == -1) {
                cerr << "couldn't bind to port " << port << ": "
                     << strerror(errno) << endl;
                continue;
            }
            BOOST_REQUIRE_EQUAL(b, 0);
            break;
        }
        
        BOOST_REQUIRE_EQUAL(string(strerror(errno)), string(strerror(0)));

        int r = listen(s, backlog);
        BOOST_REQUIRE(r == 0);

        // Connector to use up all of the connections
        ActiveEndpointT<SocketTransport> connector2("connector2");
        connector2.init(port, "localhost", 20, 1, true,
                        false /* throw on error */);
        
        cerr << "got " << connector2.numActiveConnections() << " active and "
             << connector2.numInactiveConnections() << " inactive connections"
             << endl;

        BOOST_CHECK_LT(connector2.numInactiveConnections(), 20);
        BOOST_CHECK_EQUAL(connector2.numActiveConnections(), 0);

        cerr << endl << "iter " << i << endl;
        ActiveEndpointT<SocketTransport> connector("connector");
        connector.init(port, "localhost", 0, 1, true);
        doTestConnectionError(connector, "connection timed out",
                              "Timer expired");
        cerr << "test returned" << endl;
        connector.shutdown();
        cerr << "connector down" << endl;
        connector2.shutdown();
        cerr << "connector2 down" << endl;

        BOOST_CHECK_EQUAL(connector2.numConnections(), 0);
        BOOST_CHECK_EQUAL(connector2.numInactiveConnections(), 0);
        BOOST_CHECK_EQUAL(connector2.numActiveConnections(), 0);

        BOOST_CHECK_EQUAL(TransportBase::created, TransportBase::destroyed);
        BOOST_CHECK_EQUAL(ConnectionHandler::created,
                          ConnectionHandler::destroyed);
    }

    BOOST_CHECK_EQUAL(TransportBase::created, TransportBase::destroyed);
    BOOST_CHECK_EQUAL(ConnectionHandler::created,
                      ConnectionHandler::destroyed);
}


