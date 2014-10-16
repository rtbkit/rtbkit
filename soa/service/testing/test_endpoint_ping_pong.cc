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

using namespace std;
using namespace ML;
using namespace Datacratic;

BOOST_AUTO_TEST_CASE( test_ping_pong )
{
    BOOST_REQUIRE_EQUAL(TransportBase::created, TransportBase::destroyed);
    BOOST_REQUIRE_EQUAL(ConnectionHandler::created,
                        ConnectionHandler::destroyed);

    Watchdog watchdog(5.0);

    string connectionError;

    PassiveEndpointT<SocketTransport> acceptor("acceptor");
    
    acceptor.onMakeNewHandler = [&] ()
        {
            return ML::make_std_sp(new PongConnectionHandler(connectionError));
        };
    
    int port = acceptor.init();

    cerr << "port = " << port << endl;

    BOOST_CHECK_EQUAL(acceptor.numConnections(), 0);

    ActiveEndpointT<SocketTransport> connector("connector");
    int nconnections = 1;
    connector.init(port, "localhost", nconnections);

    ACE_Semaphore gotConnectionSem(0), finishedTestSem(0);

    auto onNewConnection = [&] (std::shared_ptr<TransportBase> transport)
        {
            transport->associate
            (ML::make_std_sp(new PingConnectionHandler(connectionError,
                                                   finishedTestSem)));
            gotConnectionSem.release();
        };

    auto onConnectionError = [&] (const std::string & error)
        {
            cerr << "onConnectionError " << error << endl;

            connectionError = error;
            gotConnectionSem.release();
        };

    connector.getConnection(onNewConnection,
                            onConnectionError,
                            1.0);
    
    ACE_Time_Value waitUntil(Date::now().plusSeconds(1.0).toAce());
    int semWaitRes = gotConnectionSem.acquire(waitUntil);

    BOOST_CHECK_EQUAL(semWaitRes, 0);
    BOOST_CHECK_EQUAL(connectionError, "");
    

    BOOST_CHECK_EQUAL(acceptor.numConnections(), 1);
    BOOST_CHECK_EQUAL(connector.numConnections(), 1);
    BOOST_CHECK_EQUAL(connector.numActiveConnections(), 1);
    BOOST_CHECK_EQUAL(connector.numInactiveConnections(), 0);

    waitUntil = Date::now().plusSeconds(5.0).toAce();
    semWaitRes = finishedTestSem.acquire(waitUntil);
    
    BOOST_CHECK_EQUAL(semWaitRes, 0);
    BOOST_CHECK_EQUAL(connectionError, "");

    acceptor.closePeer();

    connector.sleepUntilIdle();
    acceptor.sleepUntilIdle();

    connector.shutdown();
    acceptor.shutdown();
    
    BOOST_CHECK_EQUAL(TransportBase::created, TransportBase::destroyed);
    BOOST_CHECK_EQUAL(ConnectionHandler::created,
                      ConnectionHandler::destroyed);
}
