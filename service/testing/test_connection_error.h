/* test_connection_error.h                                         -*- C++ -*-
   Jeremy Barnes, 16 May 2011
   Copyright (c) 2011 Datacratic.  All rights reserved.

   Testing include for endpoint connection error tests.
*/

using namespace std;
using namespace ML;
using namespace Datacratic;


template<typename Endpoint>
void doTestConnectionError(Endpoint & connector,
                           const std::string & errorRequired,
                           const std::string & errorRequired2 = "")
{
    BOOST_CHECK_EQUAL(connector.numActiveConnections(), 0);
    BOOST_CHECK_EQUAL(connector.numInactiveConnections(), 0);
    BOOST_CHECK_EQUAL(connector.threadsActive(), 1);

    string errorMsg;
    bool succeeded = false;
    ACE_Semaphore sem(0);

    auto onNewConnection
        = [&] (const std::shared_ptr<TransportBase> & transport)
        {
            cerr << "new connection" << endl;
            BOOST_CHECK_EQUAL(typeid(*transport).name(),
                              typeid(SocketTransport).name());

            succeeded = true;
            sem.release();
            transport->closeAsync();
        };

    auto onConnectionError = [&] (std::string error)
        {
            cerr << "connection error " << error << endl;
            errorMsg = error;
            sem.release();
        };

    connector.getConnection(onNewConnection, onConnectionError,
                            1.0);

    int sem_acquire_res = sem.acquire();

    BOOST_CHECK_EQUAL(sem_acquire_res, 0);

    BOOST_CHECK_EQUAL(succeeded, false);
    cerr << "errorMsg = " << errorMsg << endl;
    cerr << "errorRequired = " << errorRequired << endl;
    bool found1 = errorMsg.find(errorRequired) != string::npos;
    bool found2 = errorRequired2 != ""
        && errorMsg.find(errorRequired2) != string::npos;
    BOOST_CHECK_EQUAL(found1 || found2, true);

    BOOST_CHECK_EQUAL(connector.numActiveConnections(), 0);
    BOOST_CHECK_EQUAL(connector.numInactiveConnections(), 0);
    BOOST_CHECK_EQUAL(connector.threadsActive(), 1);
}

