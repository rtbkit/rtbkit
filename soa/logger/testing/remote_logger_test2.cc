/* endpoint_test.cc
   Jeremy Barnes, 31 January 2011
   Copyright (c) 2011 Datacratic.  All rights reserved.

   Tests for the endpoints.
*/

#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include "soa/logger/remote_input.h"
#include "soa/logger/remote_output.h"
#include <sys/socket.h>
#include "jml/utils/guard.h"
#include "jml/arch/exception_handler.h"
#include "jml/utils/testing/watchdog.h"
#include "jml/utils/testing/fd_exhauster.h"
#include "jml/arch/timers.h"

using namespace std;
using namespace ML;
using namespace Datacratic;


BOOST_AUTO_TEST_CASE( test_remote_logger )
{
    // We try to stream a whole stack of messages through and check that they all
    // get to the other end.

    RemoteInput input;
    input.listen(-1, "localhost");
    int port = input.port();

    cerr << "port = " << port << endl;

    RemoteOutput output;
    output.connect(port, "localhost");

    for (int i = 0;  i < 1000;  ++i) {
        output.logMessage("channelname", "blah blah this is another message");
        output.barrier();
    }
    
    //output.close();

    output.shutdown();
    input.shutdown();
}
