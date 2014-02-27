/* endpoint_test.cc
   Jeremy Barnes, 31 January 2011
   Copyright (c) 2011 Datacratic.  All rights reserved.

   Tests for the endpoints.
*/

#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include "soa/logger/multi_output.h"
#include "soa/logger/file_output.h"
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
    vector<string> filesOpened;

    ML::Call_Guard guard([&] () { for (auto file: filesOpened) unlink(file.c_str());});


    // We try to stream a whole stack of messages through and check that they all
    // get to the other end.

    MultiOutput output;

    auto createOutput = [&] (string key) -> std::shared_ptr<LogOutput>
        {
            cerr << "creating output for " << key << endl;
            filesOpened.push_back(key);
            return make_shared<FileOutput>(key);
        };

    output.logTo("", "tmp/logs-$(0)-$(1).txt",
                 createOutput);

    output.logMessage("HELLO", "dogs\tone");
    output.logMessage("HELLO", "dogs\ttwo");

    BOOST_CHECK_EQUAL(filesOpened, vector<string>({"tmp/logs-HELLO-dogs.txt"}));
    BOOST_CHECK_EQUAL(filesOpened, vector<string>({"tmp/logs-HELLO-dogs.txt"}));

    output.logMessage("HELLO", "cats\tone");
    output.logMessage("HELLO", "cats\ttwo");

    BOOST_CHECK_EQUAL(filesOpened, vector<string>({"tmp/logs-HELLO-dogs.txt",
                                                   "tmp/logs-HELLO-cats.txt"}));

#if 0
    output.connect(port, "localhost");

    for (int i = 0;  i < 1000;  ++i) {
        output.logMessage("channelname", "blah blah this is another message");
        output.barrier();
    }
    
    //output.close();

    output.shutdown();
    input.shutdown();
#endif
}
