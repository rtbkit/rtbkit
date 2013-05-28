/* rotating_file_logger_test.cc
   Jeremy Barnes, 31 January 2011
   Copyright (c) 2011 Datacratic.  All rights reserved.

   Tests for the endpoints.
*/

#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
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

BOOST_AUTO_TEST_CASE( test_rotating_file_logger )
{
    vector<string> filesOpened;

    ML::Call_Guard guard([&] () { for (auto file: filesOpened) unlink(file.c_str());});

    RotatingFileOutput logger;
    
    int numLinesWritten = 0;

    logger.onPreFileOpen = [&] (string filename)
        {
            cerr << "opening file " << filename << endl;
            filesOpened.push_back(filename);
        };

    logger.onPreFileClose = [&] (string filename)
        {
            cerr << "pre closing file " << filename << endl;
        };

    logger.onPostFileOpen = [&] (string filename)
        {
            cerr << "post opening file " << filename << endl;
        };
    
    
    logger.open("tmp/file-logger-%F-%T.log.gz", "2s");

    for (unsigned i = 0;  i < 60;  ++i) {
        logger.logMessage("HELLO", "This is a message " + numLinesWritten);
        ++numLinesWritten;
        ML::sleep(0.1);
    }

    logger.close();

    BOOST_CHECK_GT(filesOpened.size(), 3);
    BOOST_CHECK_LE(filesOpened.size(), 6);

    int totalLines = 0;

    for (auto file: filesOpened) {
        filter_istream stream(file);
        while (stream) {
            string line;
            getline(stream, line);

            if (line != "")
                ++totalLines;
        }
    }

    BOOST_CHECK_EQUAL(totalLines, numLinesWritten);
}

