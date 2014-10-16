/* http_long_header_test.cc
   Jeremy Barnes, 31 January 2011
   Copyright (c) 2011 Datacratic.  All rights reserved.

   Test that we can't crash the server sending long headers.
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
#include <poll.h>

using namespace std;
using namespace ML;
using namespace Datacratic;


BOOST_AUTO_TEST_CASE( test_accept_speed )
{
    Watchdog watchdog(5.0);

    string connectionError;

    PassiveEndpointT<SocketTransport> acceptor("acceptor");

    string error;
    
    struct TestHandler : HttpConnectionHandler {

        TestHandler(string & error)
            : bytesDone(0), error(error)
        {
        }

        int bytesDone;
        string & error;

        virtual void handleData(const std::string & data)
        {

            bytesDone += data.size();
            if (bytesDone > 1000000)
                throw ML::Exception("allowed infinite headers");
            HttpConnectionHandler::handleData(data);
        }

        virtual void handleError(const std::string & error)
        {
            cerr << "got error " << error << endl;
            this->error = error;
        }

        virtual void
        handleHttpHeader(const HttpHeader & header)
        {
            
            cerr << "got header " << header << endl;
        }

        virtual void handleHttpChunk(const HttpHeader & header,
                                     const std::string & chunkHeader,
                                     const std::string & chunk)
        {
            cerr << "chunkHeader " << chunkHeader << endl;
            //cerr << "chunk has " << chunk.length() << " bytes" << endl;
        }

    };

    acceptor.onMakeNewHandler = [&] ()
        {
            return ML::make_std_sp(new TestHandler(error));
        };
    
    int port = acceptor.init();

    cerr << "port = " << port << endl;

    BOOST_CHECK_EQUAL(acceptor.numConnections(), 0);

    // Date before = Date::now();

    /* Open a connection */
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

    /* Try to write 32MB of headers into the socket. */
    const char * buf = "header: 9012345678901234567890\r\n";

    int written = 0;
    int writeError = 0;

    for (unsigned i = 0;  i < 1000000;  ++i) {
        int res = write(s, buf, strlen(buf));
        if (res > 0)
            written += res;
        else if (res == 0)
            throw ML::Exception("nothing written");
        else {
            writeError = errno;
            cerr << strerror(errno) << endl;
            cerr << "error after writing " << written << " bytes"
                 << endl;
            break;
        }
    }

    // Check we didn't write more than 1MB before the error...
    BOOST_CHECK_LT(written, 1000000);
    BOOST_CHECK_EQUAL(writeError, ECONNRESET);
    BOOST_CHECK_NE(error.find("HTTP header exceeds"), string::npos);

    cerr << "wrote " << written << " bytes" << endl;

    close(s);

    acceptor.shutdown();
}
