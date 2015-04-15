#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <iostream>
#include <boost/test/unit_test.hpp>

#include "soa/service/http_parsers.h"
#include "soa/utils/print_utils.h"

using namespace std;
using namespace Datacratic;


#if 1
/* This test does progressive testing of the HttpResponseParser by sending
 * only a certain amount of bytes to check for all potential parsing faults,
 * for each step (top line, headers, body, response restart, ...). */
BOOST_AUTO_TEST_CASE( http_response_parser_test )
{
    string statusLine;
    vector<string> headers;
    string body;
    bool done;
    bool shouldClose;

    HttpResponseParser parser;
    parser.onResponseStart = [&] (const string & httpVersion, int code) {
        cerr << "response start\n";
        statusLine = httpVersion + "/" + to_string(code);
        headers.clear();
        body.clear();
        shouldClose = false;
        done = false;
    };
    parser.onHeader = [&] (const char * data, size_t size) {
        // cerr << "header: " + string(data, size) + "\n";
        headers.emplace_back(data, size);
    };
    parser.onData = [&] (const char * data, size_t size) {
        // cerr << "data\n";
        body.append(data, size);
    };
    parser.onDone = [&] (bool doClose) {
        shouldClose = doClose;
        done = true;
    };

    /* status line */
    parser.feed("HTTP/1.");
    BOOST_CHECK_EQUAL(statusLine, "");
    parser.feed("1 200 Th");
    BOOST_CHECK_EQUAL(statusLine, "");
    parser.feed("is is ");
    BOOST_CHECK_EQUAL(statusLine, "");
    parser.feed("some blabla\r");
    BOOST_CHECK_EQUAL(statusLine, "");
    parser.feed("\n");
    BOOST_CHECK_EQUAL(statusLine, "HTTP/1.1/200");

    /* headers */
    parser.feed("Head");
    BOOST_CHECK_EQUAL(headers.size(), 0);
    parser.feed("er1: value1\r\nHeader2: value2");
    BOOST_CHECK_EQUAL(headers.size(), 1);
    BOOST_CHECK_EQUAL(headers[0], "Header1: value1\r\n");
    parser.feed("\r");
    BOOST_CHECK_EQUAL(headers.size(), 1);

    /* Headers require line ending + 1 character, since the latter requires
     * testing for multiline headers. */
    parser.feed("\n");
    BOOST_CHECK_EQUAL(headers.size(), 1);
    parser.feed("H");
    BOOST_CHECK_EQUAL(headers.size(), 2);
    BOOST_CHECK_EQUAL(headers[1], "Header2: value2\r\n");
    parser.feed("ead");
    BOOST_CHECK_EQUAL(headers.size(), 2);
    parser.feed("er3: Val3\r\nC");
    BOOST_CHECK_EQUAL(headers.size(), 3);
    BOOST_CHECK_EQUAL(headers[2], "Header3: Val3\r\n");
    parser.feed("ontent-Length: 10\r\n\r");
    parser.feed("\n");
    BOOST_CHECK_EQUAL(headers.size(), 5);
    BOOST_CHECK_EQUAL(headers[3], "Content-Length: 10\r\n");
    BOOST_CHECK_EQUAL(headers[4], "\r\n");
    BOOST_CHECK_EQUAL(parser.remainingBody(), 10);

    /* body */
    parser.feed("0123");
    parser.feed("456");
    parser.feed("789");
    BOOST_CHECK_EQUAL(body, "0123456789");
    BOOST_CHECK_EQUAL(done, true);

    /* one full response and a partial one without body */
    parser.feed("HTTP/1.1 204 No content\r\n"
                "MyHeader: my value1\r\n\r\nHTTP");

    BOOST_CHECK_EQUAL(statusLine, "HTTP/1.1/204");
    BOOST_CHECK_EQUAL(headers.size(), 2);
    BOOST_CHECK_EQUAL(headers[0], "MyHeader: my value1\r\n");
    BOOST_CHECK_EQUAL(headers[1], "\r\n");
    BOOST_CHECK_EQUAL(body, "");
    BOOST_CHECK_EQUAL(done, true);
    BOOST_CHECK_EQUAL(parser.remainingBody(), 0);

    parser.feed("/1.1 666 The number of the beast\r\n"
                "Connection: close\r\n"
                "Header: value\r\n\r\n");
    BOOST_CHECK_EQUAL(statusLine, "HTTP/1.1/666");
    BOOST_CHECK_EQUAL(headers.size(), 3);
    BOOST_CHECK_EQUAL(headers[0], "Connection: close\r\n");
    BOOST_CHECK_EQUAL(headers[1], "Header: value\r\n");
    BOOST_CHECK_EQUAL(headers[2], "\r\n");
    BOOST_CHECK_EQUAL(body, "");
    BOOST_CHECK_EQUAL(done, true);
    BOOST_CHECK_EQUAL(shouldClose, true);
    BOOST_CHECK_EQUAL(parser.remainingBody(), 0);

    /* 2 full reponses with body */
    const char * payload = ("HTTP/1.1 200 This is some blabla\r\n"
                            "Header1: value1\r\n"
                            "Header2: value2\r\n"
                            "Content-Type: text/plain\r\n"
                            "Content-Length: 10\r\n"
                            "\r\n"
                            "0123456789");
    parser.feed(payload);
    BOOST_CHECK_EQUAL(body, "0123456789");
    BOOST_CHECK_EQUAL(done, true);
    parser.feed(payload);
    BOOST_CHECK_EQUAL(body, "0123456789");
    BOOST_CHECK_EQUAL(done, true);
    BOOST_CHECK_EQUAL(shouldClose, false);
}
#endif

#if 1
/* Ensures that multiline headers are correctly parsed. */
BOOST_AUTO_TEST_CASE( http_parser_multiline_header_test )
{
    vector<string> headers;

    HttpResponseParser parser;
    parser.onResponseStart = [&] (const string & httpVersion, int code) {
        headers.clear();
    };
    parser.onHeader = [&] (const char * data, size_t size) {
        headers.emplace_back(data, size);
    };

    parser.feed("HTTP/1.1 200 This is some blabla\r\n");

    parser.feed("Header1: value1\r\nH");
    BOOST_CHECK_EQUAL(headers.size(), 1);
    BOOST_CHECK_EQUAL(headers[0], "Header1: value1\r\n");

    parser.feed("eader2: value2\r\n  with another line\r\nH");
    BOOST_CHECK_EQUAL(headers.size(), 2);
    BOOST_CHECK_EQUAL(headers[1], "Header2: value2 with another line\r\n");
    parser.feed("eader3: Val3\r\n\t with tab\r\n  and space\r\nH");
    BOOST_CHECK_EQUAL(headers.size(), 3);
    BOOST_CHECK_EQUAL(headers[2], "Header3: Val3 with tab and space\r\n");
    parser.feed("eader4: Value4\r\n \r\n\r\n");
    BOOST_CHECK_EQUAL(headers.size(), 5);
    BOOST_CHECK_EQUAL(headers[3], "Header4: Value4\r\n");
    BOOST_CHECK_EQUAL(headers[4], "\r\n");
}
#endif

#if 1
/* Ensures that chunked encoding is well supported. */
BOOST_AUTO_TEST_CASE( http_parser_chunked_encoding_test )
{
    /* missing error tests:
       - invalid hex value for chunk length
       - excessive chunk size (> chunk length)
       - content-length and content-coding are mutually exclusive
       - no chunk after last-chunk
    */

    HttpResponseParser parser;

    string chunkA = randomString(0xa);
    string chunk20 = randomString(0x20);
    string chunk100 = randomString(0x100);

    int numResponses(0);
    parser.onResponseStart = [&] (const string & httpVersion, int code) {
        numResponses++;
    };

    vector<string> bodyChunks;
    parser.onData = [&] (const char * data, size_t size) {
        bodyChunks.emplace_back(data, size);
    };

    parser.feed("HTTP/1.1 200 This is some blabla\r\n"
                "Header1: value1\r\n"
                "Transfer-Encoding: chunked\r\n"
                "\r\n");
    BOOST_CHECK_EQUAL(numResponses, 1);

    string feedData = "a\r\n" + chunkA + "\r\n";
    parser.feed(feedData.c_str(), feedData.size());
    BOOST_CHECK_EQUAL(bodyChunks.size(), 1);
    BOOST_CHECK_EQUAL(bodyChunks[0], chunkA);

    feedData = "A;someext\r\n" + chunkA + "\r\n";
    parser.feed(feedData.c_str(), feedData.size());
    BOOST_CHECK_EQUAL(bodyChunks.size(), 2);
    BOOST_CHECK_EQUAL(bodyChunks[1], chunkA);

    feedData = "20;someext\r\n" + chunk20 + "\r\n";
    parser.feed(feedData.c_str(), feedData.size());
    BOOST_CHECK_EQUAL(bodyChunks.size(), 3);
    BOOST_CHECK_EQUAL(bodyChunks[2], chunk20);

    feedData = "100;otherext=value\r\n" + chunk100 + "\r\n";
    parser.feed(feedData.c_str(), feedData.size());
    BOOST_CHECK_EQUAL(bodyChunks.size(), 4);
    BOOST_CHECK_EQUAL(bodyChunks[3], chunk100);

    feedData = "0000\r\n\r\n";
    parser.feed(feedData.c_str(), feedData.size());
    BOOST_CHECK_EQUAL(bodyChunks.size(), 4);

    BOOST_CHECK_EQUAL(numResponses, 1);

    /* another response can be fed */
    parser.feed("HTTP/1.1 200 This is some blabla\r\n"
                "Header1: value1\r\n"
                "Transfer-Encoding: chunked\r\n"
                "\r\n");
    BOOST_CHECK_EQUAL(numResponses, 2);

    /* we now test chunks of multiple chunks */
    bodyChunks.clear();

    feedData = ("20\r\n" + chunk20 + "\r\n"
                "20\r\n" + chunk20 + "\r\n"
                "20\r\n" + chunk20 + "\r\n"
                "0\r\n\r\n");
    parser.feed(feedData.c_str(), feedData.size());
    BOOST_CHECK_EQUAL(bodyChunks.size(), 3);
    BOOST_CHECK_EQUAL(bodyChunks[0], chunk20);
    BOOST_CHECK_EQUAL(bodyChunks[1], chunk20);
    BOOST_CHECK_EQUAL(bodyChunks[2], chunk20);

    /* yet another response can be fed */
    parser.feed("HTTP/1.1 200 This is some blabla\r\n"
                "Header1: value1\r\n"
                "Transfer-Encoding: chunked\r\n"
                "\r\n");
    BOOST_CHECK_EQUAL(numResponses, 3);
}
#endif
