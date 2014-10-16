#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <string>
#include <boost/test/unit_test.hpp>
#include "soa/service/http_header.h"

using namespace std;

/* Ensure that long long/int64 header content-length are supported */
BOOST_AUTO_TEST_CASE(test_http_header_long_long)
{
    Datacratic::HttpHeader header;

    string response = ("HTTP/1.1 200 OK\r\n"
                       "x-amz-id-2: a4KTwfjQazJISfG4fout1ZGxh8zT9okHNw+x0IK9yeOf13sfBCPWaTU9NVX9UYDe\r\n"
                       "x-amz-request-id: 5C7BFD14CC6988A3\r\n"
                       "Date: Tue, 03 Jun 2014 12:19:58 GMT\r\n"
                       "Last-Modified: Tue, 03 Jun 2014 02:55:25 GMT\r\n"
                       "ETag: \"0d24e1a5d399439dcf60907de62101e8-173\"\r\n"
                       "Accept-Ranges: bytes\r\n"
                       "Content-Type: binary/octet-stream\r\n"
                       "Content-Length: 10893368309\r\n"
                       "Server: AmazonS3\r\n"
                       "\r\n");
    header.parse(response, false);

    BOOST_CHECK_EQUAL(header.contentLength, 10893368309);
}


/* query parsing */

BOOST_AUTO_TEST_CASE(test_http_header_query_parser_no_query)
{
    const std::string query = "GET /bid? HTTP/1.1\r\n"
                              "\r\n";
    Datacratic::HttpHeader header;
    header.parse(query);
}

BOOST_AUTO_TEST_CASE(test_http_header_query_parser_no_var)
{
    const std::string query = "GET /bid?& HTTP/1.1\r\n"
                              "\r\n";
    Datacratic::HttpHeader header;
    header.parse(query);
}

namespace {

Datacratic::HttpHeader generateParser(const std::string& q)
{

    const std::string query = "GET " + q + " HTTP/1.1\r\n\r\n";
    Datacratic::HttpHeader parser;
    parser.parse(query);
    return std::move(parser);
}

void testQueryParam(const Datacratic::HttpHeader& header,
                    const std::string& var,
                    const std::string& expectedValue)
{
    auto const& val = header.queryParams.getValue(var);
    BOOST_CHECK_EQUAL(val, expectedValue);
}
} // namespace

// From Google url
BOOST_AUTO_TEST_CASE(test_http_header_query_parser_g1)
{
    auto header = generateParser("/?arg1=1&arg2=2&bar");

    testQueryParam(header, "arg1", "1");
    testQueryParam(header, "arg2", "2");
    testQueryParam(header, "bar", "");
}

BOOST_AUTO_TEST_CASE(test_http_header_query_parser_g2)
{
    // Empty param at the end.
    auto parser = generateParser("/?foo=bar&");
    testQueryParam(parser, "foo", "bar");
}

BOOST_AUTO_TEST_CASE(test_http_header_query_parser_g3)
{
    // Empty param at the beginning.
    auto parser = generateParser("/?&foo=bar");
    testQueryParam(parser, "", "");
    testQueryParam(parser, "foo", "bar");
}

BOOST_AUTO_TEST_CASE(test_http_header_query_parser_g4)
{
    // Empty key with value.
    auto parser = generateParser("http://www.google.com?=foo");
    testQueryParam(parser, "", "foo");
}

BOOST_AUTO_TEST_CASE(test_http_header_query_parser_g5)
{
    // Empty value with key.
    auto parser = generateParser("/?foo=");
    testQueryParam(parser, "foo", "");
}

BOOST_AUTO_TEST_CASE(test_http_header_query_parser_g6)
{
    // Empty key and values.
    auto parser = generateParser("/?&&==&=");
    BOOST_CHECK_EQUAL(parser.queryParams[0].second, "");
    BOOST_CHECK_EQUAL(parser.queryParams[1].second, "");
    BOOST_CHECK_EQUAL(parser.queryParams[2].second, "=");
    BOOST_CHECK_EQUAL(parser.queryParams[3].second, "");
}
