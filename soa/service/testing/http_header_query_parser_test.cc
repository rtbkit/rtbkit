#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include "soa/service/http_header.h"

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
