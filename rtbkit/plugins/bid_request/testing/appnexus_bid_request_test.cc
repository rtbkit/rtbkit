/*
 appexus_bid_request_test.cc
 Mark Weiss, 3 April 2013
 Copyright (c) 2013 Datacratic Inc. All rights reserved.

 Test cases for the AppNexus bid request parser.
 */

#define PARSE_TEST_BOILERPLATE(___bidRequestObjType) \
    printTestHeader(filename, testname); \
    StreamingJsonParsingContext context; \
    context.init(filename); \
    ___bidRequestObjType req; \
    DefaultDescription<___bidRequestObjType> desc; \
    desc.parseJson(&req, context); \
    if (!req.unparseable.isNull()) \
        cerr << "ERROR: FIELD NOT PARSED, unparseable: " << req.unparseable << endl; \
    StreamJsonPrintingContext printContext(cout); \
    desc.printJson(&req, printContext); \
    printTestFooter();

#define BID_REQUEST_CONVERSION_TEST_BOILERPLATE(____filename, ____defaultDescType) \
    string filename = ____filename; \
    StreamingJsonParsingContext context; \
    context.init(filename); \
    ____defaultDescType req; \
    DefaultDescription<____defaultDescType> desc; \
    desc.parseJson(&req, context);


#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include "rtbkit/plugins/bid_request/appnexus_bid_request.h"
#include "soa/types/json_parsing.h"
#include "rtbkit/plugins/bid_request/appnexus_parsing.h"
#include "jml/utils/filter_streams.h"

using namespace std;
using namespace ML;
using namespace Datacratic;
using namespace RTBKIT;


std::string loadFile(const std::string & filename)
{
    ML::filter_istream stream(filename);

    string result;

    while (stream) {
        string line;
        getline(stream, line);
        result += line + "\n";
    }

    return result;
}

void printTestHeader(const std::string & filename, const std::string & testname)
{
    cerr << endl << "*** LOADING " << filename << endl << "*** TEST " << testname << endl;
}

void printTestHeader(const std::string & testname)
{
    cerr << endl << "*** LOADING " << endl << "*** TEST " << testname << endl;
}

void printTestFooter()
{
    cerr << endl;
}

void parseParentBidRequest(const std::string & filename, const std::string & testname)
{
    PARSE_TEST_BOILERPLATE(AppNexus::BidRequest)
}

void parseChildBidInfo(const std::string & filename, const std::string & testname)
{
    PARSE_TEST_BOILERPLATE(AppNexus::BidInfo)
}

void parseChildSegment(const std::string & filename, const std::string & testname)
{
    PARSE_TEST_BOILERPLATE(AppNexus::Segment)
}

void parseChildInventoryAudit(const std::string & filename, const std::string & testname)
{
    PARSE_TEST_BOILERPLATE(AppNexus::InventoryAudit)
}

void parseChildTag(const std::string & filename, const std::string & testname)
{
    PARSE_TEST_BOILERPLATE(AppNexus::Tag)
}

void parseChildMember(const std::string & filename, const std::string & testname)
{
    PARSE_TEST_BOILERPLATE(AppNexus::Member)
}


BOOST_AUTO_TEST_CASE( test_parse_appnexus_parent_bid_request )
{
    parseParentBidRequest("rtbkit/plugins/bid_request/testing/appnexus_parent_bid_request.json", "test_parse_appnexus_parent_bid_request");
}

BOOST_AUTO_TEST_CASE( test_parse_appnexus_child_bid_info )
{
    parseChildBidInfo("rtbkit/plugins/bid_request/testing/appnexus_child_bid_info.json", "test_parse_appnexus_child_bid_info");
}

BOOST_AUTO_TEST_CASE( test_parse_appnexus_child_segment )
{
    parseChildSegment("rtbkit/plugins/bid_request/testing/appnexus_child_segment.json", "test_parse_appnexus_child_segment");
}

BOOST_AUTO_TEST_CASE( test_parse_appnexus_child_inventory_audit )
{
    parseChildInventoryAudit("rtbkit/plugins/bid_request/testing/appnexus_child_inventory_audit.json", "test_parse_appnexus_child_inventory_audit");
}

BOOST_AUTO_TEST_CASE( test_parse_appnexus_child_tag )
{
    parseChildTag("rtbkit/plugins/bid_request/testing/appnexus_child_tag.json", "test_parse_appnexus_tag");
}

BOOST_AUTO_TEST_CASE( test_parse_appnexus_child_member )
{
    parseChildMember("rtbkit/plugins/bid_request/testing/appnexus_child_member.json", "test_parse_appnexus_member");
}

BOOST_AUTO_TEST_CASE( test_openrtb_from_appnexus )
{
    printTestHeader("test_openrtb_from_appnexus");

    BID_REQUEST_CONVERSION_TEST_BOILERPLATE("rtbkit/plugins/bid_request/testing/appnexus_parent_bid_request.json", AppNexus::BidRequest)

    std::string provider = "DummyProvider";
    std::string exchange = "AppNexus";
    auto ortbReq = fromAppNexus(req, provider, exchange);

    // check that the thing did actually convert
    BOOST_REQUIRE (ortbReq);

    // cerr << endl << "** Value returned for field: " << bidRequest->user->id.toString() << endl;
    BOOST_CHECK_EQUAL(ortbReq->timeAvailableMs, 100);
    // 96.246.152.18
    // OpenRTB::Device
    BOOST_CHECK_EQUAL(ortbReq->device->ua, Datacratic::UnicodeString("Mozilla/5.0 (Macintosh; U; Intel Mac OS X 10.5; en-US;rv:1.9.0.3) Gecko/2008092414 Firefox/3.0.3"));
    BOOST_CHECK_EQUAL(ortbReq->device->language, Datacratic::UnicodeString("en-US,en;q=0.8"));
    BOOST_CHECK_EQUAL(ortbReq->device->flashver, "Flash available - version unknown");
    BOOST_CHECK_EQUAL(ortbReq->device->ip, "96.246.152.18");
    // BOOST_CHECK_EQUAL(ortbReq->device->ipv6, "96.246.152.18");
    BOOST_CHECK_EQUAL(ortbReq->device->carrier, Datacratic::UnicodeString("101"));
    BOOST_CHECK_EQUAL(ortbReq->device->language, Datacratic::UnicodeString("en-US,en;q=0.8"));
    BOOST_CHECK_EQUAL(ortbReq->device->make, Datacratic::UnicodeString("1001"));
    BOOST_CHECK_EQUAL(ortbReq->device->model, Datacratic::UnicodeString("10001"));
    BOOST_CHECK_EQUAL(ortbReq->device->geo->country, "US");
    BOOST_CHECK_EQUAL(ortbReq->device->geo->region, "NY");
    BOOST_CHECK_EQUAL(ortbReq->device->geo->city, Datacratic::UnicodeString("New York"));
    BOOST_CHECK_EQUAL(ortbReq->device->geo->zip, Datacratic::UnicodeString("10014"));
    BOOST_CHECK_EQUAL(ortbReq->device->geo->dma, "501");
    // NOTE: AN provides lat and long values at greater precision than float, which is what OpenRTB stores
    // So we cast the test value and use the BOOST_CHECK for testing floating point values for equaulity within a tolerance
    BOOST_CHECK_CLOSE(ortbReq->device->geo->lat.val, (float)38.7875232696533, 0.0000001);
    BOOST_CHECK_CLOSE(ortbReq->device->geo->lon.val, (float)-77.2614831924438, 0.0000001);
    BOOST_CHECK_EQUAL(ortbReq->device->os, Datacratic::UnicodeString("iPhone"));
    BOOST_CHECK_EQUAL(ortbReq->device->osv, Datacratic::UnicodeString("N/A"));

    // OpenRTB::User
    BOOST_CHECK_EQUAL(ortbReq->user->id.toString(), "2987961585469200400");
    BOOST_CHECK_EQUAL(ortbReq->user->gender, "male");
    BOOST_CHECK_EQUAL(ortbReq->user->yob.val, Date::now().year() - 50);

    // OpenRTB::Content
    // TODO What happened to OpenRTB Content struct?
    // BOOST_CHECK_EQUAL(ortbReq->content->url.toString(), "http://www.foodandwine.com/recipes/");

    // OpenRTB::App
    BOOST_CHECK_EQUAL(ortbReq->app->publisher->id.toInt(), 1111);
    BOOST_CHECK_EQUAL(ortbReq->app->id.toString(), "0382f0b5-91a5-11e2-adf6-28cfe94b60bf");

    // OpenRTB::Site
    BOOST_CHECK_EQUAL(ortbReq->site->id.toInt(), 476);

    // OpenRTB::Impression
    BOOST_CHECK_EQUAL(ortbReq->imp.front().id.toInt(), 8984480746668973000);
    for (int width : ortbReq->imp.front().banner->w) {
      BOOST_CHECK(width == 300 || width == 320);
    }
    for (int height : ortbReq->imp.front().banner->h) {
      BOOST_CHECK(height == 160 || height == 50);
    }
    // Cast convertible enum (enum but not an enum class) to int for testing
    // 0 == OpenRTB::AdPosition::Unknown, convertible enum, converted from mapped
    // AppNexus::AdPosition enum
    BOOST_CHECK_EQUAL(ortbReq->imp.front().banner->pos.val, 0);
    BOOST_CHECK_EQUAL(ortbReq->imp.front().bidfloor.val, 1.0);

    printTestFooter();

    cerr << "\n\nWOOT\n\n" << endl;
}

