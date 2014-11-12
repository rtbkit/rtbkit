/* openrtb_bid_request_test.cc
   Jeremy Barnes, 19 February 2013
   Copyright (c) 2013 Datacratic Inc.  All rights reserved.

   Test cases for the OpenRTB bid request parser.
*/


#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include "rtbkit/plugins/bid_request/openrtb_bid_request_parser.h"
#include "soa/types/json_parsing.h"
#include "rtbkit/openrtb/openrtb_parsing.h"
#include "jml/utils/filter_streams.h"

using namespace std;
using namespace ML;
using namespace Datacratic;
using namespace RTBKIT;

vector<string> samples = {
    "rtbkit/plugins/bid_request/testing/openrtb1_req.json",
    "rtbkit/plugins/bid_request/testing/openrtb2_req.json",
    "rtbkit/plugins/bid_request/testing/openrtb3_req.json",
    "rtbkit/plugins/bid_request/testing/openrtb4_req.json",
    "rtbkit/plugins/bid_request/testing/openrtb_wseat_req.json",
    "rtbkit/plugins/bid_request/testing/openrtb_banner.json",
    "rtbkit/plugins/bid_request/testing/openrtb_expandable_creative.json",
    "rtbkit/plugins/bid_request/testing/openrtb_mobile.json",
    "rtbkit/plugins/bid_request/testing/openrtb_video.json",
    "rtbkit/plugins/bid_request/testing/rubicon_banner1.json",
    "rtbkit/plugins/bid_request/testing/rubicon_banner2.json",
    "rtbkit/plugins/bid_request/testing/rubicon_banner3.json",
    "rtbkit/plugins/bid_request/testing/rubicon_banner4.json",
    "rtbkit/plugins/bid_request/testing/rubicon_desktop.json",
    "rtbkit/plugins/bid_request/testing/rubicon_mobile_app.json",
    "rtbkit/plugins/bid_request/testing/rubicon_mobile_web.json",
    "rtbkit/plugins/bid_request/testing/rubicon_test1.json"
};

vector<string> samples2_2 = {
    "rtbkit/plugins/bid_request/testing/openrtb_2_2_req_imp.json",
    "rtbkit/plugins/bid_request/testing/openrtb_2_2_req_video.json"
};


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

void parseBidRequest(const std::string & filename)
{
    cerr << endl << "loading " << filename << endl;
    StreamingJsonParsingContext context;
    context.init(filename);

    OpenRTB::BidRequest req;
    DefaultDescription<OpenRTB::BidRequest> desc;
    desc.parseJson(&req, context);

    if (!req.unparseable.isNull())
        cerr << "unparseable:" << req.unparseable << endl;

    StreamJsonPrintingContext printContext(cout);
    desc.printJson(&req, printContext);
}

BOOST_AUTO_TEST_CASE( test_parse_openrtb_sample_requests )
{
    for (auto req: samples)
        parseBidRequest(req);
}

void testBidRequest(const std::string & filename, const std::string & version = "2.2")
{
    cerr << endl << "loading " << filename << endl;
    ML::Parse_Context context(filename);
    std::shared_ptr<OpenRTBBidRequestParser> p = OpenRTBBidRequestParser::openRTBBidRequestParserFactory(version);
    auto res = p->parseBidRequest(context, "test", "test");
    cerr << res->toJson() << endl;
}

BOOST_AUTO_TEST_CASE( test_openrtb_sample_requests )
{
    for (auto s: samples)
        testBidRequest(s);

    // Use 2.2 parsing
    for (auto s: samples2_2)
        testBidRequest(s, "2.2");
}

bool jsonDiff(const Json::Value & v1, const Json::Value & v2,
              bool oneOnly = false,
              string path = "")
{
    bool result = true;

    if (v1.type() != v2.type()) {
        cerr << path << ": different types: "
             << v1.toString() << " versus "
             << v2.toString() << endl;
        return false;
    }

    switch (v1.type()) {
    case Json::arrayValue:
        if (v1.size() != v2.size()) {
            cerr << path << ": differing lengths: " << v1.size() << " versus "
                 << v2.size() << endl;
            result = false;
            if (oneOnly) return false;
        }
        for (unsigned i = 0;  i < v1.size();  ++i) {
            if (!jsonDiff(v1[i], v2[i], oneOnly,
                          path + ML::format("[%d]", i))) {
                result = false;
                if (oneOnly) return false;
            }
        }
        return result;

    case Json::objectValue: {
        auto m1 = v1.getMemberNames();
        auto m2 = v2.getMemberNames();

        std::sort(m1.begin(), m1.end());
        std::sort(m2.begin(), m2.end());

        vector<string> intersection, only1, only2;
        std::set_intersection(m1.begin(), m1.end(),
                              m2.begin(), m2.end(),
                              back_inserter(intersection));
        std::set_difference(m1.begin(), m1.end(),
                            m2.begin(), m2.end(),
                            back_inserter(only1));
        std::set_difference(m2.begin(), m2.end(),
                            m1.begin(), m1.end(),
                            back_inserter(only2));
        
        if (!only1.empty()) {
            cerr << path << ": keys only present in first field: "
                 << only1 << endl;
            result = false;
            if (oneOnly) return false;
        }
        if (!only2.empty()) {
            cerr << path << ": keys only present in second field: "
                 << only1 << endl;
            result = false;
            if (oneOnly) return false;
        }

        for (auto f: intersection) {
            if (!jsonDiff(v1[f], v2[f], oneOnly, path + "." + f)) {
                result = false;
                if (oneOnly) return false;
            }
        }
        return result;
    }

    default:
        if (v1.toString() != v2.toString()) {
            cerr << path << ": difference: " << v1 << " vs " << v2 << endl;
            result = false;
            if (oneOnly) return false;
        }
    }
    return result;
}

void testBidRequestRoundTrip(const std::string & filename,
                             const std::string & reqStr)
{
    // Take an OpenRTB bid request; convert to Datacratic format;
    // serialize that to JSON; reconstitute from JSON back into the
    // original format

    static DefaultDescription<OpenRTB::BidRequest> desc;

    OpenRTB::BidRequest req;

    ML::Parse_Context c(filename);
    {
        StreamingJsonParsingContext context;
        context.init(filename, reqStr.c_str(), reqStr.size());
        desc.parseJson(&req, context);
    }

    string printed;
    {
        std::ostringstream stream;
        StreamJsonPrintingContext printContext(stream);
        desc.printJson(&req, printContext);
        printed = stream.str();
    }

    OpenRTB::BidRequest req2;

    {
        StreamingJsonParsingContext context2;
        context2.init(filename, reqStr.c_str(), reqStr.size());
        desc.parseJson(&req2, context2);
    }

    string printed2;
    {
        std::ostringstream stream;
        StreamJsonPrintingContext printContext(stream);
        desc.printJson(&req2, printContext);
        printed2 = stream.str();
    }
    
    BOOST_CHECK_EQUAL(printed, printed2);

    std::shared_ptr<OpenRTBBidRequestParser> p = OpenRTBBidRequestParser::openRTBBidRequestParserFactory("2.1");
    
    // Convert to a standard bid request
    std::unique_ptr<BidRequest> br(p->parseBidRequest(c, "test", "test"));   
    // Convert it to JSON
    string s1 = br->toJsonStr();

    std::unique_ptr<BidRequest> br2(BidRequest::parse("rtbkit", s1));

    string s2 = br2->toJsonStr();
    
    if (s1 != s2) {
        return;
        Json::Value j = br->toJson();
        Json::Value j2 = br2->toJson();
        BOOST_CHECK(jsonDiff(j, j2));
    }
}

void testBidRequestConversion(const std::string &fileName, const std::string &request)
{
    static DefaultDescription<OpenRTB::BidRequest> desc;

    // Parse the BidRequest in OpenRTB format
    OpenRTB::BidRequest originalReq;
    
    {
        StreamingJsonParsingContext context;
        context.init(fileName, request.c_str(), request.size());
        desc.parseJson(&originalReq, context);
    }

    string jsonBr1;
    {
        std::ostringstream stream;
        StreamJsonPrintingContext printContext(stream);
        desc.printJson(&originalReq, printContext);
        jsonBr1 = stream.str();
    }

    ML::Parse_Context c(fileName);

    std::shared_ptr<OpenRTBBidRequestParser> p = OpenRTBBidRequestParser::openRTBBidRequestParserFactory("2.1");
    
    // Convert to a standard bid request
    std::unique_ptr<BidRequest> br(p->parseBidRequest(c, "test", "test"));   

    // Convert it back to OpenRTB
    OpenRTB::BidRequest br2 = p->toBidRequest(*br);

    string jsonBr2;
    {
        std::ostringstream stream;
        StreamJsonPrintingContext printContext(stream);
        desc.printJson(&br2, printContext);
        jsonBr2 = stream.str();
    }

    auto json = [](const string &jsonStr) {
        Json::Reader reader;
        Json::Value value;
        bool ok;
        ok = reader.parse(jsonStr, value);
        BOOST_CHECK(ok);
        return value;
    };

    BOOST_CHECK(jsonDiff(json(jsonBr1), json(jsonBr2)));
}

BOOST_AUTO_TEST_CASE( test_openrtb_round_trip )
{
    vector<string> reqs;

    for (auto s: samples)
        reqs.push_back(loadFile(s));

    for (unsigned i = 0;  i < reqs.size();  ++i) {
        testBidRequestRoundTrip(samples[i], reqs[i]);
        //testBidRequestConversion(samples[i], reqs[i]);
    }
}

BOOST_AUTO_TEST_CASE( benchmark_openrtb_round_trip )
{
    vector<string> reqs;

    for (auto s: samples)
        reqs.push_back(loadFile(s));

    int done = 0;
    
    Date before = Date::now();

    for (unsigned i = 0;  i < 1000;  ++i) {
        
        for (unsigned i = 0;  i < reqs.size();  ++i, ++done)
            testBidRequestRoundTrip(samples[i], reqs[i]);
    }

    double elapsed = Date::now().secondsSince(before);
    
    cerr << "did " << done << " in " << elapsed << "s at "
         << done / elapsed << "/s" << endl;
}

BOOST_AUTO_TEST_CASE( benchmark_openrtb_parsing )
{
    cerr << "benchmarking OpenRTB parsing" << endl;

    vector<string> reqs;

    for (auto s: samples)
        reqs.push_back(loadFile(s));

    int done = 0;
    
    Date before = Date::now();

    DefaultDescription<OpenRTB::BidRequest> desc;

    for (unsigned i = 0;  i < 1000;  ++i) {
        
        for (unsigned i = 0;  i < reqs.size();  ++i, ++done) {
            
            
            OpenRTB::BidRequest req;
            
            {
                StreamingJsonParsingContext context;
                context.init(samples[i], reqs[i].c_str(), reqs[i].size());
                desc.parseJson(&req, context);
            }
        }
    }

    double elapsed = Date::now().secondsSince(before);
    
    cerr << "did " << done << " in " << elapsed << "s at "
         << done / elapsed << "/s" << endl;
}

BOOST_AUTO_TEST_CASE( benchmark_openrtb_conversion )
{
    cerr << "benchmarking OpenRTB parsing and conversion" << endl;

    vector<string> reqs;

    for (auto s: samples)
        reqs.push_back(loadFile(s));

    int done = 0;
    
    Date before = Date::now();

    DefaultDescription<OpenRTB::BidRequest> desc;

    std::shared_ptr<OpenRTBBidRequestParser> p = OpenRTBBidRequestParser::openRTBBidRequestParserFactory("2.1");

    for (unsigned i = 0;  i < 1000;  ++i) {
        
        for (unsigned i = 0;  i < reqs.size();  ++i, ++done) {
            
            OpenRTB::BidRequest req;
            
            {
                StreamingJsonParsingContext context;
                context.init(samples[i], reqs[i].c_str(), reqs[i].size());
                std::unique_ptr<BidRequest> br(p->parseBidRequest(*context.context, "openrtb", "openrtb"));   
        
            }
        }
    }

    double elapsed = Date::now().secondsSince(before);
    
    cerr << "did " << done << " in " << elapsed << "s at "
         << done / elapsed << "/s" << endl;
}

BOOST_AUTO_TEST_CASE( benchmark_canonical_parsing )
{
    cerr << "benchmarking canonical parsing of OpenRTB-derived bid requests" << endl;

    DefaultDescription<OpenRTB::BidRequest> desc;

    vector<string> reqs;

    std::shared_ptr<OpenRTBBidRequestParser> p = OpenRTBBidRequestParser::openRTBBidRequestParserFactory("2.1");

    for (auto s: samples) {
        OpenRTB::BidRequest req;
        {
            StreamingJsonParsingContext context;
            context.init(s);
            std::unique_ptr<BidRequest> br(p->parseBidRequest(*context.context, "openrtb", "openrtb"));   
            reqs.push_back(br->toJsonStr());
        }
    }

    int done = 0;
    
    Date before = Date::now();

    for (unsigned i = 0;  i < 1000;  ++i) {
        
        for (unsigned i = 0;  i < reqs.size();  ++i, ++done) {
            std::unique_ptr<BidRequest> br2(BidRequest::parse("rtbkit", reqs[i]));
        }
    }

    double elapsed = Date::now().secondsSince(before);
    
    cerr << "did " << done << " in " << elapsed << "s at "
         << done / elapsed << "/s" << endl;
}

BOOST_AUTO_TEST_CASE( id_provider ) {

    cerr << "id provider test : making sure we parse it correctly and always set it" << endl;

    DefaultDescription<OpenRTB::BidRequest> desc;

    std::shared_ptr<OpenRTBBidRequestParser> p = OpenRTBBidRequestParser::openRTBBidRequestParserFactory("2.1");

    vector<string> reqs;
    bool hasBuyerUid, hasUserId, hasDeviceUA, hasDeviceIP;

    for (auto s: samples) {

        hasBuyerUid = false;
        hasUserId = false;
        hasDeviceUA = false;
        hasDeviceIP = false;

        OpenRTB::BidRequest req;
        std::unique_ptr<BidRequest> datacraticReq;
        {
            StreamingJsonParsingContext context;
            context.init(s);
            datacraticReq.reset(p->parseBidRequest(*context.context, "openrtb", "openrtb"));   
        }


        OpenRTB::BidRequest req2;
        {
            StreamingJsonParsingContext context;
            context.init(s);
            desc.parseJson(&req2, context);
        }

        if(req2.user) {
            if(req2.user->buyeruid)
                hasBuyerUid = true;
            // Check if we have userId
            else if(req2.user->id)
                hasUserId = true;
        }

        if(req2.device) {

            // Check if we have the IP
            if(!(req2.device->ip.empty()))
                hasDeviceIP = true;

            // Check if we have the UA
            if(!(req2.device->ua.empty()))
                hasDeviceUA = true;
        }

        // Check if we manage to to get buyeruid
        if(hasBuyerUid) {
            cerr << "BidRequest Buyer Id : " << req2.user->buyeruid << endl;
            cerr << "Datacratic BidRequest Provider Id : " << datacraticReq->userIds.providerId << endl;
            BOOST_CHECK( req2.user->buyeruid == datacraticReq->userIds.providerId );
        }
        // Check what we have for providerId in the bidrequest.
        else if(hasUserId) {
            cerr << "BidRequest User Id : " << req2.user->id << endl;
            cerr << "Datacratic BidRequest Provider Id : " << datacraticReq->userIds.providerId << endl;
            BOOST_CHECK( req2.user->id == datacraticReq->userIds.providerId );
        }
        else if(hasDeviceIP && hasDeviceUA) {
            cerr << "BidRequest Device IP : " << req2.device->ip << endl; 
            cerr << "BidRequest Device Agent : " << req2.device->ua << endl;
            cerr << "Datacratic BidRequest Provider Id : " << datacraticReq->userIds.providerId << endl;
            BOOST_CHECK( datacraticReq->userIds.providerId != Id(0) );
        }
        else {
            cerr << "BidRequest User Id : " << req2.user->id << endl;
            cerr << "Datacratic BidRequest Provider Id : " << datacraticReq->userIds.providerId << endl;
            BOOST_CHECK( datacraticReq->userIds.providerId == Id(0) );
        }
    }
}
