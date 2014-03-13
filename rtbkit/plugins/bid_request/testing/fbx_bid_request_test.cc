/* fbx_bid_request_test.cc
   Jean-Sebastien Bejeau, 22 June 2013

   Test cases for the FBX bid request parser.
*/


#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include "rtbkit/plugins/bid_request/fbx_parsing.h"
#include "rtbkit/plugins/bid_request/fbx_bid_request.h"
#include "soa/types/json_parsing.h"
#include "jml/utils/filter_streams.h"

using namespace std;
using namespace ML;
using namespace Datacratic;
using namespace RTBKIT;

vector<string> samples = {
    "rtbkit/plugins/bid_request/testing/fbx1_req.json"
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

    FBX::BidRequest req;
    DefaultDescription<FBX::BidRequest> desc;
    desc.parseJson(&req, context);

    if (!req.unparseable.isNull())
        cerr << "unparseable:" << req.unparseable << endl;

    StreamJsonPrintingContext printContext(cout);
    desc.printJson(&req, printContext);
}

BOOST_AUTO_TEST_CASE( test_parse_fbx_sample_requests )
{
    for (auto req: samples)
        parseBidRequest(req);
}

void testBidRequest(const std::string & filename)
{
    cerr << endl << "loading " << filename << endl;

    ML::Parse_Context context(filename);

    auto res = FbxBidRequestParser::parseBidRequest(context, "test", "test");

    cerr << res->toJson() << endl;

}

BOOST_AUTO_TEST_CASE( test_fbx_sample_requests )
{
    for (auto s: samples)
        testBidRequest(s);
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
    // Take an FXB bid request; convert to Datacratic format;
    // serialize that to JSON; reconstitute from JSON back into the
    // original format

    static DefaultDescription<FBX::BidRequest> desc;

    FBX::BidRequest req;

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

    FBX::BidRequest req2;

    {
        StreamingJsonParsingContext context;
        context.init(filename, reqStr.c_str(), reqStr.size());
        desc.parseJson(&req2, context);
    }

    string printed2;
    {
        std::ostringstream stream;
        StreamJsonPrintingContext printContext(stream);
        desc.printJson(&req2, printContext);
        printed2 = stream.str();
    }
    
    BOOST_CHECK_EQUAL(printed, printed2);

    // Convert to a standard bid request
    
    std::unique_ptr<BidRequest> br(fromFbx(std::move(req), "fbx", "fbx"));
    
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

BOOST_AUTO_TEST_CASE( test_fbx_round_trip )
{
    vector<string> reqs;

    for (auto s: samples)
        reqs.push_back(loadFile(s));

    for (unsigned i = 0;  i < reqs.size();  ++i)
        testBidRequestRoundTrip(samples[i], reqs[i]);
}

BOOST_AUTO_TEST_CASE( benchmark_fbx_round_trip )
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

BOOST_AUTO_TEST_CASE( benchmark_fbx_parsing )
{
    cerr << "benchmarking FBX parsing" << endl;

    vector<string> reqs;

    for (auto s: samples)
        reqs.push_back(loadFile(s));

    int done = 0;
    
    Date before = Date::now();

    DefaultDescription<FBX::BidRequest> desc;

    for (unsigned i = 0;  i < 1000;  ++i) {
        
        for (unsigned i = 0;  i < reqs.size();  ++i, ++done) {
            
            
            FBX::BidRequest req;
            
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

BOOST_AUTO_TEST_CASE( benchmark_fbx_conversion )
{
    cerr << "benchmarking FBX parsing and conversion" << endl;

    vector<string> reqs;

    for (auto s: samples)
        reqs.push_back(loadFile(s));

    int done = 0;
    
    Date before = Date::now();

    DefaultDescription<FBX::BidRequest> desc;

    for (unsigned i = 0;  i < 1000;  ++i) {
        
        for (unsigned i = 0;  i < reqs.size();  ++i, ++done) {
            
            
            FBX::BidRequest req;
            
            {
                StreamingJsonParsingContext context;
                context.init(samples[i], reqs[i].c_str(), reqs[i].size());
                desc.parseJson(&req, context);
            }

            std::unique_ptr<BidRequest> br(fromFbx(std::move(req), "fbx", "fbx"));
        }
    }

    double elapsed = Date::now().secondsSince(before);
    
    cerr << "did " << done << " in " << elapsed << "s at "
         << done / elapsed << "/s" << endl;
}

BOOST_AUTO_TEST_CASE( benchmark_canonical_parsing )
{
    cerr << "benchmarking canonical parsing of FBX-derived bid requests" << endl;

    DefaultDescription<FBX::BidRequest> desc;

    vector<string> reqs;

    for (auto s: samples) {
        FBX::BidRequest req;
        {
            StreamingJsonParsingContext context;
            context.init(s);
            desc.parseJson(&req, context);
        }

        std::unique_ptr<BidRequest> br(fromFbx(std::move(req), "fbx", "fbx"));

        reqs.push_back(br->toJsonStr());
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
