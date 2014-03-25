/* historical_bid_request_test.cc
   Jeremy Barnes, 19 February 2013
   Copyright (c) 2013 Datacratic Inc.  All rights reserved.

   Test cases for the Historical bid request parser.
*/


#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include "rtbkit/common/bid_request.h"
#include "soa/types/json_parsing.h"
#include "jml/utils/filter_streams.h"

using namespace std;
using namespace ML;
using namespace Datacratic;
using namespace RTBKIT;

string bidRequest = "{\"!!CV\":\"0.1\",\"exchange\":\"abcd\",\"id\":\"8b703eb9-5ebf-4001-15e8-10d6000003a0\",\"ipAddress\":\"76.aa.xx.yy\",\"language\":\"en\",\"location\":{\"cityName\":\"Grande Prairie\",\"countryCode\":\"CA\",\"dma\":0,\"postalCode\":\"0\",\"regionCode\":\"AB\",\"timezoneOffsetMinutes\":240},\"protocolVersion\":\"0.3\",\"provider\":\"xxx1\",\"segments\":{\"xxx1\":null},\"imp\":[{\"formats\":[\"160x600\"],\"id\":\"22202919\",\"position\":\"NONE\",\"reservePrice\":0}],\"timestamp\":1336313462.550589,\"url\":\"http://emedtv.com/search.html?searchString=\xe2\x80\xa2skin\",\"userAgent\":\"Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; WOW64; Trident/5.0)\",\"userIds\":{\"ag\":\"7d837be6-94de-11e1-841f-68b599c88614\",\"abcd\":\"PV08FEPS1KQAAGcrbUsAAABY\",\"prov\":\"7d837be6-94de-11e1-841f-68b599c88614\",\"xchg\":\"PV08FEPS1KQAAGcrbUsAAABY\"}}";

BOOST_AUTO_TEST_CASE( test_parse_historical_bid_request )
{
    std::unique_ptr<BidRequest> br(BidRequest::parse("rtbkit", bidRequest));

    BOOST_CHECK_EQUAL(br->url.toString(), "http://emedtv.com/search.html?searchString=%E2%80%A2skin");
    BOOST_CHECK_EQUAL(br->userAgent, UnicodeString("Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; WOW64; Trident/5.0)"));
    
}

