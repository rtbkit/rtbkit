/** bidswitch_filters_test.cc                                 -*- C++ -*-
    JS Bejeau, 10 Jul 2014
    Copyright (c) 2013 Datacratic.  All rights reserved.

    Tests for the bidswitch filters.

    Note that these tests assume that the generic filters work properly so we
    don't waste time constructing complicated set of include/exclude statements.
 */

#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include "rtbkit/core/router/filters/testing/utils.h"
#include "rtbkit/core/router/filters/static_filters.h"
#include "rtbkit/core/agent_configuration/agent_config.h"
#include "rtbkit/common/bid_request.h"
#include "rtbkit/common/exchange_connector.h"
#include "jml/utils/vector_utils.h"

#include "rtbkit/plugins/exchange/bidswitch_exchange_connector.h"

#include <boost/test/unit_test.hpp>

using namespace std;
using namespace ML;
using namespace Datacratic;


/** Simple test to check that all a config will fail if one of its segment
    fails. Nothing to interesting really.
 */
BOOST_AUTO_TEST_CASE( bidswitch_wseat_filter )
{
    BidSwitchWSeatFilter filter; 
    CreativeMatrix creatives;

    AgentConfig c0;
    c0.creatives.push_back(Creative());
    std::string providerConfigStr = "{ \"bidswitch\": { \"seat\": \"123\" }}";
    c0.providerConfig = Json::parse(providerConfigStr);

    AgentConfig c1;
    c1.creatives.push_back(Creative());
    std::string providerConfigStr1 = "{ \"bidswitch\": { \"seat\": \"1\" }}";
    c1.providerConfig = Json::parse(providerConfigStr1);
    
    BidRequest r0;
    std::string ext1 = "{\"flight_ids\": [1200] }";
    addImp(r0, OpenRTB::AdPosition::ABOVE, ext1);
    r0.segments.addStrings("openrtb-wseat", {"aaa",  "1" } );

    
    BidRequest r1;
    addImp(r1, OpenRTB::AdPosition::ABOVE, ext1);
    r1.segments.addStrings("openrtb-wseat", { "123", "aaa" } );
    
    BidRequest r2;
    addImp(r2, OpenRTB::AdPosition::ABOVE, ext1);
    r2.segments.addStrings("openrtb-wseat", { "bbb", "aaa" } );
    
    BidRequest r3;
    addImp(r3, OpenRTB::AdPosition::ABOVE, ext1);
    r3.segments.addStrings("openrtb-wseat", { "1", "123" } );
    
    title("Bidswtich-wseat-1");
    addConfig(filter, 0, c0, creatives);
    addConfig(filter, 1, c1, creatives);


    check(filter, r0, creatives, 0, { {1}  });
    check(filter, r1, creatives, 0, { {0}  });
    check(filter, r2, creatives, 0, { {}  });
    check(filter, r3, creatives, 0, { {0,1}  });


    //TODO : Add AgentConfig with the same seat as AgentCong 0



}
