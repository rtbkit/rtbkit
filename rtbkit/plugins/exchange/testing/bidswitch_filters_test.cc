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

    // AgentConfigs
    auto doAgentConfigInit = [&] (AgentConfig & ac, const std::string & providerConfigStr) {
        ac.creatives.push_back(Creative());
        ac.providerConfig = Json::parse(providerConfigStr);
    };

    AgentConfig c0, c1, c2;
    doAgentConfigInit(c0, "{ \"bidswitch\": { \"seat\": \"123\" }}");
    doAgentConfigInit(c1, "{ \"bidswitch\": { \"seat\": \"1\" }}");
    doAgentConfigInit(c2, "{}");

    auto doBidRequestInit = [&] (BidRequest & br, const std::vector<string> & wseat) {
        // add at least one Impression
        addImp(br, OpenRTB::AdPosition::ABOVE, "{\"flight_ids\": [1200] }");
        br.segments.addStrings("openrtb-wseat", wseat );

    };

    // BidRequests 
    BidRequest r0, r1, r2, r3;
    doBidRequestInit(r0, { "aaa",  "1" } );
    doBidRequestInit(r1, { "123", "aaa" } );
    doBidRequestInit(r2, { "bbb", "aaa" });
    doBidRequestInit(r3, { "1", "123" });

    
    // Test 1
    title("Bidswtich-wseat-1");
    addConfig(filter, 0, c0, creatives);
    addConfig(filter, 1, c1, creatives);

    check(filter, r0, creatives, 0, { {1}  });
    check(filter, r1, creatives, 0, { {0}  });
    check(filter, r2, creatives, 0, { {}  });
    check(filter, r3, creatives, 0, { {0,1}  });

    removeConfig(filter, 0, c0, creatives);
    removeConfig(filter, 1, c1, creatives);

    // Test 2
    title("Bidswtich-wseat-2");
    addConfig(filter, 0, c2, creatives);
    check(filter, r0, creatives, 0, { {0}  });
    check(filter, r1, creatives, 0, { {0}  });
    check(filter, r2, creatives, 0, { {0}  });
    check(filter, r3, creatives, 0, { {0}  });

}
