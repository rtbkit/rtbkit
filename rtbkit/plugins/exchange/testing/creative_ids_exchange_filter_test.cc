/* creative_ids_exchange_filter_test.cc
   Mathieu Stefani, 02 March 2015
   Copyright (c) 2015 Datacratic.  All rights reserved.
   
   Tests for the CreativeIdsExchangeFilter
*/

#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include "rtbkit/plugins/exchange/rtbkit_exchange_connector.h"
#include "rtbkit/core/router/filters/testing/utils.h"
#include "rtbkit/core/router/filters/creative_filters.h"
#include "rtbkit/core/agent_configuration/agent_config.h"
#include "rtbkit/common/bid_request.h"
#include "rtbkit/common/exchange_connector.h"

using namespace std;
using namespace ML;
using namespace Datacratic;
using namespace RTBKIT;

/******************************************************************************/
/* EXTERNAL-IDS FILTER                                                        */
/******************************************************************************/

BOOST_AUTO_TEST_CASE( test_external_ids_filter )
{
    CreativeIdsExchangeFilter filter;
    CreativeMatrix creatives;

    // AgentConfig
    AgentConfig c0;
    c0.creatives.push_back(Creative());
    c0.externalId = 1200;

    AgentConfig c1;
    c1.creatives.push_back(Creative());
    c1.externalId = 1201;

    title("external-ids-1");
    addConfig(filter, 0, c0, creatives);
    addConfig(filter, 1, c1, creatives);

    // BidRequest
    BidRequest r0;
    std::string ext0 = R"({"external-ids": [1201] })";
    addImp(r0, OpenRTB::AdPosition::ABOVE, ext0);
    std::string ext0B = R"({"external-ids": [1200] })";
    addImp(r0, OpenRTB::AdPosition::ABOVE, ext0B);

    BidRequest r1;
    std::string ext1 = R"({"external-ids": [1200] })";
    addImp(r1, OpenRTB::AdPosition::ABOVE, ext1);

    BidRequest r2;
    std::string ext2 = R"({"external-ids": [1200, 1201] })";
    addImp(r2, OpenRTB::AdPosition::ABOVE, ext2);

    BidRequest r3;
    std::string ext3 = R"({"external-ids": [1204, 1201] })";
    addImp(r3, OpenRTB::AdPosition::ABOVE, ext3);

    BidRequest r4;
    addImp(r4, OpenRTB::AdPosition::ABOVE, { {100, 100} });
    
    // Check
    check(filter, r0, creatives, 0, { { 1 } });
    check(filter, r0, creatives, 1, { { 0 } });
    check(filter, r1, creatives, 0, { { 0 } });
    check(filter, r2, creatives, 0, { { 0,1} });
    check(filter, r3, creatives, 0, { { 1} });
    check(filter, r4, creatives, 0, { { } });
}


/******************************************************************************/
/* CREATIVE-IDS FILTER                                                        */
/******************************************************************************/

BOOST_AUTO_TEST_CASE( test_creative_ids_filter )
{
    CreativeIdsExchangeFilter filter;
    CreativeMatrix creatives;

    // AgentConfig
    AgentConfig c0;
    Creative cr0;
    cr0.id = 891;
    c0.creatives.push_back(cr0);
    c0.externalId = 123;

    AgentConfig c1;
    Creative cr1;
    cr1.id = 7812;
    Creative cr2;
    cr2.id = 6751;

    c1.creatives.push_back(cr1);
    c1.creatives.push_back(cr2); 
    c1.externalId = 321;

    title("creative-ids-1");
    addConfig(filter, 0, c0, creatives);
    addConfig(filter, 1, c1, creatives);

    // BidRequest
    BidRequest r0;
    std::string ext0 = R"({"creative-ids": { "123": [ 891 ] } })";
    addImp(r0, OpenRTB::AdPosition::ABOVE, ext0);
    std::string ext0B = R"({"creative-ids": { "321": [ 6751 ] } })";
    addImp(r0, OpenRTB::AdPosition::ABOVE, ext0B);

    BidRequest r1;
    std::string ext1 = R"({"creative-ids": { "123": [891] } })";
    addImp(r1, OpenRTB::AdPosition::ABOVE, ext1);

    BidRequest r2;
    std::string ext2 = R"({"creative-ids": { "123": [ 891 ], "321": [ 7812 ] } })";
    addImp(r2, OpenRTB::AdPosition::ABOVE, ext2);

    BidRequest r3;
    std::string ext3 = R"({"creative-ids": { "123": [ 761 ], "321": [ 6751, 7812 ] } })";
    addImp(r3, OpenRTB::AdPosition::ABOVE, ext3);

    BidRequest r4;
    addImp(r4, OpenRTB::AdPosition::ABOVE, { {100, 100} });

    // Check
    check(filter, r0, creatives, 0, { { 0 }, { } });
    check(filter, r0, creatives, 1, { { }, { 1 } });

    check(filter, r1, creatives, 0, { { 0 }, { } });
    check(filter, r2, creatives, 0, { { 0, 1 } } );
    check(filter, r3, creatives, 0, { { 1 }, { 1 } });
    check(filter, r4, creatives, 0, { { } });
}
