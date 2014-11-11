/** creative_filters_test.cc                                 -*- C++ -*-
    Rémi Attab, 28 Aug 2013
    Copyright (c) 2013 Datacratic.  All rights reserved.

    Tests for the creative filters

*/

#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include "utils.h"
#include "rtbkit/core/router/filters/creative_filters.h"
#include "rtbkit/core/agent_configuration/agent_config.h"
#include "rtbkit/common/bid_request.h"
#include "rtbkit/common/exchange_connector.h"
#include "jml/utils/vector_utils.h"

#include <boost/test/unit_test.hpp>

using namespace std;
using namespace ML;
using namespace Datacratic;


/******************************************************************************/
/* FORMAT FILTER                                                              */
/******************************************************************************/

BOOST_AUTO_TEST_CASE( testFormatFilter )
{
    CreativeFormatFilter filter;
    CreativeMatrix creatives;

    AgentConfig c0;
    c0.creatives.push_back(Creative());
    c0.creatives.push_back(Creative(100, 100));
    c0.creatives.push_back(Creative(300, 300));

    AgentConfig c1;
    c1.creatives.push_back(Creative(100, 100));
    c1.creatives.push_back(Creative(200, 200));

    BidRequest r0;
    addImp(r0, OpenRTB::AdPosition::ABOVE, { {100, 100} });
    addImp(r0, OpenRTB::AdPosition::ABOVE, { {200, 200}, {100, 100} });
    addImp(r0, OpenRTB::AdPosition::ABOVE, { {200, 200}, {300, 300} });
    addImp(r0, OpenRTB::AdPosition::ABOVE, { {300, 300} });
    addImp(r0, OpenRTB::AdPosition::ABOVE, { {400, 400} });


    title("format-1");
    addConfig(filter, 0, c0, creatives);
    addConfig(filter, 1, c1, creatives);

    check(filter, r0, creatives, 0, { {0, 1}, {0}        });
    check(filter, r0, creatives, 1, { {0, 1}, {0, 1}     });
    check(filter, r0, creatives, 2, { {0},    {1},   {0} });
    check(filter, r0, creatives, 3, { {0},    {},    {0} });
    check(filter, r0, creatives, 4, { {0}                });


    title("format-2");
    removeConfig(filter, 0, c0, creatives);

    check(filter, r0, creatives, 0, { {1},     });
    check(filter, r0, creatives, 1, { {1}, {1} });
    check(filter, r0, creatives, 2, { {},  {1} });
    check(filter, r0, creatives, 3, {          });
    check(filter, r0, creatives, 4, {          });
}


/******************************************************************************/
/* LANGUAGE FILTER                                                            */
/******************************************************************************/

BOOST_AUTO_TEST_CASE( testLanguageFilter )
{
    CreativeLanguageFilter filter;
    CreativeMatrix creatives;

    auto addCr = [] (AgentConfig& cfg, const IncludeExclude<string>& ie) {
        cfg.creatives.emplace_back();
        cfg.creatives.back().languageFilter = std::move(ie);
    };

    AgentConfig c0;
    addCr(c0, ie<string>({"fr"}, {}));
    addCr(c0, ie<string>({}, {"en"}));

    AgentConfig c1;
    addCr(c1, ie<string>({"en"}, {}));
    addCr(c1, ie<string>({"en"}, {"fr"}));

    BidRequest r0;
    r0.language = "en";
    addImp(r0, OpenRTB::AdPosition::ABOVE, { {100, 100} });
    addImp(r0, OpenRTB::AdPosition::ABOVE, { {200, 200}, {100, 100} });

    BidRequest r1;
    r1.language = "fr";
    addImp(r1, OpenRTB::AdPosition::ABOVE, { {100, 100} });

    BidRequest r2;
    r2.language = "es";
    addImp(r2, OpenRTB::AdPosition::ABOVE, { {100, 100} });


    title("language-1");
    addConfig(filter, 0, c0, creatives);
    addConfig(filter, 1, c1, creatives);

    check(filter, r0, creatives, 0, { {1}, {1} });
    check(filter, r0, creatives, 1, { {1}, {1} });
    check(filter, r1, creatives, 0, { {0}, {0} });
    title("language-1-d");
    check(filter, r2, creatives, 0, { {},  {0} });


    title("language-2");
    removeConfig(filter, 0, c0, creatives);

    check(filter, r0, creatives, 0, { {1}, {1} });
    check(filter, r0, creatives, 1, { {1}, {1} });
    check(filter, r1, creatives, 0, {          });
    check(filter, r2, creatives, 0, {          });
}


/******************************************************************************/
/* Segment FILTER                                                             */
/******************************************************************************/
BOOST_AUTO_TEST_CASE( testSegmentsFilter )
{
    CreativeSegmentsFilter filter;
    CreativeMatrix creatives;

    typedef Creative::SegmentInfo SegInfo;

    auto addCrfilter = [] (AgentConfig& cfg,
                           const std::vector<std::string> & inc,
                           const std::vector<std::string> & exc,
                           bool excludeIfNotPresent = true) {
        SegInfo si;
        for ( auto & w : inc) si.include.add(w);
        for ( auto & w : exc) si.exclude.add(w);
        si.excludeIfNotPresent = excludeIfNotPresent;
        cfg.creatives.emplace_back();
        cfg.creatives.back().segments["words"] = si;
    };

    // Ad those filters to some creatives in some agentconfigs
    AgentConfig c0;
    addCrfilter(c0, {"hey"}, {});
    addCrfilter(c0, {}, {"oh"});
    addCrfilter(c0, {"oh"}, {});

    AgentConfig c1;
    addCrfilter(c1, {"go"}, {}, false);

    AgentConfig c2;
    addCrfilter(c2, {"oh"}, {});
    addCrfilter(c2, {}, {"no"}, false);
    addCrfilter(c2, {"go"}, {});

    // Create some BR with some segments to be filtered
    BidRequest br0;
    addImp(br0, OpenRTB::AdPosition::ABOVE, { {100, 100} });
    addImp(br0, OpenRTB::AdPosition::ABOVE, { {100, 200} });
    br0.segments.addStrings("words", {"hey"});

    BidRequest br1;
    addImp(br1, OpenRTB::AdPosition::ABOVE, { {100, 100} });
    br1.segments.addStrings("words", {"oh", "go"});

    BidRequest br2;
    addImp(br2, OpenRTB::AdPosition::ABOVE, { {100, 100} });
    br2.segments.addStrings("words", {"lets"});

    BidRequest br3;
    addImp(br3, OpenRTB::AdPosition::ABOVE, { {100, 100} });

    title("segemnts-filter-include-exclude-1");
    addConfig(filter, 0, c0, creatives);

    check(filter, br0, creatives, 0, { {0}, {0}        });
    check(filter, br0, creatives, 1, { {0}, {0}        });
    check(filter, br1, creatives, 0, { { }, { }, {0}   });
    check(filter, br2, creatives, 0, { { }, {0},       });

    title("segemnts-filter-no-segment-1");
    check(filter, br3, creatives, 0, {                 });

    title("segemnts-filter-include-exclude-2");
    addConfig(filter, 1, c1, creatives);

    check(filter, br0, creatives, 0, { {0}, {0}         });
    check(filter, br0, creatives, 1, { {0}, {0}         });
    check(filter, br1, creatives, 0, { {1}, { }, {0}    });
    check(filter, br2, creatives, 0, { { }, {0},        });

    title("segemnts-filter-no-segment-2");
    check(filter, br3, creatives, 0, { {1}              });

    title("segemnts-filter-include-exclude-3");
    addConfig(filter, 2, c2, creatives);

    check(filter, br0, creatives, 0, { {0}, {0,2},      });
    check(filter, br0, creatives, 1, { {0}, {0,2},      });
    check(filter, br1, creatives, 0, { {1,2}, {2}, {0,2}});
    check(filter, br2, creatives, 0, { { }, {0,2},      });

    title("segemnts-filter-no-segment-3");
    check(filter, br3, creatives, 0, { {1}, {2}         });

}
