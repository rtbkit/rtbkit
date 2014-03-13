/* appnexus_bid_request.h                                           -*- C++ -*-
   Mark Weiss, 28 March 2013
   Copyright (c) 2013 Datacratic Inc.  All rights reserved.

   Bid request support for AppNexus.
*/

#pragma once

#include <memory>
#include "rtbkit/common/bid_request.h"
#include "appnexus_parsing.h"
#include "jml/utils/parse_context.h"

namespace RTBKIT {

std::shared_ptr<BidRequest>
fromAppNexus(const AppNexus::BidRequest & req,
            const std::string & provider,
            const std::string & exchange);

/*****************************************************************************/
/* APPNEXUS BID REQUEST PARSER                                                */
/*****************************************************************************/

/** Parser for the AppNexus bid request format.
 */

struct AppNexusBidRequestParser {

    static std::shared_ptr<BidRequest>
    parseBidRequest(const std::string & jsonValue,
                    const std::string & provider,
                    const std::string & exchange = "");

    static std::shared_ptr<BidRequest>
    parseBidRequest(ML::Parse_Context & context,
                    const std::string & provider,
                    const std::string & exchange = "");
};


} // namespace RTBKIT
