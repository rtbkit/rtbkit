/* openrtb_bid_request.h                                           -*- C++ -*-
   Jeremy Barnes, 19 February 2013
   Copyright (c) 2013 Datacratic Inc.  All rights reserved.

   Bid request support for openRTB.
*/

#pragma once

#include "rtbkit/common/bid_request.h"
#include "jml/utils/parse_context.h"

namespace RTBKIT {
    
/** Converts an OpenRTB BidRequest to RTBKIT internal BidRequest format
 */
BidRequest *
fromOpenRtb(OpenRTB::BidRequest && req,
            const std::string & provider,
            const std::string & exchange = "",
            const std::string & version = "2.1");


/** Converts RTBKIT BidRequest to OpenRTB BidRequest
 */
OpenRTB::BidRequest toOpenRtb(const BidRequest &req);

/*****************************************************************************/
/* OPENRTB BID REQUEST PARSER                                                */
/*****************************************************************************/

/** Parser for the OpenRTB bid request format.
 */

struct OpenRtbBidRequestParser {

    static OpenRTB::BidRequest
    parseBidRequest(const std::string & jsonValue);

    static BidRequest *
    parseBidRequest(const std::string & jsonValue,
                    const std::string & provider,
                    const std::string & exchange = "",
                    const std::string & version = "2.1");

    static OpenRTB::BidRequest
    parseBidRequest(ML::Parse_Context & context);

    static BidRequest *
    parseBidRequest(ML::Parse_Context & context,
                    const std::string & provider,
                    const std::string & exchange = "",
                    const std::string & version = "2.1");
        
};
} // namespace RTBKIT
