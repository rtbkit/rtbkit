/* openrtb_bid_request.h                                           -*- C++ -*-
   Jeremy Barnes, 19 February 2013
   Copyright (c) 2013 Datacratic Inc.  All rights reserved.

   Bid request support for openRTB.
*/

#pragma once


#include "rtbkit/common/bid_request.h"
#include "jml/utils/parse_context.h"

namespace RTBKIT {

BidRequest *
fromOpenRtb(OpenRTB::BidRequest && req,
            const std::string & provider,
            const std::string & exchange);


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
                    const std::string & exchange = "");

    static OpenRTB::BidRequest
    parseBidRequest(ML::Parse_Context & context);

    static BidRequest *
    parseBidRequest(ML::Parse_Context & context,
                    const std::string & provider,
                    const std::string & exchange = "");
        
};


} // namespace RTBKIT
