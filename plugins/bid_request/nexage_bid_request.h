/* nexage_bid_request.h                                           -*- C++ -*-
   
   Bid request support for Nexage.
*/

#pragma once


#include "rtbkit/common/bid_request.h"
#include "nexage_parsing.h"
#include "jml/utils/parse_context.h"

namespace RTBKIT {

BidRequest *
fromNexage(Nexage::BidRequest && req,
            const std::string & provider,
            const std::string & exchange);


/*****************************************************************************/
/* NEXAGE BID REQUEST PARSER                                                */
/*****************************************************************************/

/** Parser for the OpenRTB bid request format.
 */

struct NexageBidRequestParser {

    static BidRequest *
    parseBidRequest(const std::string & jsonValue,
                    const std::string & provider,
                    const std::string & exchange = "");

    static BidRequest *
    parseBidRequest(ML::Parse_Context & context,
                    const std::string & provider,
                    const std::string & exchange = "");
        
};


} // namespace RTBKIT
