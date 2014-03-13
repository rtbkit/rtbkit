/* fbx_bid_request.h                                           -*- C++ -*-
   Jean-Sebastien Bejeau, 19 June 2013

   Bid request support for FBX.
*/

#pragma once


#include "rtbkit/common/bid_request.h"
#include "fbx.h"
#include "jml/utils/parse_context.h"

namespace RTBKIT {

BidRequest *
fromFbx(FBX::BidRequest && req,
            const std::string & provider,
            const std::string & exchange);


/*****************************************************************************/
/* FBX BID REQUEST PARSER                                                */
/*****************************************************************************/

/** Parser for the FBX bid request format.
 */

struct FbxBidRequestParser {

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
