/* bid_request_parser.h                                           -*- C++ -*-
   Jean-Michel Bouchard, 19 August 2014
   Copyright (c) 2014 Datacratic Inc.  All rights reserved.

   Bid request parser interface
*/

#pragma once

namespace RTBKIT {
    
/*****************************************************************************/
/* BID REQUEST PARSER                                                        */
/*****************************************************************************/

template <typename T, typename V>
struct IBidRequestParser {
    virtual T parseBidRequest(V & value);
};

} // namespace RTBKIT
