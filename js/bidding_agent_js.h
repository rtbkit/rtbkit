/* bidding_agent_js.h                                                   -*- C++ -*-
   Rémi Attab, 14 December 2011
   Copyright (c) 2011 Datacratic.  All rights reserved.

   Provides messaging services between the router and the client.
*/


#ifndef __rtb__bidding_agent_js_h__
#define __rtb__bidding_agent_js_h__

#include "rtb_js.h"
#include "rtbkit/plugins/bidding_agent/bidding_agent.h"
#include "soa/js/js_wrapped.h"
#include "jml/utils/smart_ptr_utils.h"

namespace Datacratic {
namespace JS {

std::shared_ptr<RTBKIT::BiddingAgent>
from_js(const JSValue & value, std::shared_ptr<RTBKIT::BiddingAgent> *);

RTBKIT::BiddingAgent *
from_js(const JSValue & value, RTBKIT::BiddingAgent **);

} // namespace JS
} // namespace Datacratic

#endif // __rtb__bidding_agent_js_h__

