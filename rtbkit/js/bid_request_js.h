/* request_js.h                                                    -*- C++ -*-
   Jeremy Barnes, 6 April 2011
   Copyright (c) 2011 Datacratic.  All rights reserved.

   Javascript bindings for the Bid Request.
*/

#pragma once

#include "rtb_js.h"
#include "rtbkit/common/bid_request.h"

namespace Datacratic {
namespace JS {

using namespace RTBKIT;

extern const char * const bidRequestModule;

void to_js(JS::JSValue & value, const std::shared_ptr<BidRequest> & br);

std::shared_ptr<BidRequest>
getBidRequestSharedPointer(const JS::JSValue &);

std::shared_ptr<BidRequest>
from_js(const JSValue & value, std::shared_ptr<BidRequest> *);

std::shared_ptr<BidRequest>
from_js_ref(const JSValue & value, std::shared_ptr<BidRequest> *);

BidRequest *
from_js(const JSValue & value, BidRequest **);

std::shared_ptr<AdSpot>
from_js(const JSValue & value, std::shared_ptr<AdSpot> *);

AdSpot *
from_js(const JSValue & value, AdSpot **);


void to_js(JS::JSValue & value, const std::shared_ptr<SegmentList> & br);

std::shared_ptr<SegmentList>
from_js(const JSValue & value, std::shared_ptr<SegmentList> *);

std::shared_ptr<SegmentList>
from_js_ref(const JSValue & value, std::shared_ptr<SegmentList> *);

SegmentList *
from_js(const JSValue & value, SegmentList **);

SegmentList
from_js(const JSValue & value, SegmentList *);


void to_js(JS::JSValue & value, const UserIds & uids);

UserIds 
from_js(const JSValue & value, UserIds *);

void to_js(JS::JSValue & value, const Format & f);

} // namespace JS
} // namespace Datacratic
