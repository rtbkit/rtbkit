/* auction_js.h                                                    -*- C++ -*-
   Jeremy Barnes, 6 April 2011
   Copyright (c) 2011 Datacratic.  All rights reserved.

   Javascript bindings for the Auction.
*/

#pragma once

#include "rtb_js.h"
#include "rtbkit/common/auction.h"

namespace Datacratic {
namespace JS {

std::shared_ptr<RTBKIT::Auction>
from_js(const JSValue & value, std::shared_ptr<RTBKIT::Auction> *);

RTBKIT::Auction *
from_js(const JSValue & value, RTBKIT::Auction **);


} // namespace JS
} // namespace Datacratic
