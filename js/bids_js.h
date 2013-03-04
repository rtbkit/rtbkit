/** bids_js.h                                 -*- C++ -*-
    Rémi Attab, 01 Mar 2013
    Copyright (c) 2013 Datacratic.  All rights reserved.

    JS Wrappers for the Bids class.

*/

#pragma once

#include "rtb_js.h"
#include "rtbkit/common/bids.h"
#include "soa/js/js_value.h"
#include "v8.h"


namespace Datacratic {
namespace JS {


/******************************************************************************/
/* BID                                                                        */
/******************************************************************************/

RTBKIT::Bid* from_js(const JSValue& value, RTBKIT::Bid**);
RTBKIT::Bid from_js(const JSValue& value, RTBKIT::Bid*);
void to_js(JSValue & value, const RTBKIT::Bid & amount);


/******************************************************************************/
/* BIDS                                                                       */
/******************************************************************************/

RTBKIT::Bids* from_js(const JSValue& value, RTBKIT::Bids**);
RTBKIT::Bids from_js(const JSValue& value, RTBKIT::Bids*);
void to_js(JSValue & value, const RTBKIT::Bids & amount);


} // namespace JS
} // namespace Datacratic
