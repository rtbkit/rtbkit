/* currency_js.h                                                   -*- C++ -*-
   Jeremy Barnes, 11 December 2012
   Copyright (c) 2012 Datacratic Inc.  All rights reserved.

*/

#pragma once

#include "bid_request_js.h"
#include "rtbkit/common/currency.h"
#include "soa/js/js_value.h"
#include "v8.h"


namespace Datacratic {
namespace JS {


/******************************************************************************/
/* AMOUNT                                                                     */
/******************************************************************************/

RTBKIT::Amount* from_js(const JSValue& value, RTBKIT::Amount**);
RTBKIT::Amount from_js(const JSValue& value, RTBKIT::Amount*);
void to_js(JSValue & value, const RTBKIT::Amount & amount);

/** Adds a various Amount constructors to the rtb module. */
void initCurrencyFunctions(v8::Handle<v8::Object>& target);


/******************************************************************************/
/* CURRENCY POOL                                                              */
/******************************************************************************/

// Wrappers for CurrencyPool are not implemented.
RTBKIT::CurrencyPool from_js(const JSValue & value, RTBKIT::CurrencyPool *);
void to_js(JSValue & value, const RTBKIT::CurrencyPool & c);


/******************************************************************************/
/* LINE ITEMS                                                                 */
/******************************************************************************/

// Wrappers for LineItems are not implemented.
RTBKIT::LineItems from_js(const JSValue & value, RTBKIT::LineItems *);
void to_js(JSValue & value, const RTBKIT::LineItems & l);

} // namespace JS
} // namespace RTB
