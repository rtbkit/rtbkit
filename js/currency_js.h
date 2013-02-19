/* currency_js.h                                                   -*- C++ -*-
   Jeremy Barnes, 11 December 2012
   Copyright (c) 2012 Datacratic Inc.  All rights reserved.

*/

#pragma once

#include "rtb_js.h"
#include "rtbkit/common/currency.h"


namespace Datacratic {
namespace JS {

Amount from_js(const JSValue & value, Amount *);
void to_js(JSValue & value, const Amount & amount);

CurrencyPool from_js(const JSValue & value, CurrencyPool *);
void to_js(JSValue & value, const CurrencyPool & c);

LineItems from_js(const JSValue & value, LineItems *);
void to_js(JSValue & value, const LineItems & l);

} // namespace JS
} // namespace RTB
