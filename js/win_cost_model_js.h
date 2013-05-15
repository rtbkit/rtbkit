/** win_cost_model_js.h                                 -*- C++ -*-
    Eric Robert, 15 May 2013
    Copyright (c) 2013 Datacratic.  All rights reserved.

    JS Wrappers for the win cost model class.

*/

#pragma once

#include "rtb_js.h"
#include "rtbkit/common/win_cost_model.h"
#include "soa/js/js_value.h"
#include "v8.h"


namespace Datacratic {
namespace JS {


/******************************************************************************/
/* WIN COST MODEL                                                             */
/******************************************************************************/

RTBKIT::WinCostModel* from_js(const JSValue& value, RTBKIT::WinCostModel**);
RTBKIT::WinCostModel from_js(const JSValue& value, RTBKIT::WinCostModel*);
void to_js(JSValue & value, const RTBKIT::WinCostModel & wcm);


} // namespace JS
} // namespace Datacratic
