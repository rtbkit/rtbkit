/* banker_js.h                                                     -*- C++ -*-
   Jeremy Barnes, 13 January 2012
   Copyright (c) 2012 Datacratic.  All rights reserved.

   JS interface to the Banker class.
*/

#ifndef __rtb__banker_js_h__
#define __rtb__banker_js_h__

#include "rtb_js.h"
#include "rtbkit/core/banker/banker.h"

namespace Datacratic {
namespace JS {

std::shared_ptr<RTBKIT::Banker>
from_js(const JSValue & value, std::shared_ptr<RTBKIT::Banker> *);

RTBKIT::Banker *
from_js(const JSValue & value, RTBKIT::Banker **);

void to_js(JSValue & value, const std::shared_ptr<RTBKIT::Banker> & banker);


} // namespace JS
} // namespace Datacratic


#endif /* __rtb__banker_js_h__ */
