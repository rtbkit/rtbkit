/** opstats_js.h                                 -*- C++ -*-
    Rémi Attab, 19 Jul 2012
    Copyright (c) 2012 Datacratic.  All rights reserved.

    Blah

*/

#ifndef __logger__opstats_js_h__
#define __logger__opstats_js_h__

namespace Datacratic {
namespace JS {


std::shared_ptr<CarbonConnector>
from_js(const JSValue & value, std::shared_ptr<CarbonConnector> *);

CarbonConnector *
from_js(const JSValue & value, CarbonConnector **);

std::shared_ptr<CarbonConnector>
from_js_ref(const JSValue& value, std::shared_ptr<CarbonConnector>*);

void
to_js(JS::JSValue& value, const std::shared_ptr<CarbonConnector>& proxy);


} // namespace JS
} // Datacratic

#endif // __logger__opstats_js_h__
