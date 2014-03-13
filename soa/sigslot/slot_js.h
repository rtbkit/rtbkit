/* slot_js.h                                                       -*- C++ -*-
   Jeremy Barnes, 26 November 2010
   Copyright (c) 2010 Datacratic.  All rights reserved.

   JS wrapper for the slot class.
*/

#pragma once

#include "soa/sigslot/slot.h"
#include "soa/js/js_value.h"

namespace Datacratic {
namespace JS {

extern const char * const sigslotModule;

Slot * from_js(const JSValue & val, Slot ** = 0);
std::shared_ptr<Slot>
from_js(const JSValue & val, std::shared_ptr<Slot>* = 0);

void to_js(JS::JSValue & value, const std::shared_ptr<Slot> &);
void to_js(JS::JSValue & value, const Slot &);


} // namespace JS
} // namespace Datacratic
