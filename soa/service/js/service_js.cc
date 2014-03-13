/** service_js.cc                                 -*- C++ -*-
    Rémi Attab, 20 Jul 2012
    Copyright (c) 2012 Datacratic.  All rights reserved.

    Service JS module implementation.

*/


#include "v8.h"
#include "service_js.h"
#include "soa/js/js_registry.h"

using namespace std;
using namespace v8;


namespace Datacratic {
namespace JS {

const char * const serviceModule = "services";

// Node.js initialization function; called to set up the service object
extern "C" void
init(Handle<v8::Object> target)
{
    Datacratic::JS::registry.init(target, serviceModule);
}


} // namepsace JS
} // namespace Datacratic
