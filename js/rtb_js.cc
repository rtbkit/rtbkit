/* rtb_js.cc
   26 November 2010
   Copyright (c) 2010 Datacratic.  All rights reserved.

*/

#include "v8.h"
#include "rtb_js.h"
#include "soa/js/js_registry.h"

using namespace v8;
using namespace std;

namespace Datacratic {
namespace JS {

const char * const rtbModule = "rtb";

// Node.js initialization function; called to set up the RTB object
extern "C" void
init(Handle<v8::Object> target)
{
    Datacratic::JS::registry.init(target, rtbModule);
}

} // namespace JS
} // namespace RTB
