/* slot_node.cc
   Jeremy Barnes, 6 February 2011
   Copyright (c) 2011 Datacratic.  All rights reserved.

   Node.js slot module.
*/

#include "slot_js.h"
#include "soa/js/js_utils.h"

using namespace Datacratic;
using namespace Datacratic::JS;

// Node.js initialization function; called to set up the Sigslot object
extern "C" void
init(v8::Handle<v8::Object> target)
{
    registry.init(target, sigslotModule);
}
