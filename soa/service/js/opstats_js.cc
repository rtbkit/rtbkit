/* opstats_js.cc
   Jeremy Barnes, 5 August 2011
   Copyright (c) 2011 Datacratic.  All rights reserved.

   Operational stats JS wrapping
*/

#include "soa/service/carbon_connector.h"
#include "v8.h"
#include "node.h"
#include "soa/js/js_value.h"
#include "soa/js/js_utils.h"
#include "soa/js/js_wrapped.h"
#include "soa/js/js_call.h"
#include "soa/js/js_registry.h"
#include "jml/arch/timers.h"
#include "jml/utils/guard.h"
#include "soa/sigslot/slot.h"
#include "jml/utils/guard.h"
#include "soa/sigslot/slot_js.h"
#include <boost/static_assert.hpp>
#include "soa/js/js_call.h"


using namespace std;
using namespace v8;
using namespace node;


namespace Datacratic {
namespace JS {

extern Registry registry;

extern const char * const opstatsModule;

const char * const opstatsModule = "opstats";

// Node.js initialization function; called to set up the OPSTATS object
extern "C" void
init(Handle<v8::Object> target)
{
    Datacratic::JS::registry.init(target, opstatsModule);
}


/*****************************************************************************/
/* CARBON CONNECTOR JS                                                       */
/*****************************************************************************/

const char * CarbonConnectorName = "CarbonConnector";

struct CarbonConnectorJS
    : public JSWrapped2<CarbonConnector, CarbonConnectorJS,
                        CarbonConnectorName,
                        opstatsModule, true> {

    CarbonConnectorJS()
    {
    }

    CarbonConnectorJS(const v8::Handle<v8::Object> & This,
             const std::shared_ptr<CarbonConnector> & logger
                 = std::shared_ptr<CarbonConnector>())
    {
        wrap(This, logger);
    }

    static Handle<v8::Value>
    New(const Arguments & args)
    {
        try {
            if (args.Length() > 0) {
                new CarbonConnectorJS(args.This(),
                                      std::shared_ptr<CarbonConnector>
                                      (new CarbonConnector()));
                return open(args);
            }
            else {
                new CarbonConnectorJS(args.This(),
                                      std::shared_ptr<CarbonConnector>());
            }

            return args.This();
        } HANDLE_JS_EXCEPTIONS;
    }

    static void Initialize()
    {
        Persistent<FunctionTemplate> t = Register(New);

        // Instance methods
        NODE_SET_PROTOTYPE_METHOD(t, "open", open);

        registerMemberFn(&CarbonConnector::stop, "close");
        registerMemberFn(&CarbonConnector::stop, "stop");
        registerMemberFn(&CarbonConnector::dump, "dump");

        registerMemberFn(&CarbonConnector::recordHit, "recordHit");
        registerMemberFn(&CarbonConnector::recordCount, "recordCount");
        registerMemberFn(&CarbonConnector::recordStableLevel, "recordStableLevel");
        registerMemberFn(&CarbonConnector::recordLevel, "recordLevel");
        registerMemberFn(&CarbonConnector::recordOutcome, "recordOutcome");
    }

    static Handle<v8::Value>
    open(const Arguments & args)
    {
        try {
            // Make sure Node doesn't exit and we don't get GCd when the
            // event loop is running.

            ev_ref(ev_default_loop());
            v8::Persistent<v8::Object> phandle
                = v8::Persistent<v8::Object>::New(args.This());

            auto cleanup = [=] ()
                {
                    v8::Persistent<v8::Object> handle = phandle;
                    ev_unref(ev_default_loop());
                    handle.Clear();
                    handle.Dispose();
                };

            ML::Call_Guard doCleanup(cleanup);  // cleanup on exception

            if (!args[0].IsEmpty()
                && args[0]->IsArray()) {
                getShared(args)->open(getArg<vector<string> >
                                      (args, 0, "carbonAddr"),
                                      getArg(args, 1, "statsPath"),
                                      getArg(args, 2, "dumpInterval"),
                                      cleanup);
            }
            else {
                getShared(args)->open(getArg<string>(args, 0, "carbonAddr"),
                                      getArg(args, 1, "statsPath"),
                                      getArg(args, 2, "dumpInterval"),
                                      cleanup);
            }
            
            doCleanup.clear();

            return args.This();
        } HANDLE_JS_EXCEPTIONS;
    }
};

std::shared_ptr<CarbonConnector>
from_js(const JSValue & value, std::shared_ptr<CarbonConnector> *)
{
    return CarbonConnectorJS::fromJS(value);
}

CarbonConnector *
from_js(const JSValue & value, CarbonConnector **)
{
    return CarbonConnectorJS::fromJS(value).get();
}

std::shared_ptr<CarbonConnector>
from_js_ref(const JSValue& value, std::shared_ptr<CarbonConnector>*)
{
    return CarbonConnectorJS::fromJS(value);
}

void
to_js(JS::JSValue& value, const std::shared_ptr<CarbonConnector>& proxy)
{
    value = CarbonConnectorJS::toJS(proxy);
}


} // namespace JS
} // namespace Datacratic
