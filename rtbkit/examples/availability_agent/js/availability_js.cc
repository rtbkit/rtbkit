/** availability_js.cc                                 -*- C++ -*-
    Rémi Attab, 02 Aug 2012
    Copyright (c) 2012 Recoset.  All rights reserved.

    Description

*/


#include "rtbkit/examples/availability_agent/availability_agent.h"
#include "soa/service/js/service_base_js.h"
#include "soa/service/js/opstats_js.h"
#include "soa/js/js_wrapped.h"
#include "soa/js/js_call.h"
#include "soa/js/js_utils.h"
#include "soa/js/js_registry.h"
#include "soa/sigslot/slot.h"
#include "jml/utils/smart_ptr_utils.h"

#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>

using namespace std;
using namespace v8;
using namespace node;


namespace Datacratic {
namespace JS {


/******************************************************************************/
/* MODULE                                                                     */
/******************************************************************************/

extern const char * const availabilityModule;
const char * const availabilityModule = "availability";


// Node.js initialization function; called to set up the mmap object
extern "C" void
init(Handle<v8::Object> target)
{
    Datacratic::JS::registry.init(target, availabilityModule);
}


/******************************************************************************/
/* AVAILABILITY AGENT JS                                                      */
/******************************************************************************/

extern const char* const AvailabilityAgentName;
const char * const AvailabilityAgentName = "AvailabilityAgent";

struct AvailabilityAgentJS :
    public JSWrapped2<
        AvailabilityAgent,
        AvailabilityAgentJS,
        AvailabilityAgentName,
        availabilityModule>
{
    AvailabilityAgentJS() {}

    AvailabilityAgentJS(
            v8::Handle<v8::Object> This,
            const std::shared_ptr<AvailabilityAgent>& agent =
                std::shared_ptr<AvailabilityAgent>())
    {
        HandleScope scope;
        wrap(This, agent);
    }

    static Handle<v8::Value>
    New(const Arguments& args)
    {
        try {
            auto proxies = getArg(
                    args, 0, std::make_shared<ServiceProxies>(), "proxies");

            auto obj = ML::make_std_sp(new AvailabilityAgent(proxies));
            new AvailabilityAgentJS(args.This(), obj);

            return args.This();
        }
        HANDLE_JS_EXCEPTIONS;
    }


    static void
    Initialize()
    {
        Persistent<FunctionTemplate> t = Register(New);

        registerMemberFn(&AvailabilityAgent::start, "start");
        registerMemberFn(&AvailabilityAgent::shutdown, "shutdown");
        registerMemberFn(
                &AvailabilityAgent::setBidProbability, "setBidProbability");
        registerMemberFn(
                &AvailabilityAgent::setRequestBufferSize,
                "setRequestBufferSize");

        NODE_SET_PROTOTYPE_METHOD(t, "checkConfig", checkConfig);

        // registerRWProperty(&AvailabilityAgent::var, "var", v8::DontDelete);
        // NODE_SET_PROTOTYPE_METHOD(t, "bar", bar);
    }

    static Handle<Value>
    checkConfig(const Arguments& args)
    {
        try {
            Json::Value config = getArg<Json::Value>(args, 0, "config");
            Json::Value report = getShared(args)->checkConfig(config);

            return Datacratic::JS::toJS(report);
        } HANDLE_JS_EXCEPTIONS;
    }
};

std::shared_ptr<AvailabilityAgent>
from_js(const JSValue& value, std::shared_ptr<AvailabilityAgent>*)
{
    return AvailabilityAgentJS::fromJS(value);
}

std::shared_ptr<AvailabilityAgent>
from_js_ref(const JSValue& value, std::shared_ptr<AvailabilityAgent>*)
{
    return AvailabilityAgentJS::fromJS(value);
}

AvailabilityAgent*
from_js(const JSValue& value, AvailabilityAgent**)
{
    return AvailabilityAgentJS::fromJS(value).get();
}

void
to_js(JS::JSValue& value, const std::shared_ptr<AvailabilityAgent>& agent)
{
    value = AvailabilityAgentJS::toJS(agent);
}

std::shared_ptr<AvailabilityAgent>
getAvailabilityAgentSharedPointer(const JS::JSValue & value)
{
    if(AvailabilityAgentJS::tmpl->HasInstance(value))
    {
        std::shared_ptr<AvailabilityAgent> agent =
            AvailabilityAgentJS::getSharedPtr(value);
        return agent;
    }
    std::shared_ptr<AvailabilityAgent> agent;
    return agent;
}



} // namepsace JS
} // namespace Datacratic
