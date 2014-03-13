/** service_base_js.cc                                 -*- C++ -*-
    Rémi Attab, 20 Jul 2012
    Copyright (c) 2012 Datacratic.  All rights reserved.

    Service Base JS wrappers.

*/

#include "service_base_js.h"
#include "soa/service/js/opstats_js.h"
#include "soa/js/js_call.h"
#include "soa/sigslot/slot.h"

using namespace std;
using namespace v8;

namespace Datacratic {
namespace JS {


/******************************************************************************/
/* SERVICE PROXIES                                                            */
/******************************************************************************/

extern const char* const ServiceProxiesName;
const char * const ServiceProxiesName = "ServiceProxies";

struct ServiceProxiesJS :
    public JSWrapped2<
        ServiceProxies,
        ServiceProxiesJS,
        ServiceProxiesName,
        serviceModule>
{
    ServiceProxiesJS() {}

    ServiceProxiesJS(
            v8::Handle<v8::Object> This,
            const std::shared_ptr<ServiceProxies>& proxies =
                std::shared_ptr<ServiceProxies>())
    {
        HandleScope scope;
        wrap(This, proxies);
    }

    static Handle<v8::Value>
    New(const Arguments& args)
    {
        try {
            auto obj = ML::make_std_sp(new ServiceProxies());
            new ServiceProxiesJS(args.This(), obj);

            return args.This();
        }
        HANDLE_JS_EXCEPTIONS;
    }


    static void
    Initialize()
    {
        Persistent<FunctionTemplate> t = Register(New);

        NODE_SET_PROTOTYPE_METHOD(t, "bootstrap", bootstrap);
        NODE_SET_PROTOTYPE_METHOD(t, "logToCarbon", logToCarbon);
        registerMemberFn(&Datacratic::ServiceProxies::useZookeeper, "useZookeeper");
        registerMemberFn(&Datacratic::ServiceProxies::getServiceClassInstances, "getServiceClassInstances");
    }

    static Handle<Value>
    logToCarbon(const Arguments& args)
    {
        try {
            ExcCheck(args.Length() >= 1, "Invalid arguments");

            if (args[0]->IsString()) {
                string carbonURI = getArg<string>(args, 0, "carbonConnection");
                string prefix = getArg<string>(args, 1, "", "prefix");
                getShared(args)->logToCarbon(carbonURI, prefix);
            }
            else {
                auto conn = getArg<std::shared_ptr<CarbonConnector> >(
                        args, 0, "carbonConnector");

                getShared(args)->logToCarbon(conn);
            }

            return Handle<Value>();
        } HANDLE_JS_EXCEPTIONS;
    }

    static Handle<Value>
    bootstrap(const Arguments& args) {
        try {
            ExcCheck(args.Length() == 1, "Invalid argument count");

            auto arg = args[0];
            auto & serviceProxies = *getShared(args);
            if (arg->IsString()) {
                serviceProxies.bootstrap(getArg<string>(args, 0, "path"));
            } else if (arg->IsObject()) {
                auto config = getArg<Json::Value>(args, 0, "config");
                serviceProxies.bootstrap(config);
            } else {
                ExcCheck(false, "Invalid argument type");
            }

            return Handle<Value>();
        } HANDLE_JS_EXCEPTIONS;
    }
};

std::shared_ptr<ServiceProxies>
from_js(const JSValue& value, std::shared_ptr<ServiceProxies>*)
{
    return ServiceProxiesJS::fromJS(value);
}

std::shared_ptr<ServiceProxies>
from_js_ref(const JSValue& value, std::shared_ptr<ServiceProxies>*)
{
    return ServiceProxiesJS::fromJS(value);
}

ServiceProxies*
from_js(const JSValue& value, ServiceProxies**)
{
    return ServiceProxiesJS::fromJS(value).get();
}

void
to_js(JS::JSValue& value, const std::shared_ptr<ServiceProxies>& proxies)
{
    value = ServiceProxiesJS::toJS(proxies);
}

std::shared_ptr<ServiceProxies>
getServiceProxiesSharedPointer(const JS::JSValue & value)
{
    if(ServiceProxiesJS::tmpl->HasInstance(value))
    {
        std::shared_ptr<ServiceProxies> proxies =
            ServiceProxiesJS::getSharedPtr(value);
        return proxies;
    }
    std::shared_ptr<ServiceProxies> proxies;
    return proxies;
}



/******************************************************************************/
/* SERVICE BASE                                                               */
/******************************************************************************/

const char * const ServiceBaseName = "ServiceBase";

ServiceBaseJS::
ServiceBaseJS()
{}

ServiceBaseJS::
ServiceBaseJS(
        v8::Handle<v8::Object> This,
        const std::shared_ptr<ServiceBase>& service)
{
    HandleScope scope;
    wrap(This, service);
}

Handle<v8::Value>
ServiceBaseJS::
New(const Arguments& args)
{
    try {
        ExcCheck(args.Length() == 2, "Invalid arguments");

        string serviceName = getArg<string>(args, 0, "serviceName");
        auto proxies = getArg<std::shared_ptr<ServiceProxies> >(
                args, 1, "proxies");

        auto obj = ML::make_std_sp(new ServiceBase(serviceName, proxies));
        new ServiceBaseJS(args.This(), obj);

        return args.This();
    }
    HANDLE_JS_EXCEPTIONS;
}


void
ServiceBaseJS::
Initialize()
{
    Persistent<FunctionTemplate> t = Register(New);

    NODE_SET_PROTOTYPE_METHOD(t, "recordHit", recordHit);
    NODE_SET_PROTOTYPE_METHOD(t, "recordCount", recordCount);
    NODE_SET_PROTOTYPE_METHOD(t, "recordOutcome", recordOutcome);
    NODE_SET_PROTOTYPE_METHOD(t, "recordLevel", recordLevel);
}

Handle<Value>
ServiceBaseJS::
recordHit(const Arguments& args)
{
    try {
        for (int i = 0; i < args.Length(); ++i) {
            string event = getArg<string>(args, i, "event");
            getShared(args)->recordHit(event);
        }
        return Handle<Value>();
    } HANDLE_JS_EXCEPTIONS;
}

Handle<Value>
ServiceBaseJS::
recordCount(const Arguments& args)
{
    try {
        float count = getArg<float>(args, 0, "count");

        for (int i = 1; i < args.Length(); ++i) {
            string event = getArg<string>(args, i, "event");
            getShared(args)->recordCount(count, event);
        }
        return Handle<Value>();
    } HANDLE_JS_EXCEPTIONS;
}

Handle<Value>
ServiceBaseJS::
recordOutcome(const Arguments& args)
{
    try {
        float count = getArg<float>(args, 0, "outcome");

        for (int i = 1; i < args.Length(); ++i) {
            string event = getArg<string>(args, i, "event");
            getShared(args)->recordOutcome(count, event);
        }
        return Handle<Value>();
    } HANDLE_JS_EXCEPTIONS;
}

Handle<Value>
ServiceBaseJS::
recordLevel(const Arguments& args)
{
    try {
        float count = getArg<float>(args, 0, "outcome");

        for (int i = 1; i < args.Length(); ++i) {
            string event = getArg<string>(args, i, "event");
            getShared(args)->recordLevel(count, event);
        }
        return Handle<Value>();
    } HANDLE_JS_EXCEPTIONS;
}


std::shared_ptr<ServiceBase>
from_js(const JSValue& value, std::shared_ptr<ServiceBase>*)
{
    return ServiceBaseJS::fromJS(value);
}

std::shared_ptr<ServiceBase>
from_js_ref(const JSValue& value, std::shared_ptr<ServiceBase>*)
{
    return ServiceBaseJS::fromJS(value);
}

ServiceBase*
from_js(const JSValue& value, ServiceBase**)
{
    return ServiceBaseJS::fromJS(value).get();
}

void
to_js(JS::JSValue& value, const std::shared_ptr<ServiceBase>& service)
{
    value = ServiceBaseJS::toJS(service);
}

std::shared_ptr<ServiceBase>
getServiceBaseSharedPointer(const JS::JSValue & value)
{
    if(ServiceBaseJS::tmpl->HasInstance(value))
    {
        std::shared_ptr<ServiceBase> service =
            ServiceBaseJS::getSharedPtr(value);
        return service;
    }
    std::shared_ptr<ServiceBase> service;
    return service;
}






} // namepsace JS
} // namespace Datacratic
