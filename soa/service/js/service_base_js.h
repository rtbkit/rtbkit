/** service_base_js.h                                 -*- C++ -*-
    Rémi Attab, 20 Jul 2012
    Copyright (c) 2012 Datacratic.  All rights reserved.

    Service Base JS wrappers.

*/

#ifndef __service__service_base_js_h__
#define __service__service_base_js_h__

#include "service_js.h"
#include "soa/service/service_base.h"
#include "soa/js/js_wrapped.h"
#include "jml/utils/smart_ptr_utils.h"

namespace Datacratic {
namespace JS {

/******************************************************************************/
/* SERVICE PROXIES                                                            */
/******************************************************************************/

std::shared_ptr<ServiceProxies>
from_js(const JSValue& value, std::shared_ptr<ServiceProxies>*);

std::shared_ptr<ServiceProxies>
from_js_ref(const JSValue& value, std::shared_ptr<ServiceProxies>*);

ServiceProxies*
from_js(const JSValue& value, ServiceProxies**);

void
to_js(JS::JSValue& value, const std::shared_ptr<ServiceProxies>& proxies);


/******************************************************************************/
/* SERVICE BASE                                                               */
/******************************************************************************/

extern const char* const ServiceBaseName;

struct ServiceBaseJS :
    public JSWrapped2<
        ServiceBase,
        ServiceBaseJS,
        ServiceBaseName,
        serviceModule>
{
    ServiceBaseJS();
    ServiceBaseJS(
            v8::Handle<v8::Object> This,
            const std::shared_ptr<ServiceBase>& service =
                std::shared_ptr<ServiceBase>());

    static v8::Handle<v8::Value> New(const v8::Arguments& args);
    static void Initialize();
    static v8::Handle<v8::Value> recordHit(const v8::Arguments& args);
    static v8::Handle<v8::Value> recordCount(const v8::Arguments& args);
    static v8::Handle<v8::Value> recordOutcome(const v8::Arguments& args);
    static v8::Handle<v8::Value> recordLevel(const v8::Arguments& args);
};

std::shared_ptr<ServiceBase>
from_js(const JSValue& value, std::shared_ptr<ServiceBase>*);

std::shared_ptr<ServiceBase>
from_js_ref(const JSValue& value, std::shared_ptr<ServiceBase>*);

ServiceBase*
from_js(const JSValue& value, ServiceBase**);

void
to_js(JS::JSValue& value, const std::shared_ptr<ServiceBase>& service);




} // namespace JS
} // Datacratic

#endif // __service__service_base_js_h__
