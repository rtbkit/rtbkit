/** win_cost_model_js.cc                                 -*- C++ -*-
    Eric Robert, 15 May 2013
    Copyright (c) 2013 Datacratic.  All rights reserved.

    JS Wrappers for the win cost model classes.
*/

#include "win_cost_model_js.h"
#include "soa/js/js_wrapped.h"
#include "soa/js/js_call.h"
#include "soa/js/js_utils.h"
#include "soa/js/js_registry.h"
#include "soa/sigslot/slot.h"
#include "jml/utils/smart_ptr_utils.h"


using namespace std;
using namespace v8;
using namespace node;
using namespace RTBKIT;


namespace Datacratic {
namespace JS {


/******************************************************************************/
/* BID                                                                        */
/******************************************************************************/

extern const char* const WinCostModelName;
const char * const WinCostModelName = "WinCostModel";

struct WinCostModelJS : public JSWrapped2<WinCostModel, WinCostModelJS, WinCostModelName, rtbModule>
{
    WinCostModelJS() {}

    WinCostModelJS(  v8::Handle<v8::Object> This,
                     const std::shared_ptr<WinCostModel>& wcm = std::shared_ptr<WinCostModel>())
    {
        HandleScope scope;
        wrap(This, wcm);
    }

    static Handle<v8::Value>
    New(const Arguments& args)
    {
        try {
            new WinCostModelJS(args.This(), make_shared<WinCostModel>());
            return args.This();
        }
        HANDLE_JS_EXCEPTIONS;
    }

    static void
    Initialize()
    {
        Persistent<FunctionTemplate> t = Register(New);

        registerROProperty(&WinCostModel::name, "name");
        registerROProperty(&WinCostModel::data, "data");
    }
};

WinCostModel* from_js(const JSValue& value, WinCostModel**)
{
    return WinCostModelJS::fromJS(value).get();
}

WinCostModel from_js(const JSValue& value, WinCostModel*)
{
    return *WinCostModelJS::fromJS(value);
}

void to_js(JS::JSValue& value, const WinCostModel& wcm)
{
    value = WinCostModelJS::toJS(make_shared<WinCostModel>(wcm));
}


} // namepsace JS
} // namespace Datacratic
