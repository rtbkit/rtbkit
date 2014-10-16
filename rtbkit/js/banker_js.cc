/* banker_js.cc
   Jeremy Barnes, 6 April 2011
   Copyright (c) 2011 Datacratic.  All rights reserved.

   Banker JS interface.
*/


#include "banker_js.h"
#include "soa/js/js_wrapped.h"
#include "jml/utils/smart_ptr_utils.h"
#include "rtbkit/core/banker/null_banker.h"

using namespace std;
using namespace v8;
using namespace node;
using namespace RTBKIT;

namespace Datacratic {
namespace JS {


/*****************************************************************************/
/* BANKER JS                                                                 */
/*****************************************************************************/

const char * BankerName = "Banker";

struct BankerJS
    : public JSWrapped2<Banker, BankerJS, BankerName, rtbModule> {

    BankerJS(v8::Handle<v8::Object> This,
             const std::shared_ptr<Banker> & as
                  = std::shared_ptr<Banker>())
    {
        HandleScope scope;
        wrap(This, as);
    }

    BankerJS()
    {
    }

    static Handle<v8::Value>
    New(const Arguments & args)
    {
        try {
            new BankerJS(args.This());
            return args.This();
        } HANDLE_JS_EXCEPTIONS;
    }

    static void
    Initialize()
    {
        Persistent<FunctionTemplate> t = Register(New);
        
        //registerMemberFn(&Banker::addBudgetSync, "addBudget");
        //registerMemberFn(&Banker::setBudgetSync, "setBudget");
        //registerMemberFn(&Banker::authorizeBid, "authorizeBid");
        //registerMemberFn(&Banker::cancelBid, "cancelBid");
        //registerMemberFn(&Banker::winBid, "winBid");

        //registerMemberFn(&Banker::getCampaignStatusJson, "getCampaignStatus");
        //registerMemberFn(&Banker::dumpAllCampaignsJson, "dumpAllCampaigns");
    }

};

void to_js(JS::JSValue & value, const std::shared_ptr<Banker> & as)
{
    value = BankerJS::toJS(as);
}

std::shared_ptr<Banker>
from_js(const JSValue & value, std::shared_ptr<Banker> *)
{
    return BankerJS::fromJS(value);
}

Banker *
from_js(const JSValue & value, Banker **)
{
    return BankerJS::fromJS(value).get();
}


/*****************************************************************************/
/* NULL BANKER JS                                                            */
/*****************************************************************************/

const char * NullBankerName = "NullBanker";

struct NullBankerJS
    : public JSWrapped3<NullBanker,
                        NullBankerJS,
                        BankerJS,
                        NullBankerName, rtbModule, true> {

    NullBankerJS()
    {
    }

    NullBankerJS(const v8::Handle<v8::Object> & This,
           const std::shared_ptr<NullBanker> & data
               = std::shared_ptr<NullBanker>())
    {
        wrap(This, data);
    }

    static Handle<v8::Value>
    New(const Arguments & args)
    {
        try {
            new NullBankerJS
                (args.This(),
                 std::shared_ptr<NullBanker>(new NullBanker(getArg(args, 0, false, "authorize"))));
            return args.This();
        } HANDLE_JS_EXCEPTIONS;
    }
    
    static void Initialize()
    {
        Persistent<FunctionTemplate> t = Register(New);
        registerRWProperty(&NullBanker::authorize_, "authorize",
                           v8::DontDelete);
    }
};

std::shared_ptr<NullBanker>
from_js(const JSValue & value, std::shared_ptr<NullBanker> *)
{
    return NullBankerJS::fromJS(value);
}

NullBanker *
from_js(const JSValue & value, NullBanker **)
{
    return NullBankerJS::fromJS(value).get();
}

NullBanker &
from_js(const JSValue & value, NullBanker *)
{
    return *NullBankerJS::fromJS(value).get();
}

void
to_js(JSValue & value, const std::shared_ptr<NullBanker> & fset)
{
    value = registry.getWrapper(fset);
}


} // namespace JS
} // namespace Datacratic
