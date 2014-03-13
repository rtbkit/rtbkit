/** currency_js.cc                                 -*- C++ -*-
    Rémi Attab, 01 Mar 2013
    Copyright (c) 2013 Datacratic.  All rights reserved.

    JS wrapper for the currency classes.

*/


#include "currency_js.h"
#include "rtbkit/common/currency.h"
#include "soa/js/js_wrapped.h"
#include "soa/js/js_call.h"
#include "soa/js/js_utils.h"
#include "soa/js/js_registry.h"
#include "soa/sigslot/slot.h"

using namespace std;
using namespace v8;
using namespace node;
using namespace RTBKIT;


namespace Datacratic {
namespace JS {


/******************************************************************************/
/* AMOUNT JS                                                                  */
/******************************************************************************/

extern const char* const AmountName;
const char * const AmountName = "Amount";

struct AmountJS :
    public JSWrapped2<
        Amount,
        AmountJS,
        AmountName,
        bidRequestModule>
{
    AmountJS() {}

    AmountJS(
            v8::Handle<v8::Object> This,
            const std::shared_ptr<Amount>& amount =
                std::shared_ptr<Amount>())
    {
        HandleScope scope;
        wrap(This, amount);
    }

    static Handle<v8::Value>
    New(const Arguments& args)
    {
        try {
            string currency = getArg<string>(args, 1, "NONE", "currency");
            int64_t value = getArg<int64_t>(args, 2, 0, "value");

            //cerr << "new amount " << currency << " " << value << endl;

            auto obj = std::make_shared<Amount>(currency, value);
            new AmountJS(args.This(), obj);

            return args.This();
        }
        HANDLE_JS_EXCEPTIONS;
    }


    static void
    Initialize()
    {
        Persistent<FunctionTemplate> t = Register(New);

        registerROProperty(&Amount::value, "value");
        registerROProperty(&Amount::getCurrencyStr, "currencyCode");

        registerMemberFn(&Amount::isZero, "isZero");
        registerMemberFn(&Amount::isNonNegative, "isNonNegative");
        registerMemberFn(&Amount::isNegative, "isNegative");

        registerMemberFn(&Amount::limit, "limit");
        registerMemberFn(&Amount::currencyIsCompatible, "currencyIsCompatible");

        // Typedef required to disambiguate the operator -
        typedef Amount (Amount::*SubOpFn)() const;

        registerMemberFn((SubOpFn)&Amount::operator-, "sub");
        registerMemberFn(&Amount::operator+, "add");
        registerMemberFn(&Amount::operator<, "lt");
        registerMemberFn(&Amount::operator<=, "le");
        registerMemberFn(&Amount::operator>, "gt");
        registerMemberFn(&Amount::operator>=, "ge");
        registerMemberFn(&Amount::operator==, "equals");

#if 0
        registerMemberFn(&Amount::toMicro, "toMicro");
        registerMemberFn(&Amount::toUnit, "toUnit");
        registerMemberFn(&Amount::toCPM, "toCPM");
        registerMemberFn(&Amount::toMicroCPM, "toMicroCPM");
#endif

        registerMemberFn(&Amount::getCurrencyStr, "currencyCode");

        //NODE_SET_PROTOTYPE_METHOD(t, "currencyCode", getCurrencyCode);
    }

    static Handle<Value>
    getCurrencyCode(const Arguments& args)
    {
        try {
            return Datacratic::JS::toJS(getShared(args)->getCurrencyStr());
        }
        HANDLE_JS_EXCEPTIONS;

        return Handle<Value>();
    }
};


Amount from_js(const JSValue& value, Amount*)
{
    return *AmountJS::fromJS(value);
}

Amount* from_js(const JSValue& value, Amount**)
{
    return AmountJS::fromJS(value).get();
}

void to_js(JS::JSValue& value, const Amount& amount)
{
    value = AmountJS::toJS(make_shared<Amount>(amount));
}


/******************************************************************************/
/* AMOUNT CONSTRUCTOR                                                         */
/******************************************************************************/

template<typename T, typename V>
Handle<Value> amountFn(const Arguments& args)
{
    try {
        V value = getArg<V>(args, 0, 0, "value");
        return AmountJS::toJS(make_shared<T>(value));
    }
    HANDLE_JS_EXCEPTIONS;
}

template<typename Fn>
void registerModuleFn(Handle<Object>& target, const string& name, Fn fn)
{
    auto tpl = Persistent<FunctionTemplate>::New(FunctionTemplate::New(fn));
    target->Set(String::NewSymbol(name.c_str()), tpl->GetFunction());
}

void initCurrencyFunctions(Handle<v8::Object>& target)
{
    registerModuleFn(target, "MicroUSD",     amountFn<MicroUSD, int64_t>);
    registerModuleFn(target, "USD",          amountFn<USD, double>);
    registerModuleFn(target, "USD_CPM",      amountFn<USD_CPM, double>);
    registerModuleFn(target, "MicroUSD_CPM", amountFn<MicroUSD_CPM, int64_t>);


    registerModuleFn(target, "MicroEUR", amountFn<MicroEUR, int64_t>);
    registerModuleFn(target, "EUR", amountFn<EUR, double>);
    registerModuleFn(target, "EUR_CPM", amountFn<EUR_CPM, double>);
    registerModuleFn(target, "MicroEUR_CPM", amountFn<MicroEUR_CPM, int64_t>);
}

} // namepsace JS
} // namespace Datacratic
