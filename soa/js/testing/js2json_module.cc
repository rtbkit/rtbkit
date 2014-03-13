/* v8_inheritance_module.cc                                        -*- C++ -*-
   Jeremy Barnes, 26 July 2010
   Copyright (c) 2010 Datacratic.  All rights reserved.

   Module to test v8 inheritance from C++.
*/

#include <signal.h>
#include "soa/js/js_wrapped.h"
#include "soa/js/js_utils.h"
#include "soa/js/js_registry.h"
#include "v8.h"
#include "jml/compiler/compiler.h"
#include "jml/utils/smart_ptr_utils.h"
#include "soa/jsoncpp/json.h"

using namespace v8;
using namespace std;
using namespace Datacratic;
using namespace Datacratic::JS;



struct FromTo {
    virtual ~FromTo()
    {
    }
};

const char * FromToName = "FromTo";
const char * FromToModule = "ft";

struct FromToJS
    : public JSWrapped2<FromTo, FromToJS, FromToName, FromToModule> {

    FromToJS(v8::Handle<v8::Object> This,
             const std::shared_ptr<FromTo> & fromto
             = std::shared_ptr<FromTo>())
    {
        HandleScope scope;
        wrap(This, fromto);
    }

    static Persistent<v8::FunctionTemplate>
    Initialize()
    {
        Persistent<FunctionTemplate> t = Register(New, Setup);

        // Instance methods
        NODE_SET_PROTOTYPE_METHOD(t, "roundTrip", roundTrip);
        NODE_SET_PROTOTYPE_METHOD(t, "getJSON1", getJSON1);
        NODE_SET_PROTOTYPE_METHOD(t, "getJSON2", getJSON2);
        NODE_SET_PROTOTYPE_METHOD(t, "getLongLongInt", getLongLongInt);
        NODE_SET_PROTOTYPE_METHOD(t, "getJSONLongLongInt", getJSONLongLongInt);

        return t;
    }

    static Handle<Value>
    New(const Arguments & args)
    {
        try {
            new FromToJS(args.This(), ML::make_std_sp(new FromTo()));
            return args.This();
        } HANDLE_JS_EXCEPTIONS;
    }

    static Handle<v8::Value>
    roundTrip(const Arguments & args)
    {
        try {
            Json::Value val = getArg(args, 0, "arg");
            JSValue result;
            to_js(result, val);
            return result;
        } HANDLE_JS_EXCEPTIONS;
    }

    static Handle<v8::Value>
    getJSON1(const Arguments & args)
    {
        try {
            Json::Value val;
            val["a"] = 1;
            JSValue result;
            to_js(result, val);
            return result;
        } HANDLE_JS_EXCEPTIONS;
    }

    static Handle<v8::Value>
    getJSON2(const Arguments & args)
    {
        try {
            Json::Value val;
            val["a"]["b"][0u] = 1;
            val["a"]["b"][1] = 2.2;
            val["a"]["c"] = true;
            val["d"] = "string";

            JSValue result;
            to_js(result, val);
            return result;
        } HANDLE_JS_EXCEPTIONS;
    }

    static Handle<v8::Value>
    getLongLongInt(const Arguments & args)
    {
        try {
            // Json::Value val;
            // val["long_long"] = (2LL)<<33;

            JSValue result;
            to_js(result, (1LL)<<33);
            return result;
        } HANDLE_JS_EXCEPTIONS;
    }

    static Handle<v8::Value>
    getJSONLongLongInt(const Arguments & args)
    {
        try {
            Json::Value val;
            val["long_long"] = (1LL)<<33;

            JSValue result;
            to_js(result, val);
            return result;
        } HANDLE_JS_EXCEPTIONS;
    }
};

extern "C" void
init(Handle<v8::Object> target)
{
    registry.init(target, FromToModule);
}
