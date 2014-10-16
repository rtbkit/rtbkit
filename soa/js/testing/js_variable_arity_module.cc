/** js_variable_arity_test.cc
    Jeremy Barnes, 28 October 2012
    Copyright (c) 2012 Datacratic Inc.  All righte reerved.
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



struct ArityTestClass {
    virtual ~ArityTestClass()
    {
    }

    int method(int arg = 10)
    {
        return arg;
    }

    int constMethod(int arg = 6) const
    {
        return arg;
    }

    std::pair<string, int>
    twoArgMethod(string arg1 = "hello", int arg2 = 123)
    {
        return std::make_pair(arg1, arg2);
    }
};

const char * ArityTestClassName = "ArityTestClass";
const char * ArityTestModule = "ft";

struct ArityTestClassJS
    : public JSWrapped2<ArityTestClass, ArityTestClassJS, ArityTestClassName, ArityTestModule> {

    ArityTestClassJS(v8::Handle<v8::Object> This,
             const std::shared_ptr<ArityTestClass> & fromto
             = std::shared_ptr<ArityTestClass>())
    {
        HandleScope scope;
        wrap(This, fromto);
    }

    static Persistent<v8::FunctionTemplate>
    Initialize()
    {
        Persistent<FunctionTemplate> t = Register(New, Setup);

        registerMemberFn(&ArityTestClass::method, "method", 10);
        registerMemberFn(&ArityTestClass::constMethod, "constMethod", 6);
        registerMemberFn(&ArityTestClass::twoArgMethod, "twoArgMethod", "hello", 123);

#if 0
        auto m = &ArityTestClass::method;
        ArityTestClass * p = 0;
        int res = ((*p).*(m)) (2);
        cerr << "res = " << res << endl;
#endif

        return t;
    }

    static Handle<Value>
    New(const Arguments & args)
    {
        try {
            new ArityTestClassJS(args.This(), ML::make_std_sp(new ArityTestClass()));
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
};

extern "C" void
init(Handle<v8::Object> target)
{
    registry.init(target, ArityTestModule);
}

int main(int argc, char ** argv)
{
}
