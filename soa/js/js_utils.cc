/* js_utils.cc
   Jeremy Barnes, 21 July 2010
   Copyright (c) 2010 Datacratic.  All rights reserved.

   Implementation of Javascript utility functions.
*/

#include "js_utils.h"
#include "js_value.h"
#include <cxxabi.h>
#include "jml/arch/demangle.h"
#include "jml/arch/exception_internals.h"
#include "jml/arch/backtrace.h"
#include "jml/utils/string_functions.h"
#include "jml/compiler/compiler.h"

using namespace std;
using namespace v8;
using namespace ML;

namespace ML {

__thread BacktraceInfo * current_backtrace = nullptr;

} // namespace ML

namespace Datacratic {
namespace JS {


/*****************************************************************************/
/* UTILITIES                                                                 */
/*****************************************************************************/

std::string cstr(const std::string & str)
{
    return str;
}

std::string cstr(const JSValue & val)
{
    return from_js(val, (string *)0);
}

v8::Handle<v8::Value>
injectBacktrace(v8::Handle<v8::Value> value)
{
    if (value.IsEmpty())
        throw ML::Exception("no object passed for backtrace injection");

    v8::Handle<v8::Object> obj(v8::Object::Cast(*value));

    if (obj.IsEmpty())
        throw ML::Exception("can't inject backtrace");

    v8::Handle<v8::Value> js_trace = obj->Get(v8::String::NewSymbol("stack"));

    vector<string> js_trace_elements = split(cstr(js_trace), '\n');

    // Frames to skip:
    // at [C++] ML::backtrace(int)
    // at [C++] Datacratic::JS::injectBacktrace(v8::Handle<v8::Value>)
    // at [C++] Datacratic::JS::mapException(ML::Exception const&)
    // at [C++] Datacratic::JS::translateCurrentException()
    int num_frames_to_skip = 0;

    vector<ML::BacktraceFrame> backtrace;
    if (current_backtrace && abi::__cxa_current_exception_type()) {
        // Skip:
        backtrace = ML::backtrace(*current_backtrace, num_frames_to_skip);
        delete current_backtrace;
        current_backtrace = 0;
    }
    else backtrace = ML::backtrace(num_frames_to_skip);

    string cpp_trace_str = js_trace_elements.at(0) + "\n";

    v8::Handle<v8::Array> cpp_trace(v8::Array::New(backtrace.size() + 1));
    for (unsigned i = 0;  i < backtrace.size();  ++i) {
        cpp_trace_str += "    at [C++] " + backtrace[i].print_for_trace() + "\n";
        cpp_trace->Set(v8::Uint32::New(i),
                       v8::String::New(backtrace[i].print().c_str()));
    }
    
    for (unsigned i = 1;  i < js_trace_elements.size();  ++i)
        cpp_trace_str += js_trace_elements[i] + "\n";

    cpp_trace->Set(v8::Uint32::New(backtrace.size()), js_trace);

    obj->Set(v8::String::NewSymbol("cpp_trace"), cpp_trace);
    obj->Set(v8::String::NewSymbol("js_trace"), js_trace);
    obj->Set(v8::String::NewSymbol("stack"), v8::String::New(cpp_trace_str.c_str()));

    return obj;
}

v8::Handle<Value>
mapException(const std::exception & exc)
{
    return v8::ThrowException
        (injectBacktrace
         (v8::Exception::Error(v8::String::New((type_name(exc)
                                                + ": " + exc.what()).c_str()))));
}

v8::Handle<Value>
mapException(const ML::Exception & exc)
{
    //cerr << "mapping ML::Exception " << exc.what() << endl;

    return v8::ThrowException
        (injectBacktrace
         (v8::Exception::Error(v8::String::New(exc.what()))));
}

v8::Handle<v8::Value>
translateCurrentException()
{
    if (!std::current_exception()) {
        throw ML::Exception("no exception");
    }

    try {
        throw;
    }
    catch(const JSPassException&) {
        return v8::Handle<v8::Value>();
    }
    catch(const ML::Exception& ex) {
        return mapException(ex);
    }
    catch(const std::exception& ex) {
        return mapException(ex);
    }
    JML_CATCH_ALL {
        std::string msg = "unknown exception type";
        auto error = v8::Exception::Error(v8::String::New(msg.c_str()));
        return v8::ThrowException(injectBacktrace(error));
    }
}

void passJsException(const v8::TryCatch & tc);

struct NullHandle NULL_HANDLE;

ValuePromise getArg(const JSArgs & args, int argnum,
                    const std::string & name)
{
    if (args.Length() <= argnum)
        throw ML::Exception("argument %d (%s) must be present",
                            argnum, name.c_str());

    ValuePromise arg;
    arg.value  = args[argnum];

    if (arg.value->IsUndefined() || arg.value->IsNull())
        throw ML::Exception("argument %d (%s) was %s",
                            argnum, name.c_str(), cstr(arg.value).c_str());

    arg.argnum = argnum;
    arg.name   = name;

    return arg;
}


string getArg(const JSArgs & args, int argnum, const string & defvalue,
         const std::string & name)
{

    return getArg<string>(args, argnum, defvalue, name);
}

/** Convert the given value into a persistent v8 function. */
v8::Persistent<v8::Function>
from_js(const JSValue & val, v8::Persistent<v8::Function> *)
{
    v8::Handle<v8::Function> fn(v8::Function::Cast(*val));
    if (fn.IsEmpty() || !fn->IsFunction()) {
        //cerr << "fn = " << cstr(fn) << endl;
        //cerr << "fn.IsEmpty() = " << fn.IsEmpty() << endl;
        //cerr << "val->IsFunction() = " << val->IsFunction() << endl;
        throw ML::Exception("expected a function; instead we got " + cstr(val));
    }
    
    return v8::Persistent<v8::Function>::New(fn);
}

v8::Local<v8::Function>
from_js(const JSValue & val, v8::Local<v8::Function> *)
{
    v8::Local<v8::Function> fn(v8::Function::Cast(*val));
    if (fn.IsEmpty() || !fn->IsFunction()) {
        //cerr << "fn = " << cstr(fn) << endl;
        //cerr << "fn.IsEmpty() = " << fn.IsEmpty() << endl;
        //cerr << "val->IsFunction() = " << val->IsFunction() << endl;
        throw ML::Exception("expected a function; instead we got " + cstr(val));
    }

    return fn;
}

v8::Handle<v8::Function>
from_js(const JSValue & val, v8::Handle<v8::Function> *)
{
    v8::Handle<v8::Function> fn(v8::Function::Cast(*val));
    if (fn.IsEmpty() || !fn->IsFunction()) {
        //cerr << "fn = " << cstr(fn) << endl;
        //cerr << "fn.IsEmpty() = " << fn.IsEmpty() << endl;
        //cerr << "val->IsFunction() = " << val->IsFunction() << endl;
        throw ML::Exception("expected a function; instead we got " + cstr(val));
    }

    return fn;
}

v8::Handle<v8::Array>
from_js(const JSValue & val, v8::Handle<v8::Array> *)
{
    v8::Handle<v8::Array> arr(v8::Array::Cast(*val));
    if (arr.IsEmpty() || !arr->IsArray())
        throw ML::Exception("expected an array; instead we got " + cstr(val));

    return arr;
}

v8::Handle<v8::Function>
getFunction(const std::string & script_source)
{
    using namespace v8;

    HandleScope scope;
    Handle<String> source = String::New(script_source.c_str());

    TryCatch tc;
    
    // Compile the source code.
    Handle<Script> script = Script::Compile(source);

    if (script.IsEmpty() && tc.HasCaught())
        throw ML::Exception("got exception compiling: "
                            + JS::cstr(tc.Exception()));
    if (script.IsEmpty())
        throw ML::Exception("compilation returned nothing");
    
    // Run the script to get the result (which should be a function)
    Handle<Value> result = script->Run();

    if (result.IsEmpty() && tc.HasCaught())
        throw ML::Exception("got exception compiling: "
                            + JS::cstr(tc.Exception()));
    if (result.IsEmpty())
        throw ML::Exception("compilation returned nothing");
    if (!result->IsFunction())
        throw ML::Exception("result of script isn't a function");
    
    v8::Local<v8::Function> fnresult(v8::Function::Cast(*result));

    return scope.Close(fnresult);
}

v8::Handle<v8::Array>
getIndexArray(size_t sz)
{
    v8::Handle<v8::Array> result(v8::Array::New(sz));
    
    for (unsigned i = 0;  i < sz;  ++i) {
        result->Set(v8::Uint32::New(i),
                    v8::Uint32::New(i));
    }

    return result;
}

/** Call a getter function that's in the data field of the given object. */
v8::Handle<v8::Value>
callGetterFn(v8::Local<v8::String> property,
             const v8::AccessorInfo & info)
{
    try {
        HandleScope scope;
        if (!info.Data()->IsFunction())
            throw ML::Exception("isn't a function");
        v8::Local<v8::Function> fn(v8::Function::Cast(*info.Data()));
        if (fn.IsEmpty())
            throw JSPassException();
        const int argc = 1;
        v8::Local<v8::Value> argv[argc] = { property };
        return scope.Close(fn->Call(info.This(), argc, argv));
    } HANDLE_JS_EXCEPTIONS;
}

void printObj(const v8::Handle<v8::Value> & val,
              std::ostream & stream,
              int nesting)
{
    string s(nesting * 4, ' ');
    stream << s << cstr(val) << endl;
    if (val->IsObject()) {
        auto objPtr = v8::Object::Cast(*val);
        if (!objPtr)
            return;

        v8::Local<v8::Array> properties = objPtr->GetPropertyNames();

        for(int i=0; i<properties->Length(); ++i) {
            v8::Local<v8::Value> key = properties->Get(i);
            v8::Local<v8::Value> val = objPtr->Get(key);

            stream << s << "  " << cstr(key) << ": " << cstr(val) << endl;
        }

        v8::Local<v8::Value> proto = objPtr->Get(v8::String::New("prototype"));
        stream << s << "  prototype " << cstr(proto) << endl;
        if (proto->IsObject())
            printObj(proto, stream, nesting + 1);

        v8::Local<v8::Value> proto2 = objPtr->GetPrototype(); 
        stream << s << "  .__proto__ " << cstr(proto2) << endl;
        if (proto2->IsObject())
            printObj(proto2, stream, nesting + 1);
    }
}

} // namespace JS
} // namespace Datacratic
