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

using namespace v8;
using namespace std;
using namespace Datacratic;
using namespace Datacratic::JS;


struct TestException {
};

const char * TestExceptionName = "TestException";
const char * TestModule = "exc";

struct TestExceptionJS
    : public JSWrapped2<TestException, TestExceptionJS,
                        TestExceptionName, TestModule> {

    static Persistent<v8::FunctionTemplate>
    Initialize()
    {
        Persistent<FunctionTemplate> t = Register(New, Setup);
        
        // Instance methods
        NODE_SET_PROTOTYPE_METHOD(t, "testMlException", testMlException);
        NODE_SET_PROTOTYPE_METHOD(t, "testMlException2", testMlException2);
        NODE_SET_PROTOTYPE_METHOD(t, "testStdException", testStdException);
        NODE_SET_PROTOTYPE_METHOD(t, "testPassThrough", testPassThrough);
        
        return t;
    }

    static Handle<Value>
    New(const Arguments & args)
    {
        try {
            new TestExceptionJS();
            return args.This();
        } HANDLE_JS_EXCEPTIONS;
    }

    static Handle<Value>
    testMlException(const Arguments & args)
    {
        try {
            try {
                throw ML::Exception("hello");
            } catch (const ML::Exception & exc) {
                //cerr << "exc is at " << &exc << endl;
                throw;
            }
        } HANDLE_JS_EXCEPTIONS;
    }

    static Handle<Value>
    testMlException2(const Arguments & args)
    {
        try {
            throw ML::Exception("hello2");
        } HANDLE_JS_EXCEPTIONS;
    }

    static Handle<Value>
    testStdException(const Arguments & args)
    {
        try {
            try {
                throw std::logic_error("bad medicine");
            } catch (const std::exception & exc) {
                //cerr << "exc is at " << &exc << endl;
                throw;
            }
        } HANDLE_JS_EXCEPTIONS;
    }

    static Handle<Value>
    testPassThrough(const Arguments & args)
    {
        try {
            v8::Local<v8::Function> fn(v8::Function::Cast(*args[0]));
            if (fn.IsEmpty())
                throw ML::Exception("should have gotten throwing function in");

            // Call the function
            v8::Handle<v8::Value> result = fn->Call(args.This(), 0, 0);
            
            if (result.IsEmpty())
                throw JSPassException();

            throw ML::Exception("test was supposed to have a function that threw");
        } HANDLE_JS_EXCEPTIONS;
    }


};

extern "C" void
init(Handle<v8::Object> target)
{
    registry.init(target, TestModule);
}
