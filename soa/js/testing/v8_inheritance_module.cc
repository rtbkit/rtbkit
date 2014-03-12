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

struct Base {
    virtual ~Base()
    {
    }

    virtual std::string type() const
    {
        return "Base";
    }

    int number() const { return 27; }
    int number2() const { return 27; }
};

struct Derived : public Base {
    virtual std::string type() const { return "Derived"; }
    int number() const { return 37; }
};

struct ReDerived : public Derived {
    virtual std::string type() const { return "ReDerived"; }
    int number() const { return 47; }
};

const char * BaseName = "Base";
const char * Module = "inheritance";

struct BaseJS : public JSWrapped2<Base, BaseJS, BaseName, Module> {

    BaseJS()
    {
    }

    BaseJS(const v8::Arguments & args)
    {
        wrap(args.This(), new Base());
    }

    static Persistent<v8::FunctionTemplate>
    Initialize()
    {
        Persistent<FunctionTemplate> t = Register(New, Setup);
        
        // Instance methods
        NODE_SET_PROTOTYPE_METHOD(t, "type", type);
        NODE_SET_PROTOTYPE_METHOD(t, "number", number);
        NODE_SET_PROTOTYPE_METHOD(t, "number2", number2);
        NODE_SET_PROTOTYPE_METHOD(t, "otherType", otherType);
        
        return t;
    }

    static Handle<Value>
    New(const Arguments & args)
    {
        try {
            new BaseJS(args);
            return args.This();
        } HANDLE_JS_EXCEPTIONS;
    }

    static Handle<Value>
    type(const Arguments & args)
    {
        HandleScope scope;
        try {
            return v8::String::NewSymbol(getShared(args)->type().c_str());
        } HANDLE_JS_EXCEPTIONS;
    }
    
    static Handle<Value>
    number(const Arguments & args)
    {
        HandleScope scope;
        try {
            return v8::Integer::New(getShared(args)->number());
        } HANDLE_JS_EXCEPTIONS;
    }

    static Handle<Value>
    number2(const Arguments & args)
    {
        HandleScope scope;
        try {
            return v8::Integer::New(getShared(args)->number2());
        } HANDLE_JS_EXCEPTIONS;
    }

    static Handle<Value>
    otherType(const Arguments & args)
    {
        HandleScope scope;
        try {
            Handle<Object> obj = args[0]->ToObject();
            if (obj.IsEmpty())
                throw ML::Exception("arg 0 wasn't an object");
            Base * other = BaseJS::getShared(obj);
            return v8::String::New(other->type().c_str());
        } HANDLE_JS_EXCEPTIONS;
    }
};

const char * DerivedName = "Derived";

struct DerivedJS
    : public JSWrapped3<Derived, DerivedJS, BaseJS, DerivedName, Module> {

    DerivedJS()
    {
    }

    DerivedJS(const v8::Arguments & args)
    {
        wrap(args.This(), new Derived());
    }

    static Persistent<v8::FunctionTemplate>
    Initialize()
    {
        Persistent<FunctionTemplate> t = Register(New, Setup);
        
        // Instance methods
        NODE_SET_PROTOTYPE_METHOD(t, "number", number);
        NODE_SET_PROTOTYPE_METHOD(t, "otherTypeDerived", otherTypeDerived);

        return t;
    }

    static Handle<Value>
    New(const Arguments & args)
    {
        try {
            new DerivedJS(args);
            return args.This();
        } HANDLE_JS_EXCEPTIONS;
    }

    static Handle<Value>
    number(const Arguments & args)
    {
        HandleScope scope;
        try {
            return v8::Integer::New(getShared(args)->number());
        } HANDLE_JS_EXCEPTIONS;
    }

    static Handle<Value>
    otherTypeDerived(const Arguments & args)
    {
        HandleScope scope;
        try {
            Handle<Object> obj = args[0]->ToObject();
            if (obj.IsEmpty())
                throw ML::Exception("arg 0 wasn't an object");
            Derived * other = DerivedJS::getShared(obj);
            return v8::String::New(other->type().c_str());
        } HANDLE_JS_EXCEPTIONS;
    }
};

const char * ReDerivedName = "ReDerived";

struct ReDerivedJS
    : public JSWrapped3<ReDerived, ReDerivedJS, DerivedJS, ReDerivedName,
                        Module> {

    ReDerivedJS(const v8::Arguments & args)
    {
        wrap(args.This(), new ReDerived());
    }

    static Persistent<v8::FunctionTemplate>
    Initialize()
    {
        Persistent<FunctionTemplate> t = Register(New, Setup);
        
        // Instance methods
        NODE_SET_PROTOTYPE_METHOD(t, "number", number);
        
        return t;
    }

    static Handle<Value>
    New(const Arguments & args)
    {
        try {
            new ReDerivedJS(args);
            return args.This();
        } HANDLE_JS_EXCEPTIONS;
    }

    static Handle<Value>
    number(const Arguments & args)
    {
        HandleScope scope;
        try {
            return v8::Integer::New(getShared(args)->number());
        } HANDLE_JS_EXCEPTIONS;
    }
};

struct Base2 {
    virtual ~Base2()
    {
    }

    virtual std::string type() const
    {
        return "Base2";
    }

    int number() const { return 27; }
    int number2() const { return 27; }
};

const char * Base2Name = "Base2";

struct Base2JS : public JSWrapped2<Base2, Base2JS, Base2Name, Module> {

    Base2JS()
    {
    }

    Base2JS(const v8::Arguments & args)
    {
        wrap(args.This(), new Base2());
    }

    static Persistent<v8::FunctionTemplate>
    Initialize()
    {
        Persistent<FunctionTemplate> t = Register(New, Setup);
        
        // Instance methods
        NODE_SET_PROTOTYPE_METHOD(t, "type", type);
        NODE_SET_PROTOTYPE_METHOD(t, "number", number);
        NODE_SET_PROTOTYPE_METHOD(t, "number2", number2);
        
        return t;
    }

    static Handle<Value>
    New(const Arguments & args)
    {
        try {
            new BaseJS(args);
            return args.This();
        } HANDLE_JS_EXCEPTIONS;
    }

    static Handle<Value>
    type(const Arguments & args)
    {
        HandleScope scope;
        try {
            return v8::String::NewSymbol(getShared(args)->type().c_str());
        } HANDLE_JS_EXCEPTIONS;
    }
    
    static Handle<Value>
    number(const Arguments & args)
    {
        HandleScope scope;
        try {
            return v8::Integer::New(getShared(args)->number());
        } HANDLE_JS_EXCEPTIONS;
    }

    static Handle<Value>
    number2(const Arguments & args)
    {
        HandleScope scope;
        try {
            return v8::Integer::New(getShared(args)->number2());
        } HANDLE_JS_EXCEPTIONS;
    }
};

extern "C" void
init(Handle<v8::Object> target)
{
    registry.init(target, Module);
}
