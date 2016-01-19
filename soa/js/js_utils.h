/* js_utils.h                                                      -*- C++ -*-
   Jeremy Barnes, 21 July 2010
   Copyright (c) 2010 Datacratic.  All rights reserved.

   Utility functions for js.
*/

#pragma once

#include <v8/v8.h>
#include <string>
#include "jml/arch/exception.h"
#include "jml/utils/exc_assert.h"
#include "jml/compiler/compiler.h"
#include "jml/arch/demangle.h"
#include "jml/arch/format.h"
#include "jml/utils/positioned_types.h"
#include <boost/shared_ptr.hpp>
#include "js_value.h"
#include <iostream>
#include <set>
#include <unordered_map>
#include "js_registry.h"
#include <boost/tuple/tuple.hpp>
#include <boost/type_traits/remove_reference.hpp>
#include <boost/type_traits/is_abstract.hpp>
#include <boost/type_traits/is_pod.hpp>
#include <boost/bind.hpp>
#include <tuple>
#include <array>

namespace ML {

template<typename F, class Underlying> class distribution;

template<typename T, size_t I, typename Sz, bool Sf, typename P, class A>
struct compact_vector;

} // namespace ML

namespace Datacratic {
namespace JS {


/*****************************************************************************/
/* UTILITIES                                                                 */
/*****************************************************************************/

std::string cstr(const std::string & str);

std::string cstr(const JSValue & val);

template<typename T>
std::string cstr(const v8::Local<T> & str)
{
    return cstr(JSValue(str));
}

template<typename T>
std::string cstr(const v8::Handle<T> & str)
{
    return cstr(JSValue(str));
}

struct JSPassException : public std::exception {
//    v8::Persistent<v8::Value> jsException;
//    v8::Persistent<v8::Value> jStackTrace;
//    v8::Persistent<v8::Message> jsMessage;
};

/** Throws a JsPassException object recording the given values. */
void passJsException(const v8::TryCatch & tc);

/** A function that will translate the current exception into a v8::Value
    that represents the exception in Javascript.

    This function should be called from within a catch(...) statement.  It
    uses magic to actually figure out what the exception is and how to
    translate it into a JS exception.  It works for all kinds of exceptions,
    not only those that are known about; unknown exceptions will generate a
    message with the type name of the exception.
*/
v8::Handle<v8::Value> translateCurrentException();

/** Modify the given exception object to include a C++ backtrace. */
v8::Handle<v8::Value> injectBacktrace(v8::Handle<v8::Value> value);

/** Macro to use after a try { block to catch and translate javascript
    exceptions. */
#define HANDLE_JS_EXCEPTIONS                                    \
    catch (...) {                                               \
        return Datacratic::JS::translateCurrentException();     \
    }

#define HANDLE_JS_EXCEPTIONS_SETTER                             \
    catch (...) {                                               \
        Datacratic::JS::translateCurrentException();            \
        return;                                                 \
    }

/** Convert a ML::Exception into a Javascript exception. */
v8::Handle<v8::Value>
mapException(const ML::Exception & exc);

/** Convert a std::exception into a Javascript exception. */
v8::Handle<v8::Value>
mapException(const std::exception & exc);

/** A class that will be implicitly converted into the correct null v8 handle
    for the situation. */
extern struct NullHandle {

    template<typename T> 
    operator v8::Handle<T>()
    {
        return v8::Handle<T>();
    }

    template<typename T> 
    operator v8::Local<T>()
    {
        return v8::Local<T>();
    }

} NULL_HANDLE;

/** Convert the Value handle to an Object handle.  Will throw if it isn't an
    object.
*/
inline v8::Handle<v8::Object>
toObject(v8::Handle<v8::Value> handle)
{
    ExcAssert(!handle.IsEmpty());
    if (!handle->IsObject())
        throw ML::Exception("value " + cstr(handle) + " is not an object");
    v8::Handle<v8::Object> object = handle->ToObject();
    if (object.IsEmpty())
        throw ML::Exception("value " + cstr(handle) + " is not an object");
    return object;
}

/** Convert the Value handle to an Array handle.  Will throw if it isn't an
    object.
*/
inline v8::Handle<v8::Array>
toArray(v8::Handle<v8::Value> handle)
{
    ExcAssert(!handle.IsEmpty());
    if (!handle->IsArray())
        throw ML::Exception("value " + cstr(handle) + " is not an array");
    v8::Handle<v8::Array> array(v8::Array::Cast(*handle));
    if (array.IsEmpty())
        throw ML::Exception("value " + cstr(handle) + " is not an array");
    return array;
}

template<typename T>
v8::Handle<v8::Value>
toJS(const T & t)
{
    JSValue val;
    to_js(val, t);
    return val;
}

template<typename T>
void to_js(JSValue & val, const std::shared_ptr<T> & p)
{
    if (!p)
        val = v8::Null();
    else
        val = registry.getWrapper(p);
}

template<typename T>
void to_js(JSValue & val, const std::vector<T> & v)
{
    v8::HandleScope scope;
    v8::Local<v8::Array> arr(v8::Array::New(v.size()));
    for (unsigned i = 0;  i < v.size();  ++i)
        arr->Set(v8::Uint32::New(i), toJS(v[i]));
    val = scope.Close(arr);
}

template<typename T, std::size_t SIZE>
void to_js(JSValue & val, const std::array<T, SIZE> & v)
{
    v8::HandleScope scope;
    v8::Local<v8::Array> arr(v8::Array::New(v.size()));
    for (unsigned i = 0;  i < v.size();  ++i)
        arr->Set(v8::Uint32::New(i), toJS(v[i]));
    val = scope.Close(arr);
}

template<typename T>
void to_js(JSValue & val, const std::set<T> & s)
{
    v8::HandleScope scope;
    v8::Local<v8::Array> arr(v8::Array::New(s.size()));
    int count = 0;
    for(auto i = s.begin(); i != s.end(); ++i)
    {
        arr->Set(v8::Uint32::New(count), toJS(*i));
        count++;
    }
    val = scope.Close(arr);
}

template<typename T>
void to_js(JSValue & val, const std::map<std::string, T> & s)
{
    v8::HandleScope scope;
    v8::Local<v8::Object> obj= v8::Object::New();
    for(auto i = s.begin(); i != s.end(); ++i)
    {
        obj->Set(v8::String::NewSymbol(i->first.c_str()), toJS(i->second));
    }
    val = scope.Close(obj);
}

template<typename T>
std::map<std::string, T>
from_js(const JSValue & val, const std::map<std::string, T> * = 0)
{
    if(!val->IsObject()) {
        throw ML::Exception("invalid JSValue for map extraction");
    }
    
    std::map<std::string, T> result;

    v8::HandleScope scope;

    auto objPtr = v8::Object::Cast(*val);

    v8::Local<v8::Array> properties = objPtr->GetOwnPropertyNames();

    for(int i=0; i<properties->Length(); ++i)
    {
        v8::Local<v8::Value> key = properties->Get(i);
        v8::Local<v8::Value> val = objPtr->Get(key);
        T val2 = from_js(JSValue(val), (T *)0);
        result[cstr(key)] = val2;
    }

    return result;
}

template<typename T>
void to_js(JSValue & val, const std::unordered_map<std::string, T> & s)
{
    v8::HandleScope scope;
    v8::Local<v8::Object> obj= v8::Object::New();
    for(auto i = s.begin(); i != s.end(); ++i)
    {
        obj->Set(v8::String::NewSymbol(i->first.c_str()), toJS(i->second));
    }
    val = scope.Close(obj);
}

template<typename K, typename V, typename H>
void to_js(JSValue & val, const std::unordered_map<K, V, H> & s)
{
    v8::HandleScope scope;
    v8::Local<v8::Object> obj= v8::Object::New();
    for(auto i = s.begin(); i != s.end(); ++i)
    {
        obj->Set(toJS(i->first), toJS(i->second));
    }
    val = scope.Close(obj);
}

template<typename K, typename V, typename H>
std::map<K, V, H>
from_js(const JSValue & val, const std::map<K, V, H> * = 0)
{
    if(!val->IsObject()) {
        throw ML::Exception("invalid JSValue for map extraction");
    }
    
    std::unordered_map<K, V, H> result;

    v8::HandleScope scope;

    auto objPtr = v8::Object::Cast(*val);

    v8::Local<v8::Array> properties = objPtr->GetOwnPropertyNames();

    for(int i=0; i<properties->Length(); ++i)
    {
        v8::Local<v8::Value> key = properties->Get(i);
        v8::Local<v8::Value> val = objPtr->Get(key);
        K key2 = from_js(JSValue(key), (K *)0);
        V val2 = from_js(JSValue(val), (V *)0);
        result[key2] = val2;
    }

    return result;
}

template<typename T, typename V>
void to_js(JSValue & val, const boost::tuple<T, V> & v)
{
    v8::HandleScope scope;
    v8::Local<v8::Array> arr(v8::Array::New(2));
    arr->Set(v8::Uint32::New(0), toJS(v.template get<0>()));
    arr->Set(v8::Uint32::New(1), toJS(v.template get<1>()));
    val = scope.Close(arr);
}

template<typename T, typename V>
void to_js(JSValue & val, const std::pair<T, V> & v)
{
    v8::HandleScope scope;
    v8::Local<v8::Array> arr(v8::Array::New(2));
    arr->Set(v8::Uint32::New(0), toJS( v.first  ));
    arr->Set(v8::Uint32::New(1), toJS( v.second ));
    val = scope.Close(arr);
}

template<typename T, typename V>
std::pair<T,V>
from_js(const JSValue & val, const std::pair<T,V> * = 0)
{
    if(!val->IsArray()) {
        throw ML::Exception("invalid JSValue for pair extraction");
    }

    auto arrPtr = v8::Array::Cast(*val);
    if(arrPtr->Length() != 2) {
        throw ML::Exception("invalid length for pair extraction");
    }

    return std::make_pair(from_js(JSValue(arrPtr->Get(0)),(T *) 0),
            from_js(JSValue(arrPtr->Get(1)),(V *) 0));
}

template<class Tuple, int Arg, int Size>
struct TupleOpsJs {

    static void unpack(v8::Local<v8::Array> & arr,
                       const Tuple & tuple)
    {
        JSValue val;
        to_js(val, std::get<Arg>(tuple));
        arr->Set(v8::Uint32::New(Arg), val);
        TupleOpsJs<Tuple, Arg + 1, Size>::unpack(arr, tuple);
    }

    static void pack(v8::Array & array,
                     Tuple & tuple)
    {
        if (Arg >= array.Length()) return;
        auto & el = std::get<Arg>(tuple);
        el = from_js(JSValue(array.Get(Arg)), &el);
        TupleOpsJs<Tuple, Arg + 1, Size>::pack(array, tuple);
    }
};

template<class Tuple, int Size>
struct TupleOpsJs<Tuple, Size, Size> {

    static void unpack(v8::Local<v8::Array> & array,
                       const Tuple & tuple)
    {
    }

    static void pack(v8::Array & array,
                     Tuple & tuple)
    {
    }
};

template<typename... Args>
void to_js(JSValue & val, const std::tuple<Args...> & v)
{
    v8::HandleScope scope;
    v8::Local<v8::Array> arr(v8::Array::New(sizeof...(Args)));
    TupleOpsJs<std::tuple<Args...>, 0, sizeof...(Args)>::unpack(arr, v);
    val = scope.Close(arr);
}

template<typename... Args>
std::tuple<Args...>
from_js(const JSValue & val, const std::tuple<Args...> * v = 0)
{
    if (!val->IsArray())
        throw ML::Exception("invalid JSValue for tuple extraction");

    auto arrPtr = v8::Array::Cast(*val);

    std::tuple<Args...> result;
    TupleOpsJs<std::tuple<Args...>, 0, sizeof...(Args)>::pack(*arrPtr, result);
    return result;
}

/** Class to deal with passing JS arguments, either from an Arguments
    structure or from an array of values.
*/
struct JSArgs {
    JSArgs(const v8::Arguments & args)
        : This(args.This()), args1(&args), args2(0), argc(args.Length())
    {
    }

    JSArgs(const v8::Handle<v8::Object> & This,
           int argc, const v8::Handle<v8::Value> * argv)
        : This(This), args1(0), args2(argv), argc(argc)
    {
    }

    v8::Handle<v8::Value> operator [] (unsigned index) const
    {
        if (index >= argc)
            return v8::Undefined();

        if (args1) return (*args1)[index];
        else return args2[index];
    }

    unsigned Length() const { return argc; }

    v8::Handle<v8::Object> Holder() const
    {
        if (args1) return args1->Holder();
        return This;
    }

    v8::Handle<v8::Function> Callee() const
    {
        if (args1) return args1->Callee();
        return v8::Handle<v8::Function>();
    }

    v8::Handle<v8::Object> This;
    const v8::Arguments * args1;
    const v8::Handle<v8::Value> * args2;
    unsigned argc;
};


/** Convert the given value into a persistent v8 function. */
v8::Persistent<v8::Function>
from_js(const JSValue & val, v8::Persistent<v8::Function> * = 0);

/** Same, but for a local version */
v8::Handle<v8::Function>
from_js(const JSValue & val, v8::Handle<v8::Function> * = 0);

/** Same, but for a local version */
v8::Local<v8::Function>
from_js(const JSValue & val, v8::Local<v8::Function> * = 0);

v8::Handle<v8::Array>
from_js(const JSValue & val, v8::Handle<v8::Array> * = 0);

template<typename T>
std::vector<T>
from_js(const JSValue & val, const std::vector<T> * = 0)
{
    if(!val->IsArray()) {
        throw ML::Exception("invalid JSValue for vector extraction");
    }

    std::vector<T> result;
    auto arrPtr = v8::Array::Cast(*val);
    for(int i=0; i<arrPtr->Length(); ++i)
    {
        result.push_back( from_js(JSValue(arrPtr->Get(i)), (T *)0) );
    }
    return result;
}

template<typename T>
std::set<T>
from_js(const JSValue & val, const std::set<T> * = 0)
{
    if(!val->IsArray()) {
        throw ML::Exception("invalid JSValue for set extraction");
    }

    std::set<T> result;
    auto arrPtr = v8::Array::Cast(*val);
    for(int i=0; i<arrPtr->Length(); ++i)
    {
        result.insert( from_js(JSValue(arrPtr->Get(i)), (T *)0) );
    }
    return result;
}

template<typename T>
void from_js(const JSValue & jsval, const T * value,
             typename boost::enable_if<typename boost::is_same<T, void>::type, void *>::type = 0)
{
}

template<typename T>
void from_js(const JSValue & jsval, T * value,
             typename boost::enable_if<typename boost::is_same<T, void>::type, void *>::type = 0)
{
}

template<typename T, typename U>
ML::distribution<T, U>
from_js_ref(const JSValue & val, ML::distribution<T, U> * = 0)
{
    return from_js(val, (const ML::distribution<T, U> *)0);
}

template<typename V8Value>
void to_js(JSValue & val, const v8::Handle<V8Value> & val2)
{
    val = val2;
}

template<typename V8Value>
void to_js(JSValue & val, const v8::Local<V8Value> & val2)
{
    val = val2;
}

template<typename V8Value>
void to_js(JSValue & val, const v8::Persistent<V8Value> & val2)
{
    val = val2;
}

template<typename T, size_t I, typename Sz, bool Sf, typename P, class A>
void to_js(JSValue & val, const ML::compact_vector<T, I, Sz, Sf, P, A> & v)
{
    v8::HandleScope scope;

    v8::Local<v8::Array> arr(v8::Array::New(v.size()));
    for (unsigned i = 0;  i < v.size();  ++i)
        arr->Set(v8::Uint32::New(i), toJS(v[i]));
    val = scope.Close(arr);
}

//template<typename T>
//void to_js(JSValue & val, T * const &)
//{
//    to_js(val, (T *)0);
//}

//void from_js(JSValue & jsval, const void * = 0);
//void from_js(const JSValue & jsval, void * = 0);

struct ValuePromise {
    ValuePromise()
        : argnum(-1)
    {
    }

    ValuePromise(const JSValue & value)
        : value(value), argnum(-1)
    {
    }

    ValuePromise(const JSValue & value,
                 const std::string & name,
                 int argnum)
        : value(value), name(name), argnum(argnum)
    {
    }

    JSValue value;
    std::string name;
    int argnum;

    template<typename T>
    operator T () const
    {
        try {
            return from_js(this->value, (T *)0);
        } catch (const std::exception & exc) {
            if (argnum == -1)
                throw ML::Exception("value \"%s\" could not be "
                                    "converted to a %s: %s",
                                    cstr(this->value).c_str(),
                                    ML::type_name<T>().c_str(),
                                    exc.what());
            throw ML::Exception("argument %d (%s): value \"%s\" could not be "
                                "converted to a %s: %s",
                                this->argnum, this->name.c_str(),
                                cstr(this->value).c_str(),
                                ML::type_name<T>().c_str(),
                                exc.what());
        }
    }

    template<typename T>
    decltype(from_js_ref(*(JSValue *)0, (T *)0)) getRef() const
    {
        try {
            return from_js_ref(this->value, (T *)0);
        } catch (const std::exception & exc) {
            if (argnum == -1)
                throw ML::Exception("value \"%s\" could not be "
                                    "converted to a %s: %s",
                                    cstr(this->value).c_str(),
                                    ML::type_name<T>().c_str(),
                                    exc.what());
            throw ML::Exception("argument %d (%s): value \"%s\" could not be "
                                "converted to a %s: %s",
                                this->argnum, this->name.c_str(),
                                cstr(this->value).c_str(),
                                ML::type_name<T>().c_str(),
                                exc.what());
        }
    }
};

ValuePromise getArg(const JSArgs & args, int argnum,
                    const std::string & name);


std::string
getArg(const JSArgs & args, int argnum, const std::string & defvalue,
       const std::string & name);

template<typename T, typename A>
T getArg(const JSArgs & args, int argnum,
         const std::string & name,
         T (*fn) (A))
{
    ValuePromise vp = getArg(args, argnum, name);
    return fn(vp.value);
}

template<typename T>
T getArg(const JSArgs & args, int argnum, const T & defvalue,
         const std::string & name)
{
    if (args.Length() <= argnum)
        return defvalue;

    return getArg(args, argnum, name);
}

template<typename T>
typename boost::remove_reference<T>::type
getArg(const JSArgs & args, int argnum, const std::string & name,
       typename boost::disable_if<typename boost::is_abstract<typename boost::remove_reference<T>::type>::type>::type * = 0)
{
    return getArg(args, argnum, name)
        .operator typename boost::remove_reference<T>::type ();
}

template<typename T>
decltype(from_js_ref(*(JSValue *)0, (T *)0))
getArg(const JSArgs & args, int argnum, const std::string & name,
       typename boost::enable_if<typename boost::is_abstract<typename boost::remove_reference<T>::type>::type>::type * = 0)
{
    return getArg(args, argnum, name).getRef<T>();
}

/** Funky template voodoo to handle a read/write property stored in a plain
    member variable.  To be instantiated in the Initialize() function to
    generate and add the correct functions to make read/write properties
    accessible.

    Usage:

    // in cpp

    struct MyObject {
        MyObject(string name = "", int value = 0)
            : name(name), value(value)
        {
        }

        string name;
        int value;
    };

    static const char * MyObjectName = "MyObject";

    struct MyObjectJS
        : public JSWrapped2<MyObject, MyObjectJS, MyObjectName> {

        IntHandlerJS(const Arguments & args)
        {
            wrap(args.This(), new MyObject());
        }


        static Handle<v8::Value>
        New(const Arguments & args)
        {
            try {
                new MyObjectJS(args);
                return args.This();
            } HANDLE_JS_EXCEPTIONS;
        }

        static void Initialize()
        {
            Persistent<FunctionTemplate> t = Register();
            RWPropertyHandler<MyObjectJS, MyObject, int, &MyObjectJS::value>
                handleValue(t, "value");
            RWPropertyHandler<MyObjectJS, MyObject, string, &MyObjectJS::name>
                handleName(t, "name");
        }
    };

    // in js
    
    var obj = new MyObjectJS();

    sys.puts(obj.name);  // returns ""
    sys.puts(obj.value);  // returns 0
    obj.name = "hello";
    obj.value = 4;
    sys.puts(obj.name);  // returns "hello"
    sys.puts(obj.value);  // returns 4
    obj.value = "four";  // throws as not convertible to an int
*/
template<typename Base, typename Shared, typename T, T Shared::* Member,
         unsigned options = v8::ReadOnly | v8::DontDelete>
struct ROPropertyHandler {

    ROPropertyHandler(v8::Persistent<v8::FunctionTemplate> t,
                      const char * name)
    {
        t->InstanceTemplate()
            ->SetAccessor(v8::String::NewSymbol(name), getter, 0,
                          v8::Handle<v8::Value>(), v8::DEFAULT,
                          v8::PropertyAttribute(options));
    }

    static v8::Handle<v8::Value>
    getter(v8::Local<v8::String> property,
           const v8::AccessorInfo & info)
    {
        try {
            return toJS((*Base::getShared(info.This())).*Member);
        } HANDLE_JS_EXCEPTIONS;
    }
};

template<typename Base, typename Shared, typename T, T (Shared::* Fn) () const,
         unsigned options = v8::ReadOnly | v8::DontDelete>
struct GetterHandler {

    GetterHandler(v8::Persistent<v8::FunctionTemplate> t,
                    const char * name)
    {
        t->InstanceTemplate()
            ->SetAccessor(v8::String::NewSymbol(name), getter, 0,
                          v8::Handle<v8::Value>(), v8::DEFAULT,
                          v8::PropertyAttribute(options));
    }

    static v8::Handle<v8::Value>
    getter(v8::Local<v8::String> property,
           const v8::AccessorInfo & info)
    {
        try {
            return toJS(((*Base::getShared(info.This())).*Fn) ());
        } HANDLE_JS_EXCEPTIONS;
    }
};

template<typename Base, typename Shared, typename T, T Shared::* Member,
         unsigned options = v8::DontDelete>
struct RWPropertyHandler {

    RWPropertyHandler(v8::Persistent<v8::FunctionTemplate> t,
                      const char * name)
    {
        t->InstanceTemplate()
            ->SetAccessor(v8::String::NewSymbol(name), getter, setter,
                          v8::Handle<v8::Value>(), v8::DEFAULT,
                          v8::PropertyAttribute(options));
    }

    static v8::Handle<v8::Value>
    getter(v8::Local<v8::String> property,
           const v8::AccessorInfo & info)
    {
        try {
            return toJS((*Base::getShared(info.This())).*Member);
        } HANDLE_JS_EXCEPTIONS;
    }
    
    static void
    setter(v8::Local<v8::String> property,
           v8::Local<v8::Value> value,
           const v8::AccessorInfo & info)
    {
        try {
            *Base::getShared(info.This()).*Member = from_js(JSValue(value), (T *)0);
        } catch (...) {
            std::cerr << "error setting field" << std::endl;
            throw;
        }
    }
    
};


inline ValuePromise
fromJS(const v8::Handle<v8::Value> & value)
{
    return ValuePromise(value);
}

#if 0
template<typename T>
std::shared_ptr<T>
from_js(const v8::Handle<v8::Value> & value,
        const std::shared_ptr<T> *)
{
    return registry.getObject(value);
}
#endif

/** Turn a string containing Javascript source into a function.  The source
    file has to be of the following form:

    fn = getFunction("function f1(val) { return val; };  f1;");

    Note the extra "f1" at the end, which provides the return value of the
    script (which is copied into the function).

    An exception will be thrown on any error; on success the returned
    handle is guaranteed to be a valid handle to a function.

    Note that this function can only run within a v8 context.
*/
v8::Handle<v8::Function>
getFunction(const std::string & script_source);

/** Turn a string containing Javascript source into a function.  The source
    file has to be of the following form:

    fn = getFunction("function f1(val) { return val; };  f1;");

    Note the extra "f1" at the end, which provides the return value of the
    script (which is copied into the function).

    An exception will be thrown on any error; on success the returned
    handle is guaranteed to be a valid handle to a function.

    Note that this function can only run within a v8 context.

    The given global object is used.
*/
v8::Handle<v8::Function>
getFunction(const std::string & script_source,
            v8::Handle<v8::Object> global);



// Convert a member pointer to a value
template<typename T, typename Obj>
v8::Local<v8::Value> pmToValue(T (Obj::* ptr))
{
    BOOST_STATIC_ASSERT(sizeof(T (Obj::*)) == sizeof(void *));

    union {
        void * vptr;
        size_t sz;
        T (Obj::* ptr);
    } x;
    
    x.ptr = ptr;
    
    ExcAssert(x.sz <= sizeof(Obj));

    return v8::External::Wrap(x.vptr);
}

// Convert a value back into a member pointer
template<typename T, typename Obj>
T Obj::* valueToPm(v8::Handle<v8::Value> val)
{
    union {
        void * vptr;
        size_t sz;
        T (Obj::* ptr);
    } x;
    
    x.vptr = v8::External::Unwrap(val);

    ExcAssert(x.sz <= sizeof(Obj));

    return x.ptr;
}

// Note: these will be leaked...
// Holds a pointer to a member function AND a set of default argument values
// to use when not enough are supplied.
template<typename T, typename Obj, typename... Args>
struct PmfInfo {
    template<typename... Defaults>
    PmfInfo(T (Obj::* pmf) (Args...),
              Defaults... defaults)
        : pmf(pmf), defaultArgs(sizeof...(Args))
    {
        //using namespace std;
        //cerr << "adding " << sizeof...(Defaults) << " default args to "
        //     << sizeof...(Args) << " existing" << endl;
        addDefaults(sizeof...(Args) - sizeof...(Defaults),
                    sizeof...(Args), defaults...);
    }
    
    template<typename Arg1, typename... Rest>
    void addDefaults(int argNum, size_t numArgs, Arg1 arg1, Rest... rest)
    {
        addDefault(arg1, argNum, numArgs);
        addDefaults(argNum + 1, numArgs, rest...);
    }

    // End of recursion; we should have reached the end of them both
    void addDefaults(int argNum, size_t numArgs)
    {
        ExcAssertEqual(argNum, numArgs);
    }

    template<typename X>
    void addDefault(X arg, int argNum, size_t numArgs)
    {
        defaultArgs[argNum] = v8::Persistent<v8::Value>::New(JS::toJS(arg));
    }

    T (Obj::* pmf) (Args...);
    std::vector<JSValue> defaultArgs;
};

// Convert a member pointer to a value
template<typename T, typename Obj, typename... Args, typename... Defaults>
v8::Local<v8::Value> pmfToValue(T (Obj::* pmf) (Args...),
                                Defaults... defaults)
{
    PmfInfo<T, Obj, Args...> * res
        = new PmfInfo<T, Obj, Args...>(pmf, defaults...);
    return v8::External::Wrap(res);
}

template<typename T, typename Obj, typename... Args, typename... Defaults>
v8::Local<v8::Value> pmfToValue(T (Obj::* pmf) (Args...) const,
                                Defaults... defaults)
{
    PmfInfo<T, const Obj, Args...> * res
        = new PmfInfo<T, const Obj, Args...>(pmf, defaults...);
    return v8::External::Wrap(res);
}

template<typename T, typename Obj, typename... Args>
struct PmfReturnValue {
    typedef T (Obj::* type) (Args...);
};

// Convert a value back into a member pointer
template<typename T, typename Obj, typename... Args>
//T (Obj::*) (Args...)
typename PmfReturnValue<T, Obj, Args...>::type
valueToPmf(v8::Handle<v8::Value> val)
{
    return reinterpret_cast<PmfInfo<T, Obj, Args...> *>(v8::External::Unwrap(val))
        ->pmf;
}

template<typename T, typename Obj, typename... Args>
const PmfInfo<T, Obj, Args...> *
valueToPmfInfo(v8::Handle<v8::Value> val)
{
    return reinterpret_cast<PmfInfo<T, Obj, Args...> *>(v8::External::Unwrap(val));
}

template<typename T, typename... Args>
v8::Local<v8::Value>
lambdaToValue(const boost::function<T (Args...)> & lambda)
{
    // TODO: this memory will be leaked...
    boost::function<T (Args...)> * fn
        = new boost::function<T (Args...)>(lambda);
    return v8::External::Wrap(fn);
}

template<typename T, typename... Args>
const boost::function<T (Args...)> &
valueToLambda(const v8::Handle<v8::Value> & val)
{
    return *reinterpret_cast<boost::function<T (Args...)> *>
        (v8::External::Unwrap(val));
}

template<typename Obj, typename Base, typename RT>
static v8::Handle<v8::Value>
lambdaGetter(v8::Local<v8::String> property,
             const v8::AccessorInfo & info)
{
    try {
        boost::function<RT (const Obj &)> fn
            = valueToLambda<RT, const Obj &>(info.Data());
        Obj & o = *Base::getShared(info.This());
        RT value = fn(o);
        return JS::toJS(value);
    } HANDLE_JS_EXCEPTIONS;
}

template<typename T, typename Obj, typename Base>
struct PropertyGetter {
    static v8::Handle<v8::Value>
    getter(v8::Local<v8::String> property,
           const v8::AccessorInfo & info)
    {
        try {
            T (Obj::* pm) = valueToPm<T, Obj>(info.Data());
            Obj & o = *Base::getShared(info.This());
            T value = o.*pm;
            return JS::toJS(value);
        } HANDLE_JS_EXCEPTIONS;
    }

    static v8::Handle<v8::Value>
    pmfGetter(v8::Local<v8::String> property,
              const v8::AccessorInfo & info)
    {
        try {
            auto pmf = valueToPmf<T, Obj>(info.Data());
            Obj & o = *Base::getShared(info.This());
            T value = (o.*pmf) ();
            return JS::toJS(value);
        } HANDLE_JS_EXCEPTIONS;
    }

};


template<typename T, typename Obj, typename Base>
struct PropertySetter {
    static void
    setter(v8::Local<v8::String> property,
           v8::Local<v8::Value> value,
           const v8::AccessorInfo & info)
    {
        try {
            T (Obj::* pm) = valueToPm<T, Obj>(info.Data());
            Obj & o = *Base::getShared(info.This());
            T & var = o.*pm;
            var = from_js(JSValue(value), (T *)0);
        } HANDLE_JS_EXCEPTIONS_SETTER;
    }

    static void
    pmfSetter(v8::Local<v8::String> property,
              v8::Local<v8::Value> value,
              const v8::AccessorInfo & info)
    {
        try {
            void (Obj::* setter) (const T &) = valueToPmf<void, Obj, const T &>
                (info.Data());
            Obj & o = *Base::getShared(info.This());
            (o.*setter) (from_js(JSValue(value), (const T *)0));
        } HANDLE_JS_EXCEPTIONS;
    }
};

// We make from_js_ref make a temporary copy of anything that's just a POD
// field; anything else needs to a) have a from_js_ref specialization, or
// b) be extractable as a pointer to avoid copying
template<typename T>
T
from_js_ref(const JSValue & val, T *,
            typename boost::enable_if<typename boost::is_pod<T>::type>::type * = 0)
{
    return from_js(val, (T*)0);
}

// Allow std::string to be passed by value as well

inline std::string
from_js_ref(const JSValue & val, std::string *)
{
    return from_js(val, (std::string *)0);
}

// And vectors

template<typename T>
inline std::vector<T>
from_js_ref(const JSValue & val, std::vector<T> *)
{
    return from_js(val, (std::vector<T> *)0);
}

// Anything else we require that we can extract a pointer to a real object
// to avoid copying complex objects; this can be overridden by defining a
// function like for std::string above
template<typename T>
const T &
from_js_ref(const JSValue & val, T *,
            typename boost::disable_if<typename boost::is_pod<T>::type>::type * = 0)
{
    return *from_js(val, (T**)0);
}

using ML::TypeList;
using ML::InPosition;
using ML::MakeInPositionList;

// Template that, given an InPosition<Arg, Index> argument, will actually
// extract the argument from a JS::JsArgs and pass it on
template<typename Param>
struct CallWithJsArgs {
};

// Implementation of that template with the InPosition argument unpacked
template<typename Arg, int Index>
struct CallWithJsArgs<InPosition<Arg, Index> > {

    static Arg
    getArgAtPosition(const JS::JSArgs & args,
                     const std::vector<JSValue> & defaults = std::vector<JSValue>())
    {
        try {
            if (Index >= args.Length() && Index < defaults.size()
                && !defaults[Index].IsEmpty()) {
                //using namespace std;
                //cerr << "defaults.size() = " << defaults.size() << endl;
                //cerr << "using default value " << cstr(defaults.at(Index))
                //     << " for arg " << Index << endl;
                //cerr << "&defaults[0] = " << &defaults[0] << endl;
                return from_js(defaults[Index], (Arg *)0);
            }
            return from_js(JSValue(args[Index]), (Arg *)0);
        } catch (const std::exception & exc) {
            throw ML::Exception("calling %s.%s(): argument %d (%s): "
                                "value \"%s\" could not be "
                                "converted to a %s: %s",
                                cstr(args.Holder()->GetConstructorName()).c_str(),
                                cstr(args.Callee()->GetName()).c_str(),
                                Index, "name",
                                cstr(args[Index]).c_str(),
                                ML::type_name<Arg>().c_str(),
                                exc.what());
        }
    }
};

template<typename Arg, int Index>
struct CallWithJsArgs<InPosition<const Arg &, Index> > {

    static decltype(from_js_ref(*(JSValue *)0, (Arg *)0))
    getArgAtPosition(const JS::JSArgs & args,
                     const std::vector<JSValue> & defaults = std::vector<JSValue>())
    {
        //using namespace std;
        //cerr << "getting arg " << argNum << " for reference" << endl;
        //cerr << "value is " << cstr(args[argNum]) << endl;
        std::string message;
        try {
            if (Index >= args.Length() && Index < defaults.size()
                && !defaults[Index].IsEmpty())
                return from_js_ref(defaults[Index], (Arg *)0);
            JSValue v(args[Index]);
            return from_js_ref(v, (Arg *)0);
        } catch (const std::exception & exc) {
            message
                = ML::format("calling %s.%s(): argument %d: "
                             "value \"%s\" could not be "
                             "converted to a %s: %s",
                             cstr(args.Holder()->GetConstructorName()).c_str(),
                             cstr(args.Callee()->GetName()).c_str(),
                             Index,
                             cstr(args[Index]).c_str(),
                             ML::type_name<Arg>().c_str(),
                             exc.what());
        }
        //cerr << "got message: " << message << endl;
        throw ML::Exception(message);
    }
};

// Given a boost::function type Fn and a TypeList of InPosition values,
// this calls the function with the JS arguments unpacked
template<typename List>
struct CallPmfWithTypePositionList {
};

// Implementation of that template with the List argument unpacked
template<typename... ArgsWithPosition>
struct CallPmfWithTypePositionList<TypeList<ArgsWithPosition...> > {

    template<typename R, typename... Args, typename Obj>
    static R call(R (Obj::* pmf) (Args...), Obj & obj,
                  const JS::JSArgs & args,
                  const std::vector<JSValue> & defaults = std::vector<JSValue>())
    {
        return (obj.*pmf)(CallWithJsArgs<ArgsWithPosition>
                          ::getArgAtPosition(args, defaults)...);
    }

    template<typename... Args, typename Obj>
    static void call(void (Obj::* pmf) (Args...), Obj & obj,
                     const JS::JSArgs & args,
                     const std::vector<JSValue> & defaults = std::vector<JSValue>())
    {
        (obj.*pmf)(CallWithJsArgs<ArgsWithPosition>
                   ::getArgAtPosition(args, defaults)...);
    }

    template<typename R, typename... Args, typename Obj>
    static R call(R (Obj::* pmf) (Args...) const, const Obj & obj,
                  const JS::JSArgs & args,
                  const std::vector<JSValue> & defaults = std::vector<JSValue>())
    {
        return (obj.*pmf)(CallWithJsArgs<ArgsWithPosition>
                          ::getArgAtPosition(args, defaults)...);
    }
    
    template<typename... Args, typename Obj>
    static void call(void (Obj::* pmf) (Args...) const, const Obj & obj,
                     const JS::JSArgs & args,
                     const std::vector<JSValue> & defaults = std::vector<JSValue>())
    {
        (obj.*pmf)(CallWithJsArgs<ArgsWithPosition>
                   ::getArgAtPosition(args, defaults)...);
    }
};


template<typename Obj, typename... Args>
v8::Handle<v8::Value>
callPmf(void (Obj::*pmf) (Args...) const, const Obj & obj,
        const v8::Arguments & args,
        const std::vector<JSValue> & defaults = std::vector<JSValue>())
{
    typedef typename MakeInPositionList<0, Args...>::List TypePositionList;
    CallPmfWithTypePositionList<TypePositionList>
        ::call(pmf, obj, args, defaults);
    return args.This();
}

template<typename Obj, typename... Args>
v8::Handle<v8::Value>
callPmf(void (Obj::*pmf) (Args...), Obj & obj,
        const v8::Arguments & args,
        const std::vector<JSValue> & defaults = std::vector<JSValue>())
{
    typedef typename MakeInPositionList<0, Args...>::List TypePositionList;
    CallPmfWithTypePositionList<TypePositionList>
        ::call(pmf, obj, args, defaults);
    return args.This();
}

template<typename Obj, typename R, typename... Args>
v8::Handle<v8::Value>
callPmf(R (Obj::*pmf) (Args...) const, const Obj & obj,
        const v8::Arguments & args,
        const std::vector<JSValue> & defaults = std::vector<JSValue>())
{
    typedef typename MakeInPositionList<0, Args...>::List TypePositionList;
    return JS::toJS(CallPmfWithTypePositionList<TypePositionList>
                    ::call(pmf, obj, args, defaults));
}

template<typename Obj, typename R, typename... Args>
v8::Handle<v8::Value>
callPmf(R (Obj::*pmf) (Args...), Obj & obj,
        const v8::Arguments & args,
        const std::vector<JSValue> & defaults = std::vector<JSValue>())
{
    typedef typename MakeInPositionList<0, Args...>::List TypePositionList;
    return JS::toJS(CallPmfWithTypePositionList<TypePositionList>
                    ::call(pmf, obj, args, defaults));
}

template<typename R, typename Obj, typename Base, typename... Args>
struct MemberFunctionCaller {

    static v8::Handle<v8::Value>
    call(const v8::Arguments & args)
    {
        try {
            auto info = valueToPmfInfo<R, Obj, Args...>(args.Data());
            Obj & o = *Base::getShared(args);
            v8::Handle<v8::Value> result = callPmf(info->pmf, o, args,
                                                   info->defaultArgs);
            return result;
        } HANDLE_JS_EXCEPTIONS;
    }
};


template<typename R, typename Obj, typename Base>
struct LambdaCaller {

    static v8::Handle<v8::Value>
    call(const v8::Arguments & args)
    {
        try {
            const boost::function<R (Obj &, const v8::Arguments &)> & fn
                = valueToLambda<R, Obj &, const v8::Arguments &>(args.Data());
            Obj & o = *Base::getShared(args);
            return JS::toJS(fn(0, args));
        } HANDLE_JS_EXCEPTIONS;
    }
};

template<typename Obj, typename Base>
struct LambdaCaller<void, Obj, Base> {

    static v8::Handle<v8::Value>
    call(const v8::Arguments & args)
    {
        try {
            const boost::function<void (Obj &, const v8::Arguments &)> & fn
                = valueToLambda<void, Obj &,
                                const v8::Arguments &>(args.Data());
            Obj & o = *Base::getShared(args);
            fn(0, args);
            return args.This();
        } HANDLE_JS_EXCEPTIONS;
    }
};

/** Call a getter function that's in the data field of the given object. */
v8::Handle<v8::Value>
callGetterFn(v8::Local<v8::String> property,
             const v8::AccessorInfo & info);


/*****************************************************************************/
/* CALL IN JS CONTEXT                                                        */
/*****************************************************************************/

/** Arrange for the given function to be called from the JS thread eventually.

    Puts the callback on the libev loop that Node uses internally for this
    task.
*/
void callInJsThread(const boost::function<void ()> & fn);

/** Used to set up a callback to something that must be in JS from something
    that may or may not be in the JS context.
*/

template<typename Fn>
void callInJsContext(const Fn & fn)
{
    // If we happen to already be in the JS interpreter then we can simply call
    // the functio from this thread.
    if (v8::Locker::IsLocked()) {
        fn();
        return;
    }
    
    callInJsThread(fn);
}

/** This callback, no matter which thread it is called from, will call the
    given v8 function with the given This pointer from a JS context.  Use
    when you need to give a JS callback to some C++ code that might be called
    outside of the Javascript thread.
*/
boost::function<void ()>
createCrossThreadCallback(v8::Handle<v8::Function> fn,
                          v8::Handle<v8::Object> This);

boost::function<void ()>
createCrossThreadCallback(v8::Handle<v8::Function> fn,
                          v8::Handle<v8::Object> This,
                          v8::Handle<v8::Value> arg1);

boost::function<void ()>
createCrossThreadCallback(v8::Handle<v8::Function> fn,
                          v8::Handle<v8::Object> This,
                          v8::Handle<v8::Value> arg1,
                          v8::Handle<v8::Value> arg2);

boost::function<void ()>
createCrossThreadCallback(v8::Handle<v8::Function> fn,
                          v8::Handle<v8::Object> This,
                          v8::Handle<v8::Value> arg1,
                          v8::Handle<v8::Value> arg2,
                          v8::Handle<v8::Value> arg3);

// Convert a callback to be called in JS context

template<typename R, typename... Args>
boost::function<void (Args...)>
createAsyncJsCallback(const boost::function<R (Args...)> & fn)
{
    auto newFn = [=] (Args... args)
        {
            boost::function<void ()> cb
                = std::bind<void>(fn, args...);

            callInJsContext(cb);
        };

    return newFn;
}

template<typename T, typename Obj, typename Base>
struct AsyncCallbackSetter {
    static void
    setter(v8::Local<v8::String> property,
           v8::Local<v8::Value> value,
           const v8::AccessorInfo & info)
    {
        try {
            T (Obj::* pm) = valueToPm<T, Obj>(info.Data());
            Obj & o = *Base::getShared(info.This());
            T & var = o.*pm;
            var = createAsyncJsCallback(from_js(JSValue(value), (T *)0));
        } HANDLE_JS_EXCEPTIONS_SETTER;
    }
};


// Returns an array with numbers from 0 to sz-1 that can be used as
// the indexed array accessor to list entries.
v8::Handle<v8::Array>
getIndexArray(size_t sz);

// Print out a JS object including it's entire prototype chain
void printObj(const v8::Handle<v8::Value> & val,
              std::ostream & stream,
              int nesting = 0);

template<typename Val>
inline
std::ostream & operator << (std::ostream & stream,
                            const v8::Handle<Val> & val)
{
    printObj(val, stream, 0);
    return stream;
}

} // namespace JS
} // namespace Datacratic
