/* js_wrapped.h                                                    -*- C++ -*-
   Jeremy Barnes, 15 July 2010
   Copyright (c) 2010 Datacratic Inc.  All rights reserved.

   Some code is from node_object_wrap.h, from node.js.  Its license is
   included here:

   Copyright 2009, 2010 Ryan Lienhart Dahl. All rights reserved.
   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to
   deal in the Software without restriction, including without limitation the
   rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
   sell copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:
   
   The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.
   
   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
   IN THE SOFTWARE. 
*/

#pragma once

#include "jml/utils/exc_assert.h"
#include "jml/arch/demangle.h"
#include "jml/compiler/compiler.h"
#include "jml/utils/string_functions.h"
#include "jml/arch/backtrace.h"
#include "jml/utils/smart_ptr_utils.h"
#include <node/node.h>
#include <v8/v8.h>
#include <boost/shared_ptr.hpp>
#include <boost/type_traits/is_base_of.hpp>
#include <boost/static_assert.hpp>
#include <boost/utility/enable_if.hpp>
#include <iostream>
#include "js_utils.h"
#include "js_registry.h"


namespace Datacratic {
namespace JS {


/** Return the std::type_info node for the wrapper class for a given
    object. */
const std::type_info & getWrapperTypeInfo(v8::Handle<v8::Value> handle);


/*****************************************************************************/
/* JSWRAPPEDBASE                                                             */
/*****************************************************************************/

class JSWrappedBase {
public:
    JSWrappedBase()
        : refs_(0), size_in_bytes_(0)
    {
    }
    
    virtual ~JSWrappedBase();

    template <class T>
    static inline T * unwrap(const v8::Handle<v8::Object> & handle)
    {
        ExcAssert(!handle.IsEmpty());
        ExcAssert(handle->InternalFieldCount() == 2);
        return static_cast<T*>(v8::Handle<v8::External>::Cast
                               (handle->GetInternalField(0))->Value());
    }

    v8::Persistent<v8::Object> js_object_;

    /** If there is an object that owns this and must stay alive for this
        JS object to be valid, then it goes here.
    */
    std::shared_ptr<void> owner_;

protected:
    /** Set up the object by making handle contain an external reference
        to the given object. */
    void wrap(v8::Handle<v8::Object> handle,
              size_t object_size,
              const std::type_info & wrappedType);

    /** Set this object up to be garbage collected once there are no more
        references to it in the javascript. */
    void registerForGarbageCollection();
    
    /* Ref() marks the object as being attached to an event loop.
     * Refed objects will not be garbage collected, even if
     * all references are lost.
     */
    virtual void ref();
    
    /* Unref() marks an object as detached from the event loop.  This is its
     * default state.  When an object with a "weak" reference changes from
     * attached to detached state it will be freed. Be careful not to access
     * the object after making this call as it might be gone!
     * (A "weak reference" means an object that only has a
     * persistant handle.)
     *
     * DO NOT CALL THIS FROM DESTRUCTOR
     */
    virtual void unref();
    
    /** Method called to actually dispose of an object once everything is
        finished.  This function won't be called if deleter() is overridden.
        Default simply uses operator delete.
    */
    virtual void dispose();

    /** Method to return the deleter function.  By default, it's the
        weakCallback function below, which will delete with the dispose()
        method.  Override to use something else. */
    virtual v8::WeakReferenceCallback getGarbageCollectionCallback() const;

    virtual std::string getJsTypeName() const
    {
        return "UNKNOWN";
    }

    static void
    Setup(const v8::Persistent<v8::FunctionTemplate> & ft,
          const std::string & name,
          const v8::Handle<v8::Object> & target)
    {
        using namespace v8;

        //using namespace std;
        //cerr << "setup " << name << " ft = " << *ft << endl;

        // Get the context
        HandleScope scope;
        v8::Local<v8::Context> context = v8::Context::GetCurrent();
        v8::Local<v8::Object> global = context->Global();

        // Check for the stash
        v8::Local<v8::Value> stash
            = global->Get(v8::String::NewSymbol("__cpp_stash__"));
        v8::Local<v8::Object> stash_obj;
        if (!stash.IsEmpty())
            stash_obj = v8::Object::Cast(*stash);

        if (stash_obj.IsEmpty()
            || stash_obj->IsUndefined() || stash_obj->IsNull()) {
            //cerr << "*** Creating stash" << endl;
            v8::Local<v8::Object> new_stash = v8::Object::New();
            global->Set(v8::String::NewSymbol("__cpp_stash__"), new_stash);
            stash_obj = new_stash;
        }

        v8::Local<v8::Value> fn
            = stash_obj->Get(v8::String::NewSymbol(name.c_str()));
        v8::Local<v8::Function> f;
        if (fn.IsEmpty() || !fn->IsFunction()) {
            //cerr << "  creating function " << name << endl;
            f = ft->GetFunction();
            stash_obj->Set(v8::String::NewSymbol(name.c_str()), f);
        }
        else f = v8::Function::Cast(*fn);
        
        // Create the function from the template
        //Local<v8::Function> f = ft->GetFunction();

        // Finally, add the new object into the target object under the
        // given name
        target->Set(String::NewSymbol(name.c_str()), f);
    }

    /** Helper function that acts as a constructor for when we're creating
        an object from a C++ class extended by a Javascript class.  It
        creates a different class (which is JS compatible) and replaces the
        prototype of that class with the prototype of the desired class so
        that inheritance works as expected.
    */
    template<typename Wrapper>
    static v8::Handle<v8::Value>
    createExtendedObject(const v8::Arguments & args)
    {
        // Create the new object
        v8::Handle<v8::Value> res = Wrapper::New(args);

        // Get the function we inherit from
        v8::Local<v8::Function> fn(v8::Function::Cast(*args.Data()));
        if (fn.IsEmpty())
            throw ML::Exception("could not extract prototype from " + cstr(args.Data()));
        
        // Run the constructor
        v8::Handle<v8::Value> args2[args.Length()];
        for (unsigned i = 0;  i < args.Length();  ++i)
            args2[i] = args[i];

        fn->Call(args.This(), args.Length(), args2);

        auto newPrototype = fn->Get(v8::String::NewSymbol("prototype"));
        args.This()->SetPrototype(newPrototype);
        
        return args.This();
    }
    
    /** Helper function that can be called to implement an "extend" class
        method.  This works in concert with createExtendedObject.  See
        recommendation_js.cc for examples.
    */
    template<typename Wrapper>
    static v8::Handle<v8::Value>
    extendImpl(const v8::Arguments & args)
    {
        // called with a single argument, which is the function (class) to
        // extend from.

        // Find things to inherit with
        v8::Local<v8::Function> fn(v8::Function::Cast(*args[0]));
        if (fn.IsEmpty())
            throw ML::Exception("call to non-function for score: " + cstr(args[0]));

        // Create a new function template for our result
        v8::Local<v8::FunctionTemplate> result
            (v8::FunctionTemplate::New(&createExtendedObject<Wrapper>, fn));

        result->InstanceTemplate()->SetInternalFieldCount(2);

        if (!args[1]->IsUndefined()) {
            v8::Local<v8::String> name(v8::String::Cast(*args[1]));
            if (!name.IsEmpty()) {
                result->SetClassName(name);
            }
        }
        else {
            v8::Local<v8::String> name(v8::String::Cast(*fn->GetName()));
            if (!name.IsEmpty()) {
                result->SetClassName(name);
            }
        }

        // Make it inherit from the JS recommendation scorer
        result->Inherit(Wrapper::tmpl);

        // Finally return the fuction
        v8::Handle<v8::Value> res = result->GetFunction();

        // Make our function inherit from this result function
        //...
        auto fnPrototype = toObject(fn->Get(v8::String::NewSymbol("prototype")));
        auto resPrototype = toObject(toObject(res)->Get(v8::String::NewSymbol("prototype")));
        fnPrototype->SetPrototype(resPrototype);
    
        return res;
    }

    static v8::Persistent<v8::FunctionTemplate>
    RegisterBase(const char * name,
                 const char * module,
                 v8::InvocationCallback constructor,
                 SetupFunction setup = Setup)
    {
        using namespace v8;
        
        Persistent<FunctionTemplate> t
            = v8::Persistent<FunctionTemplate>
            ::New(FunctionTemplate::New(constructor));

        t->InstanceTemplate()->SetInternalFieldCount(2);
        t->SetClassName(v8::String::NewSymbol(name));

        registry.get_to_know(name, module, t, setup);

        return t;
    }

    
private:
    // Called back once an object is garbage collected.
    static void garbageCollectionCallback
        (v8::Persistent<v8::Value> value, void *data);

    int refs_; // ro
    size_t size_in_bytes_; ///< Because v8 wants to know for GC heuristics
};


/*****************************************************************************/
/* JSWRAPPED                                                                 */
/*****************************************************************************/

template<typename Shared>
struct JSWrapped : public JSWrappedBase {

    typedef JSWrapped root_type;
    typedef Shared root_shared_type;
    
    JSWrapped()
    {
        ++created;
    }

    virtual ~JSWrapped()
    {
        ++destroyed;
    }

    static uint64_t created, destroyed;

    virtual size_t memusage(void * object) const
    {
        return sizeof(Shared) + sizeof(*this);
    }

    void wrap(v8::Handle<v8::Object> handle, Shared * object)
    {
        //if (!object)
        //    throw ML::Exception("wrapping null pointer");
        JSWrappedBase::wrap(handle, memusage(object), typeid(*this));
        shared_.reset(object);
    }

    void wrap(v8::Handle<v8::Object> handle, std::shared_ptr<Shared> object)
    {
        //if (!object)
        //    throw ML::Exception("wrapping null pointer");
        JSWrappedBase::wrap(handle, memusage(object.get()), typeid(*this));
        setWrappedObject(object);
    }

    static inline JSWrapped *
    getWrapper(v8::Handle<v8::Value> handle,
               v8::Persistent<v8::FunctionTemplate> tmpl,
               const char * className,
               const char * module)
    {
        if (!tmpl->HasInstance(handle)) {
            //ML::backtrace();
            throw ML::Exception("we're not the right object: wanted C++ "
                                + ML::type_name<Shared>() + " got JS "
                                + cstr(handle));
        }

        JSWrapped * result
            = JSWrappedBase::unwrap<JSWrapped>(toObject(handle));
        if (!result)
            throw ML::Exception("unrwapped closed object");
        return result;
    }

    static inline JSWrapped *
    getWrapper(const v8::Arguments & args,
               v8::Persistent<v8::FunctionTemplate> tmpl,
               const char * className,
               const char * module)
    {
        return getWrapper(args.This(), className, module);
    }

    static void unwrappedException(const char * className, const char * module)
    {
        throw ML::Exception("Using close()d or partially constructed "
                            "object of type %s: either you called close() "
                            "or you used \"new %s\" when you needed "
                            "constructor arguments or to use a factory "
                            "function (in module %s)", className, className,
                            module);
    }

    static inline Shared *
    getShared(v8::Handle<v8::Value> handle,
              v8::Persistent<v8::FunctionTemplate> tmpl,
              const char * className, const char * module)
    {
        Shared * result = getWrapper(handle, tmpl, className, module)
            ->shared_.get();
        if (!result) unwrappedException(className, module);
        return result;
    }

    static inline Shared *
    getShared(const v8::Arguments & args,
              v8::Persistent<v8::FunctionTemplate> tmpl,
              const char * className, const char * module)
    {
        return getShared(args.This(), tmpl, className, module);
    }

    static inline std::shared_ptr<Shared>
    getSharedPtr(v8::Handle<v8::Value> handle,
                 v8::Persistent<v8::FunctionTemplate> tmpl,
                 const char * className, const char * module)
    {
        std::shared_ptr<Shared> result
            = getWrapper(handle, tmpl, className, module)->shared_;
        if (!result) unwrappedException(className, module);
        return result;
    }

    static inline std::shared_ptr<Shared>
    getSharedPtr(const v8::Arguments & args,
                 v8::Persistent<v8::FunctionTemplate> tmpl,
                 const char * className, const char * module)
    {
        return getSharedPtr(args.This(), tmpl, className, module);
    }

    static inline void
    setShared(v8::Handle<v8::Value> handle,
              v8::Persistent<v8::FunctionTemplate> tmpl,
              const char * className,
              const char * module,
              const std::shared_ptr<Shared> & shared)
    {
        ExcAssert(shared);
        getWrapper(handle, tmpl, className, module)->setWrappedObject(shared);
    }

    static v8::Handle<v8::Value>
    ObjectStats(const v8::Arguments & args)
    {
        v8::HandleScope scope;
        try {
            v8::Local<v8::Object> result = v8::Object::New();
            result->Set(v8::String::NewSymbol("created"),
                        v8::Integer::New(created));
            result->Set(v8::String::NewSymbol("destroyed"),
                        v8::Integer::New(destroyed));
            return scope.Close(result);
        } HANDLE_JS_EXCEPTIONS;
    }
    
    static v8::Handle<v8::Value>
    cppDetach(const v8::Arguments & args)
    {
        try {
            JSWrapped * me = JSWrappedBase::unwrap<JSWrapped>(args.This());
            me->shared_.reset();
            return v8::Undefined();
        } HANDLE_JS_EXCEPTIONS;
    }

    static v8::Handle<v8::Value>
    cppType(const v8::Arguments & args)
    {
        v8::HandleScope scope;
        try {
            JSWrapped * me = JSWrappedBase::unwrap<JSWrapped>(args.This());
            if (!me || !me->shared_)
                throw ML::Exception("cppType() called on unwrapped or closed"
                                    "object");

            //std::cerr << "args.This() = " << cstr(args.This()) << std::endl;
            //std::cerr << "me = " << me << std::endl;
            //std::cerr << "me->shared_ = " << me->shared_ << std::endl;
            //std::cerr << "type_name(*me) = " << ML::type_name(*me) << std::endl;

            return scope.Close
                (v8::String::NewSymbol(ML::type_name(*me->shared_).c_str()));
        } HANDLE_JS_EXCEPTIONS;
    }

    static v8::Handle<v8::Value>
    noConstructorError(const char * class_name = 0, const char * module = 0)
    {
        std::string msg;
        if (!class_name)
            msg = "Class does not have a constructor";
        else msg = ML::format("Class %s does not have a constructor",
                              class_name);
        return v8::ThrowException
            (injectBacktrace
             (v8::Exception::Error
              (v8::String::New(msg.c_str()))));
    }

    static v8::Handle<v8::Value>
    NoConstructor(const v8::Arguments & args)
    {
        return noConstructorError();
    }

    static v8::Persistent<v8::FunctionTemplate>
    RegisterBase(const char * name,
                 const char * module,
                 v8::InvocationCallback constructor,
                 SetupFunction setup = Setup)
    {
        using namespace v8;

        Persistent<FunctionTemplate> t
            = JSWrappedBase::RegisterBase(name, module, constructor, setup);
        
        // Class methods
        t->Set(String::NewSymbol("objectStats"),
               FunctionTemplate::New(ObjectStats));

        // Instance methods
        // Detach the JS object from its backing CPP object
        NODE_SET_PROTOTYPE_METHOD(t, "cppDetach", cppDetach);

        NODE_SET_PROTOTYPE_METHOD(t, "cppType", cppType);

        registry.get_to_know(name, module, t, setup);

        return t;
    }

    const std::shared_ptr<Shared> & getWrappedObject() const
    {
        return shared_;
    }

    /** Modify the wrapped object.  Will notify anything that has
        signed up to onSetShared or onResetShared.  It is valid to
        pass in a null pointer.
    */
    void setWrappedObject(const std::shared_ptr<Shared> & newObject)
    {
        if (newObject == shared_) return;
        if (shared_)
            onResetWrappedObject(js_object_, shared_);
        shared_ = newObject;
        if (newObject)
            onSetWrappedObject(js_object_, newObject);
    }

protected:
    /** Function that can be overridden to be notified every time a new
        shared object is associated.  Guaranteed that newObject will not
        be a null pointer.

        Used (for example) as a hook to associate array data with the v8
        indexed handlers.
    */
    virtual void
    onSetWrappedObject(const v8::Handle<v8::Object> & This,
                       const std::shared_ptr<Shared> & newObject)
    {
    }

    /** Function that can be overridden to be notified every time a new
        shared object is disassociated.  Guaranteed that oldObject will not
        be a null pointer and that onResetShared will be called one time
        for every time onSetWrapperObject is called.
    */
    virtual void
    onResetWrappedObject(const v8::Handle<v8::Object> & This,
                         const std::shared_ptr<Shared> & oldObject)
    {
    }
    
private:
    std::shared_ptr<Shared> shared_;
};

template<typename Shared>
uint64_t JSWrapped<Shared>::created = 0;

template<typename Shared>
uint64_t JSWrapped<Shared>::destroyed = 0;

struct DoInitialize {
    template<typename Fn>
    DoInitialize(Fn fn, void (*dfn) (void) = 0)
        : dfn(dfn)
    {
        fn();
    }

    void (*dfn) (void);

    ~DoInitialize()
    {
        if (dfn) dfn();
    }

    operator int () const { return (size_t)this; }
};


/*****************************************************************************/
/* JSWRAPPED2                                                                */
/*****************************************************************************/

// WARNING: the Wrapper object MUST inherit from JSWrapped2 as the FIRST
// object in its inheritence, as we don't do a dynamic cast that could
// fix up the pointers for a different configuration.
template<typename Shared, typename Wrapper,
         const char * const & ClassNameT,
         const char * const & ModuleT,
         bool defaultWrapper = true>
struct JSWrapped2 : public JSWrapped<Shared> {

    typedef Shared shared_type;
    typedef Shared base_shared_type;
    typedef JSWrapped<Shared> base_type;

    typedef base_type root_type;
    typedef Shared root_shared_type;

    static const char * const & ClassName;
    static const char * const & Module;
    
    static v8::Persistent<v8::FunctionTemplate> tmpl;

    typedef JSWrapped2<Shared, Wrapper, ClassNameT, ModuleT, defaultWrapper>
         WrapperType;

    JSWrapped2()
    {
        int x JML_UNUSED = initializer;
    }

    static void check_tmpl()
    {
        if (*tmpl) return;
        std::cerr << "&tmpl = " << &tmpl << " *tmpl = " << *tmpl << std::endl;
        throw ML::Exception("template for class " + std::string(ClassNameT)
                            + " isn't initialized");
    }

    static v8::Local<v8::Object>
    toJS(const std::shared_ptr<Shared> & shared)
    {
        check_tmpl();
        v8::Local<v8::Object> result = tmpl->GetFunction()->NewInstance();
        if (result.IsEmpty()) throw JSPassException();
        setShared(result, shared);
        return result;
    }

    static v8::Local<v8::Object>
    toJS(Shared & shared, const std::shared_ptr<void> & owner)
    {
        check_tmpl();
        v8::Local<v8::Object> result = tmpl->GetFunction()->NewInstance();
        if (result.IsEmpty()) throw JSPassException();
        setShared(result, shared, owner);
        return result;
    }

    static std::shared_ptr<Shared>
    fromJS(v8::Handle<v8::Object> obj)
    {
        return getSharedPtr(obj);
    }

    virtual size_t memusage(void * object) const
    {
        return sizeof(Shared) + sizeof(Wrapper);
    }

    static inline Wrapper * getWrapper(v8::Handle<v8::Value> handle)
    {
        check_tmpl();
        if (!tmpl->HasInstance(handle)) {
            //ML::backtrace();
            throw ML::Exception("we're not the right object: wanted C++ "
                                + ML::type_name<Shared>() + " got JS "
                                + cstr(handle));
        }

        Wrapper * result = JSWrappedBase::unwrap<Wrapper>(toObject(handle));
        if (!result)
            throw ML::Exception("unwrapped closed object");
        return result;
    }

    static inline Wrapper * getWrapper(const v8::Arguments & args)
    {
        return getWrapper(args.This());
    }

    static inline Shared *
    getShared(v8::Handle<v8::Value> handle)
    {
        return base_type::getShared(handle, tmpl, ClassName, Module);
    }

    static inline Shared *
    getShared(const v8::Arguments & args)
    {
        return base_type::getShared(args, tmpl, ClassName, Module);
    }

    static inline std::shared_ptr<Shared>
    getSharedPtr(v8::Handle<v8::Value> handle)
    {
        return base_type::getSharedPtr(handle, tmpl, ClassName, Module);
    }

    static inline std::shared_ptr<Shared>
    getSharedPtr(const v8::Arguments & args)
    {
        return base_type::getSharedPtr(args, tmpl, ClassName, Module);
    }

    static inline void
    setShared(v8::Handle<v8::Value> handle,
              const std::shared_ptr<Shared> & shared)
    {
        JSWrapped<Shared>::setShared(handle, tmpl, ClassName, Module, shared);
    }

    // setShared when we also want to record a shared pointer of an owning
    // object that guarantees that the shared object isn't deleted
    template<typename Owner>
    static inline void
    setShared(v8::Handle<v8::Value> handle,
              Shared & shared,
              const std::shared_ptr<Owner> & owner)
    {
        auto sharedPtr = ML::make_unowned_std_sp(shared);
        JSWrapped<Shared>::setShared(handle, tmpl, ClassName, Module,
                                     sharedPtr);
        getWrapper(handle)->owner_ = owner;
    }

    static v8::Handle<v8::Value>
    NoConstructor(const v8::Arguments & args)
    {
        return JSWrapped<Shared>::noConstructorError(ClassName, Module);
    }

    static v8::Handle<v8::Value>
    wrapperType(const v8::Arguments & args)
    {

        try {
            return JS::toJS(ClassName);
        } HANDLE_JS_EXCEPTIONS;
    }

    virtual std::string getJsTypeName() const
    {
        return std::string(ModuleT) + "." + std::string(ClassNameT);
    }

    static void addMethods()
    {
        using namespace v8;
        NODE_SET_PROTOTYPE_METHOD(tmpl, "wrapperType", wrapperType);
        //tmpl->Set(String::NewSymbol("wrapperType"),
        //          FunctionTemplate::New(wrapperType));
    }

    static v8::Persistent<v8::FunctionTemplate>
    Register(SetupFunction setup = JSWrapped<Shared>::Setup)
    {
        //std::cerr << "Register " << ClassName << " tmpl = "
        //          << &tmpl << std::endl;
        tmpl = JSWrapped<Shared>::
            RegisterBase(ClassName, Module, NoConstructor, setup);
        //std::cerr << "    *tmpl = " << *tmpl << std::endl;
        addMethods();
        check_tmpl();
        return tmpl;
        //return v8::Persistent<v8::FunctionTemplate>
        //    ::New(tmpl);
    }
    
    static v8::Persistent<v8::FunctionTemplate>
    Register(v8::InvocationCallback constructor,
             SetupFunction setup = JSWrapped<Shared>::Setup)
    {
        //std::cerr << "Register " << ClassName << " tmpl = "
        //          << &tmpl << std::endl;
        tmpl = JSWrapped<Shared>::
            RegisterBase(ClassName, Module, constructor, setup);
        //std::cerr << "    *tmpl = " << *tmpl << std::endl;
        addMethods();
        check_tmpl();
        return tmpl;
        //return v8::Persistent<v8::FunctionTemplate>
        //    ::New(tmpl);
    }

    /** Function used by the registry to construct this object from a
        shared pointer to a base class.  Used by the magic that allows
        a wrapper to be found by the registry for any derived class of
        a base.

        Does manipulation of the internals of a shared pointer.  Not for
        the unwary.
    */
    static v8::Local<v8::Object>
    constructMe(void * smart_ptr, const void * object)
    {
        // Change the pointer
        *(const void **)smart_ptr = object;

        // Cast to the correct type (it's guaranteed to be that)
        std::shared_ptr<Shared> & sp
            = *(std::shared_ptr<Shared> *)smart_ptr;

        return toJS(sp);
    }

    static void
    unwrapMe(const v8::Handle<v8::Value> & wrapper,
             void * outputPtr,
             const std::type_info & wrapperType)
    {
        /* Check that it is possible to convert our type into the given
           wrapper type. */
        
        throw ML::Exception("unwrapMe");
    }

    static void InitializeFunction()
    {
        registry.introduce(ClassNameT, Module, Wrapper::Initialize);
        if (defaultWrapper)
            registry.isWrapper<Shared, Wrapper>(constructMe, unwrapMe);
    }

    static void DestructionFunction()
    {
    }

    static DoInitialize initializer;

    template<typename T, typename Obj>
    static void registerRWProperty(T (Obj::* ptr), const char * name,
                                   unsigned options = v8::DontDelete)
    {
        tmpl->InstanceTemplate()
            ->SetAccessor(v8::String::NewSymbol(name),
                          PropertyGetter<T, Obj, JSWrapped2>::getter,
                          PropertySetter<T, Obj, JSWrapped2>::setter,
                          pmToValue(ptr),
                          v8::DEFAULT,
                          v8::PropertyAttribute(options));
    }

    template<typename T, typename Obj>
    static void registerROProperty(T (Obj::* ptr), const char * name,
                                   unsigned options
                                   = v8::DontDelete | v8::ReadOnly)
    {
        tmpl->InstanceTemplate()
            ->SetAccessor(v8::String::NewSymbol(name),
                          PropertyGetter<T, Obj, JSWrapped2>::getter,
                          0,
                          pmToValue(ptr),
                          v8::DEFAULT,
                          v8::PropertyAttribute(options));
    }

    /** Register a read-only property from a getter member function. */
    template<typename T, typename Obj>
    static void registerROProperty(T (Obj::* pmf) () const,
                                   const char * name,
                                   unsigned options
                                   = v8::DontDelete | v8::ReadOnly)
    {
        tmpl->InstanceTemplate()
            ->SetAccessor(v8::String::NewSymbol(name),
                          PropertyGetter<T, Obj, JSWrapped2>::pmfGetter,
                          0,
                          pmfToValue(pmf),
                          v8::DEFAULT,
                          v8::PropertyAttribute(options));
    }

#if 0
    /** Register a read-write property from a getter and setter function. */
    template<typename T, typename Obj>
    static void registerRWPropertyGetterSetter
        (T (Obj::* setter) () const,
         void (Obj::* getter) (const T &),
         const char * name,
         unsigned options = v8::DontDelete | v8::ReadOnly)
    {
        tmpl->InstanceTemplate()
            ->SetAccessor(v8::String::NewSymbol(name),
                          PropertyGetter<T, Obj, JSWrapped2>::pmfGetterPair,
                          PropertySetter<T, Obj, JSWrapped2>::pmfSetterPair,
                          setterGetterToValue(setter, getter),
                          v8::DEFAULT,
                          v8::PropertyAttribute(options));
    }
#endif

    template<typename Fn>
    static void registerROProperty(const Fn & lambda,
                                   const char * name,
                                   unsigned options
                                   = v8::DontDelete | v8::ReadOnly,
                                   decltype(lambda(*(Shared *)0))* = 0)
    {
        typedef decltype(lambda(*(Shared *)0)) RT;

        boost::function<RT (const Shared &)> fn
            = lambda;

        tmpl->InstanceTemplate()
            ->SetAccessor(v8::String::NewSymbol(name),
                          &lambdaGetter<Shared, JSWrapped2, RT>,
                          0,
                          lambdaToValue(fn),
                          v8::DEFAULT,
                          v8::PropertyAttribute(options));
    }

    static void registerROProperty(const std::string & getterFn,
                                   const char * name,
                                   unsigned options
                                   = v8::DontDelete | v8::ReadOnly)
    {
        // TODO: set this up directly by calling on the prototype so we
        // don't have a trip through C++ to deal with
        tmpl->InstanceTemplate()
            ->SetAccessor(v8::String::NewSymbol(name),
                          &callGetterFn,
                          0,
                          getFunction(getterFn),
                          v8::DEFAULT,
                          v8::PropertyAttribute(options));
    }

    static void registerROProperty(const std::string & getterFn,
                                   unsigned options
                                   = v8::DontDelete | v8::ReadOnly)
    {
        v8::Handle<v8::Function> fn = getFunction("(" + getterFn + ")");
        v8::Handle<v8::String> name = fn->GetName()->ToString();
        if (name.IsEmpty())
            throw ML::Exception("Function name was not a string");

        // TODO: set this up directly by calling on the prototype so we
        // don't have a trip through C++ to deal with
        tmpl->InstanceTemplate()
            ->SetAccessor(name,
                          &callGetterFn,
                          0,
                          fn,
                          v8::DEFAULT,
                          v8::PropertyAttribute(options));
    }

    /** This function is used in a very particular case.

        If we have a C++ object that has a data member that is a
        boost::function<...>, then calling this function will expose that
        data member as a data member in the JS object.  Assigning a JS
        function to that member will overwrite the boost::function with
        a synthisized callback that performs the following:

        1.  Check if we're currently in the thread that's active in the JS
            runtime.  If so, call the JS function immediately.
        2.  If not, then queue the JS function on libuv so that node will
            eventually call it back with the supplied parameters.  Once
            enqueued, the boost::function will then return.

        Note that the return code of the function has to be void, as there
        is no way for an asynchonous function's return code to be later
        applied at the source of the call (which has already terminated).
        Any information fedback needs to be passed through callbacks.

        Note also that to make this function work, you need to
        #include "soa/js/js_call.h"
    */

    template<typename R, typename Obj, typename... Args>
    static void registerAsyncCallback(boost::function<R (Args...)> (Obj::* ptr),
                                      const char * name,
                                      unsigned options = v8::DontDelete)
    {
        typedef boost::function<R (Args...)> T;

        tmpl->InstanceTemplate()
            ->SetAccessor(v8::String::NewSymbol(name),
                          PropertyGetter<T, Obj, JSWrapped2>::getter,
                          AsyncCallbackSetter<T, Obj, JSWrapped2>::setter,
                          pmToValue(ptr),
                          v8::DEFAULT,
                          v8::PropertyAttribute(options));
    }

    template<typename R, typename Obj, typename... Args, typename... Defaults>
    static void registerMemberFn(R (Obj::* pmf) (Args... args) const,
                                 const char * name,
                                 Defaults... defaults)
    {
        // Set it up so that the member function is called when we
        // call the JS version

        v8::Local<v8::Signature> sig = v8::Signature::New(tmpl);
        v8::Local<v8::FunctionTemplate> cb
            =  v8::FunctionTemplate::New
            (MemberFunctionCaller<R, const Obj, JSWrapped2, Args...>::call,
             pmfToValue(pmf, defaults...), sig);
        tmpl->PrototypeTemplate()->Set(v8::String::NewSymbol(name),
                                       cb);
    }

    template<typename R, typename Obj, typename... Args, typename... Defaults>
    static void registerMemberFn(R (Obj::* pmf) (Args... args),
                                 const char * name,
                                 Defaults... defaults)
    {
        // Set it up so that the member function is called when we
        // call the JS version

        v8::Local<v8::Signature> sig = v8::Signature::New(tmpl);
        v8::Local<v8::FunctionTemplate> cb
            =  v8::FunctionTemplate::New
            (MemberFunctionCaller<R, Obj, JSWrapped2, Args...>::call,
             pmfToValue(pmf, defaults...), sig);
        tmpl->PrototypeTemplate()->Set(v8::String::NewSymbol(name),
                                       cb);
    }

    static void addMemberFn(const std::string & functionSource,
                            const char * name,
                            unsigned options = v8::DontDelete | v8::ReadOnly)
    {
        v8::Handle<v8::Function> fn = getFunction("(" + functionSource + ")");
        tmpl->PrototypeTemplate()->Set(v8::String::NewSymbol(name), fn);
    }

    static void addMemberFn(const std::string & functionSource,
                            unsigned options = v8::DontDelete | v8::ReadOnly)
    {
        v8::Handle<v8::Function> fn = getFunction("(" + functionSource + ")");
        v8::Handle<v8::String> name = fn->GetName()->ToString();
        if (name.IsEmpty())
            throw ML::Exception("Function name was not a string");
        tmpl->PrototypeTemplate()->Set(name, fn);
    }

    static v8::Handle<v8::Value>
    extend(const v8::Arguments & args)
    {
        return JSWrappedBase::extendImpl<Wrapper>(args);
    }
};

template<typename Shared, typename Wrapper,
         const char * const & ClassNameT, const char * const & ModuleNameT,
         bool defaultWrapper>
const char * const &
JSWrapped2<Shared, Wrapper, ClassNameT, ModuleNameT, defaultWrapper>::
ClassName = ClassNameT;

template<typename Shared, typename Wrapper,
         const char * const & ClassNameT, const char * const & ModuleNameT,
         bool defaultWrapper>
const char * const &
JSWrapped2<Shared, Wrapper, ClassNameT, ModuleNameT, defaultWrapper>::
Module = ModuleNameT;

template<typename Shared, typename Wrapper,
         const char * const & ClassNameT, const char * const & ModuleNameT,
         bool defaultWrapper>
DoInitialize
JSWrapped2<Shared, Wrapper, ClassNameT, ModuleNameT, defaultWrapper>::
initializer(InitializeFunction, DestructionFunction);

template<typename Shared, typename Wrapper,
         const char * const & ClassNameT, const char * const & ModuleNameT,
         bool defaultWrapper>
v8::Persistent<v8::FunctionTemplate>
JSWrapped2<Shared, Wrapper, ClassNameT, ModuleNameT, defaultWrapper>::tmpl;


/*****************************************************************************/
/* JSWRAPPED3                                                                */
/*****************************************************************************/

/** A wrapped object with a base class.  This sets up the given Wrapper class
    (which must derive from JSWrapped3 in the curiously recurring template
    pattern) to wrap an object derived from that wrapped in Base.  For example,

    class Base {
        virtual ~Base();
    };
    
    class Derived : Base {
    };

    class BaseJS: public JSWrapped2<Base, BaseJS> {
    };

    class DerivedJS: public JSWrapped3<Derived, DerivedJS, BaseJS> {
    };

    The methods described in BaseJS will be available to DerivedJS via the
    prototype of DerivedJS.
*/

template<typename Shared, typename Wrapper, typename Base,
         const char * const & ClassNameT,
         const char * const & ModuleNameT,
         bool defaultWrapper = true>
struct JSWrapped3 : public Base {

    static const char * const & ClassName;
    static const char * const & Module;

    typedef Shared shared_type;
    typedef typename Base::base_shared_type base_shared_type;

    typedef typename Base::root_type root_type;
    typedef typename Base::root_shared_type root_shared_type;

    typedef boost::is_base_of<Base, Wrapper>
    WrapperIsDerivedFromBase;

    typedef boost::is_base_of<typename Base::shared_type, Shared>
    SharedIsDerivedFromBaseShared;

    static v8::Persistent<v8::FunctionTemplate> tmpl;

    typedef JSWrapped3<Shared, Wrapper, Base, ClassNameT, ModuleNameT,
                       defaultWrapper>
         WrapperType;

    JSWrapped3()
    {
        int x JML_UNUSED = initializer;
        BOOST_STATIC_ASSERT(WrapperIsDerivedFromBase::value);
        BOOST_STATIC_ASSERT(SharedIsDerivedFromBaseShared::value);
    }

    static v8::Local<v8::Object>
    toJS(const std::shared_ptr<Shared> & shared)
    {
        v8::Local<v8::Object> result = tmpl->GetFunction()->NewInstance();
        if (result.IsEmpty()) throw JSPassException();
        setShared(result, shared);
        return result;
    }

    template<typename Owner>
    static v8::Local<v8::Object>
    toJS(Shared & shared, const std::shared_ptr<Owner> & owner)
    {
        v8::Local<v8::Object> result = tmpl->GetFunction()->NewInstance();
        if (result.IsEmpty()) throw JSPassException();
        setShared(result, shared, owner);
        return result;
    }

    static std::shared_ptr<Shared>
    fromJS(v8::Handle<v8::Object> obj)
    {
        return getSharedPtr(obj);
    }

    virtual size_t memusage(void * object) const
    {
        return sizeof(Shared) + sizeof(Wrapper);
    }

    static inline Wrapper * getWrapper(v8::Handle<v8::Value> handle)
    {
        if (!tmpl->HasInstance(handle)) {
            //ML::backtrace();
            throw ML::Exception("we're not the right object: wanted C++ "
                                + ML::type_name<Shared>() + " got JS "
                                + cstr(handle));
        }

        Base * base = Base::getWrapper(handle);
        ExcAssert(base);
        Wrapper * result = dynamic_cast<Wrapper *>(base);
        if (!result)
            throw ML::Exception("1Base object " + ML::type_name<Base>()
                                + " is not an instance of "
                                + ML::type_name<Wrapper>()
                                + " (it's a " + ML::type_name(*base) + ")");
        return result;
    }

    static inline Wrapper * getWrapper(const v8::Arguments & args)
    {
        return getWrapper(args.This());
    }

    static inline Shared * getShared(v8::Handle<v8::Value> handle)
    {
        root_shared_type * base = root_type::getShared(handle, tmpl, ClassName,
                                                       Module);
        ExcAssert(base);
        Shared * result = dynamic_cast<Shared *>(base);
        if (!result)
            throw ML::Exception("2Base shared object "
                                + ML::type_name<typename Base::shared_type>()
                                + " is not an instance of "
                                + ML::type_name<Shared>()
                                + " (it's a " + ML::type_name(*base) + ")");
        return result;
    }

    static inline Shared * getShared(const v8::Arguments & args)
    {
        return getShared(args.This());
    }

    static inline std::shared_ptr<Shared>
    getSharedPtr(v8::Handle<v8::Value> handle)
    {
        std::shared_ptr<typename Base::base_shared_type> base
            = Base::getWrapper(handle)->getWrappedObject();
        if (!base)
            throw ML::Exception("unrwapped closed object");

        std::shared_ptr<Shared> result
            = std::dynamic_pointer_cast<Shared>(base);
        if (!result)
            throw ML::Exception("Base shared object "
                                + ML::type_name<typename Base::shared_type>()
                                + " is not an instance of "
                                + ML::type_name<Shared>()
                                + " (it's a " + ML::type_name(*base) + ")");
        
        return result;
    }

    static inline std::shared_ptr<Shared>
    getSharedPtr(const v8::Arguments & args)
    {
        return getSharedPtr(args.This());
    }

    static inline void
    setShared(v8::Handle<v8::Value> handle,
              const std::shared_ptr<Shared> & shared)
    {
        ExcAssert(shared);
        getWrapper(handle)->setWrappedObject(shared);
    }

    // setShared when we also want to record a shared pointer of an owning
    // object that guarantees that the shared object isn't deleted
    template<typename Owner>
    static inline void
    setShared(v8::Handle<v8::Value> handle,
              Shared & shared,
              const std::shared_ptr<Owner> & owner)
    {
        ExcAssert(shared);
        auto sharedPtr = ML::make_unowned_std_sp(shared);
        getWrapper(handle)->setWrappedObject(sharedPtr);
        getWrapper(handle)->owner_ = owner;
    }

    static v8::Handle<v8::Value>
    NoConstructor(const v8::Arguments & args)
    {
        return Base::noConstructorError(ClassName, Module);
    }

    static v8::Handle<v8::Value>
    wrapperType(const v8::Arguments & args)
    {
        try {
            return JS::toJS(ClassName);
        } HANDLE_JS_EXCEPTIONS;
    }

    static void addMethods()
    {
        using namespace v8;
        NODE_SET_PROTOTYPE_METHOD(tmpl, "wrapperType", wrapperType);
    }

    static v8::Persistent<v8::FunctionTemplate>
    Register(SetupFunction setup = Base::Setup)
    {
        return Register(NoConstructor, setup);
    }

    static v8::Persistent<v8::FunctionTemplate>
    Register(v8::InvocationCallback constructor,
             SetupFunction setup = Base::Setup)
    {
        //std::cerr << "Register " << ClassName << " tmpl = "
        //          << &tmpl << std::endl;

        using namespace v8;
        
        Persistent<FunctionTemplate> t
            = Base::RegisterBase(ClassName, Module, constructor, setup);
        
        Persistent<FunctionTemplate> base = registry[Base::ClassName];
        t->Inherit(base);

        tmpl = t;

        using namespace std;
        //cerr << "    &tmpl = " << &tmpl << " *tmpl = " << *tmpl << endl;

        addMethods();

        return tmpl;
    }

    /** Function used by the registry to construct this object from a
        shared pointer to a base class.  Used by the magic that allows
        a wrapper to be found by the registry for any derived class of
        a base.

        Does manipulation of the internals of a shared pointer.  Not for
        the unwary.
    */
    static v8::Local<v8::Object>
    constructMe(void * smart_ptr, const void * object)
    {
        // Change the pointer
        *(const void **)smart_ptr = object;

        // Cast to the correct type (it's guaranteed to be that)
        std::shared_ptr<Shared> & sp
            = *(std::shared_ptr<Shared> *)smart_ptr;

        return toJS(sp);
    }

    static void
    unwrapMe(const v8::Handle<v8::Value> & wrapper,
             void * outputPtr,
             const std::type_info & wrapperType)
    {
        /* Check that it is possible to convert our type into the given
           wrapper type. */
        
        throw ML::Exception("unwrapMe");
    }

    static void InitializeFunction()
    {
        registry.introduce(ClassNameT, ModuleNameT, Wrapper::Initialize,
                           Base::ClassName);
;
        if (defaultWrapper)
            registry.isWrapper<Shared, Wrapper>(constructMe, unwrapMe);
        registry.isBase<Base, Shared>();
    }

    static void DestructionFunction()
    {
    }

    static DoInitialize initializer;

    template<typename T, typename Obj>
    static void registerRWProperty(T (Obj::* ptr), const char * name,
                                   unsigned options = v8::DontDelete)
    {
        tmpl->InstanceTemplate()
            ->SetAccessor(v8::String::NewSymbol(name),
                          PropertyGetter<T, Obj, JSWrapped3>::getter,
                          PropertySetter<T, Obj, JSWrapped3>::setter,
                          pmToValue(ptr),
                          v8::DEFAULT,
                          v8::PropertyAttribute(options));
    }

    template<typename T, typename Obj>
    static void registerROProperty(T (Obj::* ptr), const char * name,
                                   unsigned options
                                   = v8::DontDelete | v8::ReadOnly)
    {
        tmpl->InstanceTemplate()
            ->SetAccessor(v8::String::NewSymbol(name),
                          PropertyGetter<T, Obj, JSWrapped3>::getter,
                          0,
                          pmToValue(ptr),
                          v8::DEFAULT,
                          v8::PropertyAttribute(options));
    }

    template<typename T, typename Obj>
    static void registerROProperty(T (Obj::* pmf) () const,
                                   const char * name,
                                   unsigned options
                                   = v8::DontDelete | v8::ReadOnly)
    {
        tmpl->InstanceTemplate()
            ->SetAccessor(v8::String::NewSymbol(name),
                          PropertyGetter<T, Obj, JSWrapped3>::pmfGetter,
                          0,
                          pmfToValue(pmf),
                          v8::DEFAULT,
                          v8::PropertyAttribute(options));
    }

    template<typename R, typename Obj, typename... Args>
    static void registerAsyncCallback(boost::function<R (Args...)> (Obj::* ptr),
                                      const char * name,
                                      unsigned options = v8::DontDelete)
    {
        typedef boost::function<R (Args...)> T;

        tmpl->InstanceTemplate()
            ->SetAccessor(v8::String::NewSymbol(name),
                          PropertyGetter<T, Obj, JSWrapped3>::getter,
                          AsyncCallbackSetter<T, Obj, JSWrapped3>::setter,
                          pmToValue(ptr),
                          v8::DEFAULT,
                          v8::PropertyAttribute(options));
    }

    template<typename R, typename Obj, typename... Args>
    static void registerMemberFn(R (Obj::* pmf) (Args... args) const,
                                 const char * name)
    {
        // Set it up so that the member function is called when we
        // call the JS version

        v8::Local<v8::Signature> sig = v8::Signature::New(tmpl);
        v8::Local<v8::FunctionTemplate> cb
            =  v8::FunctionTemplate::New
            (MemberFunctionCaller<R, Obj, JSWrapped3, Args...>::call,
             pmfToValue(pmf), sig);
        tmpl->PrototypeTemplate()->Set(v8::String::NewSymbol(name),
                                       cb);
    }

    template<typename R, typename Obj, typename... Args>
    static void registerMemberFn(R (Obj::* pmf) (Args... args),
                                 const char * name)
    {
        // Set it up so that the member function is called when we
        // call the JS version

        v8::Local<v8::Signature> sig = v8::Signature::New(tmpl);
        v8::Local<v8::FunctionTemplate> cb
            =  v8::FunctionTemplate::New
            (MemberFunctionCaller<R, Obj, JSWrapped3, Args...>::call,
             pmfToValue(pmf), sig);
        tmpl->PrototypeTemplate()->Set(v8::String::NewSymbol(name),
                                       cb);
    }

    static v8::Handle<v8::Value>
    extend(const v8::Arguments & args)
    {
        return JSWrappedBase::extendImpl<Wrapper>(args);
    }
};

template<typename Shared, typename Wrapper, typename Base,
         const char * const & ClassNameT, const char * const & ModuleNameT,
         bool defaultWrapper>
const char * const &
JSWrapped3<Shared, Wrapper, Base, ClassNameT, ModuleNameT, defaultWrapper>::
ClassName = ClassNameT;

template<typename Shared, typename Wrapper, typename Base,
         const char * const & ClassNameT, const char * const & ModuleNameT,
         bool defaultWrapper>
const char * const &
JSWrapped3<Shared, Wrapper, Base, ClassNameT, ModuleNameT, defaultWrapper>::
Module = ModuleNameT;

template<typename Shared, typename Wrapper, typename Base,
         const char * const & ClassNameT, const char * const & ModuleNameT,
         bool defaultWrapper>
DoInitialize
JSWrapped3<Shared, Wrapper, Base, ClassNameT, ModuleNameT, defaultWrapper>::
initializer(InitializeFunction, DestructionFunction);

template<typename Shared, typename Wrapper, typename Base,
         const char * const & ClassNameT, const char * const & ModuleNameT,
         bool defaultWrapper>
v8::Persistent<v8::FunctionTemplate>
JSWrapped3<Shared, Wrapper, Base, ClassNameT, ModuleNameT, defaultWrapper>::
tmpl;

} // namespace JS
} // namespace Datacratic
