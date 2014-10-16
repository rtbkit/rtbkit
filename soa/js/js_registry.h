/* js_registry.h                                                   -*- C++ -*-
   Jeremy Barnes, 27 July 2010
   Copyright (c) 2010 Datacratic.  All rights reserved.

   Registry for JS classes.
*/

#pragma once

#include <map>
#include <vector>
#include <v8/v8.h>
#include <boost/function.hpp>
#include "jml/utils/exc_assert.h"
#include "jml/arch/demangle.h"
#include "jml/arch/rtti_utils.h"
#include "jml/arch/backtrace.h"
#include "js_value.h"
#include <boost/shared_ptr.hpp>
#include <iostream>


namespace Datacratic {
namespace JS {


typedef boost::function<void (const v8::Persistent<v8::FunctionTemplate> &,
                              const std::string & name,
                              const v8::Handle<v8::Object> &)> SetupFunction;
typedef boost::function<void ()> InitFunction;


enum OnConstructorError {
    THROW_ON_ERROR,
    RETURN_NULL_ON_ERROR
};

const std::type_info & getWrapperTypeInfo(v8::Handle<v8::Value> obj);


/*****************************************************************************/
/* REGISTRY                                                                  */
/*****************************************************************************/

struct Registry {

    Registry();

    ~Registry();

    const v8::Persistent<v8::FunctionTemplate> &
    operator [] (const std::string & name) const;

    /** Initialize the given target by instantiating all of the objects that
        are specified in the registry.
    */
    void init(v8::Handle<v8::Object> target, const std::string & module);


    /** Bring in an object from another registry without actually causing it
        to be instantiated.
    */
    void import(const Registry & other, const std::string & name);

    bool empty() const { return templates.empty(); }

    v8::Local<v8::Function>
    getConstructor(const std::string & cls) const;

    v8::Local<v8::Object>
    constructInstance(const std::string & cls,
                      OnConstructorError e = THROW_ON_ERROR) const;

    v8::Local<v8::Object>
    constructInstance(const std::string & cls,
                      const v8::Arguments & args,
                      OnConstructorError e = THROW_ON_ERROR) const;

    v8::Local<v8::Object>
    constructInstance(const std::string & cls,
                      int argc, v8::Handle<v8::Value> * argv,
                      OnConstructorError e = THROW_ON_ERROR) const;


    void introduce(const std::string & name,
                   const std::string & module,
                   const InitFunction init,
                   const std::string & base = "");

    void get_to_know(const std::string & name,
                     const std::string & module,
                     const v8::Persistent<v8::FunctionTemplate> & tmpl,
                     SetupFunction setup);

    /** Type for a factory function that constructs an instance from a) a
        pointer to a smart pointer to a base class, and b) an adjusted
        pointer to the derived object.
    */
    typedef v8::Local<v8::Object> (* ConstructWrapperFunction) (void *, const void *);

    typedef void (* UnwrapFunction) (const v8::Handle<v8::Value> &, void *,
                                     const std::type_info &);

    /** Indicate that the given wrapper class wraps the given object. */
    template<typename Shared, typename Wrapper>
    void isWrapper(ConstructWrapperFunction construct,
                   UnwrapFunction unwrap)
    {
        const std::type_info * ti = &typeid(Shared);

        //using namespace std;
        //cerr << ML::type_name<Wrapper>() << " is wrapper for "
        //     << ML::type_name<Shared>() << endl;

        if (type_entries.count(ti))
            throw ML::Exception("double wrap of type "
                                + ML::type_name<Shared>()
                                + ": " + type_entries[ti].wrapper
                                + " and " + ML::type_name<Wrapper>());

        type_entries[ti].construct = construct;
        type_entries[ti].wrapper = ML::type_name<Wrapper>();

        ti = &typeid(Wrapper);
        if (type_entries.count(ti))
            throw ML::Exception("double wrap of type "
                                + ML::type_name<Wrapper>()
                                + ": " + type_entries[ti].wrapper
                                + " and " + ML::type_name<Wrapper>());
        
        type_entries[ti].unwrap = unwrap;
    }

    /** Notify that the given base is the base class for the derived type.
        This is used to determine which objects might wrap a given wrapper
        class.
    */
    template<typename Base, typename Derived>
    void isBase()
    {
#if 0
        using namespace std;
        cerr << ML::type_name<Derived>()
             << " derives from " << ML::type_name<Base>()
             << endl;
#endif
    }

    /** Return the maximally specific Javascript wrapper object for the
        given base class.

        For example, in the following class hierarchy:

        Base <- Derived1 <- Derived2
             <- Derived1A
        wrapped with the following JS class hierarchy:

        BaseJS <- Derived1JS <- Derived2JS
               <- Derived1AJS

        then calling getWrapper with a shared pointer to Base would return
        the Derived1JS if the object was Derived1, Derived2JS if the object
        was Derived2, etc.  If the object was another class derived from
        Base (that wasn't wrapped), then the BaseJS wrapper object would
        be returned.

        This works by making wrapper objects register themselves using the
        isWrapper() call, and recording the type information necessary
        to do so.

        A null pointer or a pointer to a class that wasn't actually
        convertible will throw an exception.
    */
    template<typename Base>
    v8::Local<v8::Object>
    getWrapper(const std::shared_ptr<Base> & object) const
    {
        if (!object) {
            //ML::backtrace();
            throw ML::Exception("getWrapper(): no object");
        }

        const std::type_info * ti = &typeid(*object);

        // Is there a direct wrapper function?
        auto it = type_entries.find(ti);
        if (it != type_entries.end() && ti != &typeid(Base)) {
            const TypeEntry & entry = it->second;

            // Offset the pointer to do the equivalent of a dynamic_cast
            const void * ptr = ML::is_convertible(*object, *ti);

            // Create a stack-based shared ptr to be on the stack
            std::shared_ptr<Base> sp = object;

            // Construct the instance
            return entry.construct(&sp, ptr);
        }    

        // No direct wrapper.  Look for the most specific type under Base
        // that matches.

        // ...
        // TODO: do

        // No more specific wrapper.  We use the base handler
        ti = &typeid(Base);
        it = type_entries.find(ti);

        if (it != type_entries.end()) {
            const TypeEntry & entry = it->second;

            // No dynamic cast because the pointer is already there
            void * ptr = (void *)(object.get());

            // Create a stack-based shared ptr to be on the stack
            std::shared_ptr<Base> sp = object;

            // Construct the instance
            return entry.construct(&sp, ptr);
        }

        throw ML::Exception("don't know how to wrap a "
                            + ML::type_name(*object));
    }

#if 1
    /** Convert a wrapped object into its underlying value.  Uses the type
        info system to figure out where it comes from.
    */
    template<typename Object>
    std::shared_ptr<Object>
    getObject(v8::Local<v8::Value> wrapped) const
    {
        const std::type_info & wrapperType
            = getWrapperTypeInfo(wrapped);

        const std::type_info & objectType = typeid(Object);

        auto it = type_entries.find(&wrapperType);
        if (it == type_entries.end())
            throw ML::Exception("Can't convert wrapper of type "
                                + ML::type_name(wrapperType)
                                + " to object of type "
                                + ML::type_name(objectType));

        // Look up how to get an object out of it
        std::shared_ptr<Object> result;
        it->second.unwrap(wrapped, &result, objectType);

        return result;
    }

    template<typename Object>
    std::shared_ptr<Object>
    getObject(v8::Handle<v8::Value> wrapped) const
    {
        const std::type_info & wrapperType
            = getWrapperTypeInfo(wrapped);

        const std::type_info & objectType = typeid(Object);

        auto it = type_entries.find(&wrapperType);
        if (it == type_entries.end())
            throw ML::Exception("Can't convert wrapper of type "
                                + ML::type_name(wrapperType)
                                + " to object of type "
                                + ML::type_name(objectType));

        // Look up how to get an object out of it
        std::shared_ptr<Object> result;
        it->second.unwrap(wrapped, &result, objectType);

        return result;
    }
#endif

private:
    /** Entry on how to construct a wrapper for a given type. */
    struct TypeEntry {
        TypeEntry()
            : construct(0), unwrap(0)
        {
        }

        ConstructWrapperFunction construct;
        UnwrapFunction unwrap;
        std::string wrapper;
    };
    
    typedef std::map<const std::type_info *, TypeEntry> TypeEntries;
    TypeEntries type_entries;

    struct Entry {
        Entry()
            : setup(0), imported(false)
        {
        }

        std::string name;
        std::string base;
        std::string module;
        InitFunction init;
        v8::Persistent<v8::FunctionTemplate> tmpl;
        SetupFunction setup;
        bool imported;
    };

    int num_uninitialized;

    std::map<std::string, Entry> templates;

    void initialize();
    void do_initialize(const std::string & name);
};



template<typename Derived, typename DerivedWrapper, typename Base>
v8::Handle<v8::Object>
doDerived(Registry & registry,
          const std::shared_ptr<Base> & handler,
          const char * name)
{
    v8::Handle<v8::Object> result;

    std::shared_ptr<Derived> cast
        = std::dynamic_pointer_cast<Derived>(handler);

    if (cast) {
        result = registry.constructInstance(name);
        ExcAssert(!result.IsEmpty());
        DerivedWrapper::setShared(result, cast);
    }
    
    return result;
}

extern Registry registry;

} // namespace JS
} // namespace Datacratic
