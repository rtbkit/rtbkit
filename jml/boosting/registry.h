/* registry.h                                                      -*- C++ -*-
   Jeremy Barnes, 20 June 2003
   Copyright (c) 2003 Jeremy Barnes.  All rights reserved.
   $Source$

   Contains the functions to register polymorphic heirarchies of objects
   that can all be serialized and reconstituted in one big happy family.
*/

#ifndef __boosting__registry_h__
#define __boosting__registry_h__


#include "jml/db/persistent.h"
#include "jml/db/nested_archive.h"
#include "jml/utils/hash_map.h"
#include "jml/arch/demangle.h"
#include <iostream>


#define NO_SMART_PTRS 0
#undef VERSION

namespace ML {

/* Factory class.  Note that it is expected that this will be specialised
   for the particular type of base, to hold the arguments that it is expected
   to have.

   The default one attempts to be generic in how it operates, by allowing
   constructors up to 3 arguments.
*/

template<class Base>
class Factory_Base {
public:
    virtual ~Factory_Base() {}
    virtual std::shared_ptr<Base> create() const = 0;
    virtual std::shared_ptr<Base>
    reconstitute(DB::Store_Reader & store) const = 0;
};


/* The default implementation of a factory.  Default constructs the derived
   class.  Needs to be overridden for ones that take arguments. */
template<class Base, class Derived>
class Object_Factory : public Factory_Base<Base> {
public:
    virtual ~Object_Factory() {}
    virtual std::shared_ptr<Base> create() const
    {
        return std::shared_ptr<Base>(new Derived());
    }
    virtual std::shared_ptr<Base>
    reconstitute(DB::Store_Reader & store) const
    {
        std::shared_ptr<Derived> result(new Derived());
        result->reconstitute(store);
        return result;
    }
};

template<class Base>
class Registry {
public:
#if !NO_SMART_PTRS
    typedef std::hash_map<std::string,
                          std::shared_ptr<Factory_Base<Base> > >
        entries_type;
#else
    typedef std::hash_map<std::string, Factory_Base<Base> *>
        entries_type;
#endif
    entries_type entries;

#if NO_SMART_PTRS
    ~Registry()
    {
        for (typename entries_type::const_iterator it = entries.begin();
             it != entries.end();  ++it) delete(it->second);
    }
#endif

    static Registry & singleton()
    {
        if (registry == 0) {
            registry = new Registry();
            //cerr << "registering registry for "
            //     << demangle(typeid(Base).name()) << endl;
        }
        return *registry;
    }

    static Registry *registry;

    static const unsigned VERSION = 0;

    void serialize(DB::Store_Writer & store, const Base * obj)
    {
        std::string classid = obj->class_id();
        if (entries.count(classid) == 0)
            throw Exception(demangle(typeid(Base).name())
                            + ": Attempt to serialize unregistered class "
                            + classid);

        DB::Nested_Writer writer;
        obj->serialize(writer);

        store << classid << DB::compact_size_t(VERSION);

        /* Write out the contents. */
        store << writer;
    }

    static std::string entry_list()
    {
        std::string result;
        const Registry & registry = singleton();
        for (typename entries_type::const_iterator it
                 = registry.entries.begin();
             it != registry.entries.end();  ++it) {
            if (result != "") result += ", ";
            result += it->first;
        }
        return result;
    }

    void dump_entries() const
    {
        std::cerr << "Known entries in registry: " << entry_list() << std::endl;
    }

    std::shared_ptr<Base> reconstitute(DB::Store_Reader & store) const
    {
        std::string key;
        DB::compact_size_t version;
        DB::compact_size_t length;
        store >> key >> version >> length;
        
        /* Make sure we know about the object. */
        if (entries.count(key) == 0) {
            dump_entries();
            store.skip(length);
            throw Exception("object with key " + key + " not recognised by "
                            + demangle(typeid(Base).name()));
        }

        /* Make sure we know about the version of information we're writing. */
        if (version > VERSION) {
            store.skip(length);
            throw Exception("version too high");
        }

        return entries.find(key)->second->reconstitute(store);
    }

    /* Version that takes one argument. */
    template<class A1>
    std::shared_ptr<Base>
    reconstitute(DB::Store_Reader & store, A1 a1) const
    {
        //using namespace std;
        //cerr << __PRETTY_FUNCTION__ << endl;
        std::string key;
        DB::compact_size_t version;
        DB::compact_size_t length;
        store >> key >> version >> length;
        
        //cerr << "key = " << key << endl;

        /* Make sure we know about the object. */
        if (entries.count(key) == 0) {
            dump_entries();
            store.skip(length);
            throw Exception("object with key " + key + " not recognised by "
                            + demangle(typeid(Base).name()));
        }

        /* Make sure we know about the version of information we're writing. */
        if (version > 0) {
            store.skip(length);
            throw Exception("version too high");
        }

        std::shared_ptr<Base> result
            = entries.find(key)->second->reconstitute(a1, store);
        return result;
    }

    std::shared_ptr<Base> create(const std::string & key) const
    {
        /* Make sure we know about the object. */
        if (entries.count(key) == 0) {
            dump_entries();
            throw Exception("object with key " + key + " not recognised by "
                            + demangle(typeid(Base).name()));
        }

        return entries.find(key)->second->create();
    }

    bool known_type(const std::string & key) const
    {
        return entries.count(key);
    }
};

template<class Base> Registry<Base> * Registry<Base>::registry;


/** A convenience initialisation class, that allows simple factories to be
    registered quickly and conveniently. */
template<class Base, class Derived,
         class Factory_ = Object_Factory<Base, Derived> >
class Register_Factory {
public:
    Register_Factory(Factory_ * factory, const std::string & classid)
    {
        Registry<Base>::singleton().entries[classid]
#if !NO_SMART_PTRS            
            = std::shared_ptr<Factory_>(factory);
#else
            = factory;
#endif        
    }

    Register_Factory(const std::string & classid)
    {
        //cerr << "registering factory of type "
        //     << demangle(typeid(Factory_).name()) << " for classid "
        //     << classid << endl;
        Registry<Base>::singleton().entries[classid]
#if !NO_SMART_PTRS
        = std::shared_ptr<Factory_>(new Factory_());
#else
         = new Factory_();
#endif
    }
};


} // namespace ML



#endif /* __boosting__registry_h__ */
