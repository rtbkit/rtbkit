/* signal.h                                                      -*- C++ -*-
   Jeremy Barnes, 4 November 2010
   Copyright (c) 2010 Datacratic.  All rights reserved.

   A signal framework that allows for signals to be set up.
*/

#pragma once

#include "slot.h"
#include "jml/arch/rtti_utils.h"
#include <boost/function.hpp>
#include <unordered_map>
#include <boost/signals2.hpp>
#include "jml/utils/string_functions.h"

namespace Datacratic {


/*****************************************************************************/
/* SIGNAL INFO                                                               */
/*****************************************************************************/

/** Information about a given signal.  Used to allow discovery and
    exploration of signals.
*/
struct SignalInfo {

    SignalInfo()
        : callbackType(0), objectType(0), inherited(false)
    {
    }

    /** Return a JSON representation of the signal info object. */
    Json::Value toJson() const;

    std::string eventName;  ///< Name of the event
    const std::type_info * callbackType;  ///< Type of the callback function
    const std::type_info * objectType;    ///< Type of the object with callback
    bool inherited;         ///< Is it inherited from our parent?
};


/*****************************************************************************/
/* SIGNAL REGISTRY BASE                                                      */
/*****************************************************************************/

/** Base class that handles the type-independent (non templated) parts of the
    signal registration.
*/

struct SignalRegistryBase {

    /** Construct a new signal registry that doesn't inherit signals from
        anywhere.
    */
    SignalRegistryBase();

    /** Construct a new signal registry that inherits signals from the given
        base class.
    */
    SignalRegistryBase(SignalRegistryBase & inheritFrom);

    ~SignalRegistryBase();

    /** Return the number of signals that are registered for this class. */
    size_t size() const { return signals.size(); }

    /** Set up for the given object to call the given signal on an event
        with the given name.

        Will throw an exception if the given event isn't found or the class
        that we're trying to add it for isn't convertible to the class that
        the event expects.
    */
    template<typename Class>
    SlotDisconnector on(const std::string & name,
                        Class * object,
                        const Slot & slot,
                        int priority = 0) const
    {
        return doOn(name, object, typeid(Class), slot, priority);
    }

    /** List the names of the signals.  Can be used for discovery. */
    std::vector<std::string> names() const;

    /** Get information about the given signal. */
    SignalInfo info(const std::string & name) const;

    /** Inherit the entire set of signals from the other given signal
        registry.
    */
    void inheritSignals(SignalRegistryBase & inheritFrom);

protected:
    /** Callback function to register a callback of a given type with a
        given function.

        The void * is actually the this pointer for the owning class.  The
        passed-in thisType parameter gives the type info node for the
        actual class that this is.  Doing things this way allows us to use
        the internal compiler exception handling machinery to deal with
        different object types at run-time without requiring anything like
        a common base class, etc.

        The return value should provide a boost::function that can be used
        to disconnect the slot.
    */
    typedef SlotDisconnector (* RegisterFn) (void * thisPtr,
                                             const std::type_info & thisType,
                                             const std::string & name,
                                             const Slot & slotToRegister,
                                             int priority,
                                             void * data);
    
    /** Information about a signal.  This is the internal version with
        extra housekeeping information.
    */
    struct Info : public SignalInfo {
        Info()
            : registerFn(0), data(0)
        {
        }

        RegisterFn registerFn;  ///< Function to cause an actual registration
        void * data;            ///< Data to pass to the registration function
    };

    typedef std::unordered_map<std::string, Info> Signals;

    /// The set of signals
    Signals signals;

    /** Do the work to actually add a registration. */
    SlotDisconnector doOn(const std::string & name,
                          void * object,
                          const std::type_info & objectType,
                          const Slot & slot,
                          int priority) const;

    /** Add the given event to the set of known events. */
    void add(const std::string & eventName,
             //const std::string & description, // TODO: would be nice...
             const std::type_info & callbackType,
             const std::type_info & objectType,
             RegisterFn registerFn, void * data = 0);

    /** Subscribe to this to be notified of new events.  By doing that,
        we can subscribe to all of our parents' notifications and thereby
        add them to our list of notifications.
    */
    typedef void (NewEvent) (const SignalRegistryBase &, const Info &);
    
    /** List of events to be notified of. */
    boost::signals2::signal<NewEvent> newEvent;

    /** List of registrations to parents. */
    std::vector<boost::signals2::connection> parentRegistrations;

    /** Inherit this signal from the other handler.  The following are the
        error conditions:
        1.  If the same signal is inherited from more than one place;
        2.  If an inherited signal has a different callback type to the
            locally registered version.
    */
    void addSignal(const Info & signal, bool inherited);
};


/*****************************************************************************/
/* SIGNAL REGISTRY                                                           */
/*****************************************************************************/

/** Type-dependent part of the signal registry. */

template<typename Class>
struct SignalRegistry : public SignalRegistryBase {

    SignalRegistry()
    {
    }

    template<typename Fn,
             SlotDisconnector (Class::* AddFn) (const std::string & name,
                                                const Slot & slot,
                                                int priority,
                                                void * data)>
    static SlotDisconnector doRegister(void * thisPtr,
                                       const std::type_info & thisType,
                                       const std::string & name,
                                       const Slot & slotToRegister,
                                       int priority,
                                       void * data)
    {
        Class * cl = (Class *)ML::is_convertible(thisType, typeid(Class),
                                                 thisPtr);
        if (!cl)
            throw ML::Exception("object of type " + ML::demangle(thisType)
                                + " at "
                                + ML::format("%p", thisPtr)
                                + " is not convertible to "
                                + ML::type_name<Class>()
                                + " trying to register for notification "
                                + name);
        return (cl ->* AddFn) (name, slotToRegister, priority);
    }

    template<typename Fn,
             SlotDisconnector (Class::* AddFn) (const boost::function<Fn> &,
                                                int priority)>
    struct Helper {
        static SlotDisconnector doRegister(void * thisPtr,
                                           const std::type_info & thisType,
                                           const std::string & name,
                                           const Slot & slotToRegister,
                                           int priority,
                                           void * data)
        {
            Class * cl = (Class *)ML::is_convertible(thisType, typeid(Class),
                                                     thisPtr);
            if (!cl)
                throw ML::Exception("object of type " + ML::demangle(thisType)
                                    + " at "
                                    + ML::format("%p", thisPtr)
                                    + " is not convertible to "
                                    + ML::type_name<Class>()
                                    + " trying to register for notification "
                                    + name);
            return (cl ->* AddFn) (slotToRegister.as<Fn>(), priority);
        }
    };

    template<typename Fn,
             SlotDisconnector (Class::* AddFn) (const Slot & slot,
                                                int priority)>
    void add(const std::string & name,
             void (Class::* addFn) (const Slot & slot, int priority))
    {
        SignalRegistryBase::
            add(name, typeid(Fn), typeid(Class), &doRegister<Fn, AddFn>, 0);
    }

    template<typename Fn,
             SlotDisconnector (Class::* AddFn) (const boost::function<Fn> & slot, int priority)>
    void add(const std::string & name)
    {
        RegisterFn fn = &Helper<Fn, AddFn>::doRegister;
        SignalRegistryBase::add(name, typeid(Fn), typeid(Class), fn, 0);
    }
};

struct DoRegisterSignals {
    template<typename Fn>
    DoRegisterSignals(Fn fn)
    {
        fn();
    }
};

} // namespace Datacratic
