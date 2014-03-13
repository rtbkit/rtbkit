/* signal.cc
   Jeremy Barnes, 15 November 2010
   Copyright (c) 2010 Datacratic.  All rights reserved.
   
   Code to deal with notifiers.
*/

#include "signal.h"
#include "jml/utils/pair_utils.h"
#include "v8.h"
#include "soa/js/js_utils.h"
#include "soa/jsoncpp/json.h"

using namespace std;
using namespace ML;


namespace Datacratic {

/*****************************************************************************/
/* SIGNAL INFO                                                               */
/*****************************************************************************/

Json::Value
SignalInfo::
toJson() const
{
    Json::Value result;
    result["name"] = eventName;
    result["callbackType"] = ML::demangle(*callbackType);
    result["objectType"]   = ML::demangle(*objectType);
    result["inherited"]    = inherited;
    return result;
}


/*****************************************************************************/
/* SIGNAL REGISTRY BASE                                                      */
/*****************************************************************************/

SignalRegistryBase::
SignalRegistryBase()
{
}

SignalRegistryBase::
SignalRegistryBase(SignalRegistryBase & inheritFrom)
{
    inheritSignals(inheritFrom);
}

SignalRegistryBase::
~SignalRegistryBase()
{
    std::for_each(parentRegistrations.begin(), parentRegistrations.end(),
                  [&] (boost::signals2::connection & conn)
                  { conn.disconnect(); });
}

std::vector<std::string>
SignalRegistryBase::
names() const
{
    return vector<string>(ML::first_extractor(signals.begin()),
                          ML::first_extractor(signals.end()));
}

SignalInfo
SignalRegistryBase::
info(const std::string & name) const
{
    auto it = signals.find(name);
    if (it == signals.end())
        throw Exception("no signal by name of " + name);
    return it->second;
}

SlotDisconnector
SignalRegistryBase::
doOn(const std::string & name,
     void * object,
     const std::type_info & objectType,
     const Slot & slot,
     int priority) const
{
    auto it = signals.find(name);
    if (it == signals.end())
        throw ML::Exception("no signal " + name);
    return it->second.registerFn(object, objectType, name, slot, priority, 0);
}

void
SignalRegistryBase::
inheritSignals(SignalRegistryBase & inheritFrom)
{
    for (auto it = inheritFrom.signals.begin(),
             end = inheritFrom.signals.end();
         it != end;  ++it)
        addSignal(it->second, true /* inherited */);
    
    parentRegistrations.push_back
        (inheritFrom.newEvent.connect
         (boost::bind(&SignalRegistryBase::addSignal, this, _2, true)));
}

void
SignalRegistryBase::
add(const std::string & eventName,
    //const std::string & description, // TODO: would be nice...
    const std::type_info & callbackType,
    const std::type_info & objectType,
    RegisterFn registerFn, void * data)
{
    Info info;
    info.eventName = eventName;
    info.callbackType = &callbackType;
    info.objectType = &objectType;
    info.registerFn = registerFn;
    info.data = data;
    info.inherited = false;
    
    addSignal(info, false /* inherited */);
}

void
SignalRegistryBase::
addSignal(const Info & signal, bool inherited)
{
    std::string name = signal.eventName;
    bool exists = signals.count(name);
    Info & info = signals[name];
    
    if (exists) {
        if (info.inherited && inherited)
            throw ML::Exception("signal " + signal.eventName
                                + " was inherited twice");

        if (!info.inherited && !inherited)
            throw ML::Exception("signal " + signal.eventName
                                + " was added twice");
        
        if (info.callbackType != signal.callbackType)
            throw ML::Exception("signal " + signal.eventName
                                + " had inherited callback type "
                                + ML::demangle(*signal.callbackType)
                                + " but local callback type "
                                + ML::demangle(*info.callbackType));
        
        // The inherited version shouldn't override our local version
        if (!info.inherited && inherited)
            return;  
    }
    
    info = signal;
    info.inherited = inherited;
    
    // Notify anything that needs to know that we have a new event
    newEvent(*this, info);
}

} // namespace Datacratic
