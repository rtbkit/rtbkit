/* signal_test.cc
   Jeremy Barnes, 16 November 2010
   Copyright (c) 2010 Datacratic.  All rights reserved.

   Test of the signal functionality.
*/


#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include <iostream>
#include "jml/arch/exception_handler.h"
#include "jml/utils/vector_utils.h"
#include "jml/utils/string_functions.h"

#include "soa/sigslot/signal.h"
#include "soa/sigslot/slot_impl_js.h"

#include "soa/js/js_call.h"

using namespace std;
using namespace ML;
using namespace Datacratic;


struct ThingWithSignals {

    virtual ~ThingWithSignals()
    {
    }

    typedef void (Event1) (const std::string &);

    SlotDisconnector onEvent1(const boost::function<Event1> & fn,
                              int priority = 0)
    {
        return event1.connect(priority, fn);
    }

    boost::signals2::signal<Event1> event1;

    typedef void (Event2) (int i);

    virtual SlotDisconnector onEvent2(const boost::function<Event2> & fn,
                                      int priority = 0)
    {
        return event2.connect(priority, fn);
    }
    
    boost::signals2::signal<Event2> event2;

    static void addSignals()
    {
        signals.add<Event1, &ThingWithSignals::onEvent1>("event1");
        signals.add<Event2, &ThingWithSignals::onEvent2>("event2");
    }

    void on(const std::string & event, const Slot & slot, int priority = 0)
    {
        signals.on(event, this, slot, priority);
    }

    template<typename Fn, typename Fn2>
    void on(const std::string & event, const boost::function<Fn2> & fn,
            int priority = 0)
    {
        Slot n = Slot::fromF<Fn>(fn);
        on(event, n, priority);
    }

    template<typename Fn, typename Fn2>
    void on(const std::string & event, Fn2 fn, int priority = 0)
    {
        Slot n = Slot::fromF<Fn>(fn);
        on(event, n, priority);
    }

    static SignalRegistry<ThingWithSignals> signals;
    static DoRegisterSignals reg;
};

RegisterJsOps<ThingWithSignals::Event1> reg1;
RegisterJsOps<ThingWithSignals::Event2> reg2;

SignalRegistry<ThingWithSignals> ThingWithSignals::signals;
DoRegisterSignals ThingWithSignals::reg(addSignals);


BOOST_AUTO_TEST_CASE( test_signal_object )
{

    using namespace v8;

    ThingWithSignals thing;

    BOOST_CHECK_EQUAL(thing.signals.size(), 2);

    vector<string> strs;

    auto fn = [&] (const std::string & str) { strs.push_back(str); }; 
    boost::function<void (const std::string &)> fn_ = fn;
    
    thing.on<void (const std::string &)>("event1", fn);
    thing.on<void (const std::string &)>("event1", fn_);
    
    thing.event1("hello");

    BOOST_CHECK_EQUAL(strs, vector<string>({"hello", "hello"}));

    int total = 0;

    auto fn2 = [&] (int i) { total += i; };

    thing.on<void (int i)>("event2", fn2);

    thing.event2(6);

    BOOST_CHECK_EQUAL(total, 6);
    
    {
        // Check that adding the wrong callback type throws
        JML_TRACE_EXCEPTIONS(false);
        BOOST_CHECK_THROW(thing.on<void (int i)>("event1", fn2), ML::Exception);
    }
}

struct DerivedThingWithSignals : public ThingWithSignals {

    virtual ~DerivedThingWithSignals()
    {
    }

    typedef void (Event3) (int i, const std::string & s);

    boost::signals2::signal<Event3> event3;

    SlotDisconnector onEvent3(const boost::function<Event3> & fn,
                              int priority = 0)
    {
        return event3.connect(priority, fn);
    }

    static void addSignals()
    {
        signals.inheritSignals(ThingWithSignals::signals);
        signals.add<Event3, &DerivedThingWithSignals::onEvent3>("event3");
    }

    void on(const std::string & event, const Slot & slot)
    {
        signals.on(event, this, slot);
    }

    template<typename Fn, typename Fn2>
    void on(const std::string & event, const boost::function<Fn2> & fn)
    {
        Slot n = Slot::fromF<Fn>(fn);
        on(event, n);
    }

    template<typename Fn, typename Fn2>
    void on(const std::string & event, Fn2 fn)
    {
        Slot n = Slot::fromF<Fn>(fn);
        on(event, n);
    }

    static SignalRegistry<DerivedThingWithSignals> signals;
    static DoRegisterSignals reg;
};

SignalRegistry<DerivedThingWithSignals> DerivedThingWithSignals::signals;
DoRegisterSignals DerivedThingWithSignals::reg(addSignals);

RegisterJsOps<DerivedThingWithSignals::Event3> reg3;

BOOST_AUTO_TEST_CASE( test_signal_derived_object )
{

    using namespace v8;

    DerivedThingWithSignals thing;

    BOOST_CHECK_EQUAL(thing.signals.size(), 3);

    vector<string> strs;

    auto fn = [&] (const std::string & str) { strs.push_back(str); }; 
    boost::function<void (const std::string &)> fn_ = fn;
    
    thing.on<void (const std::string &)>("event1", fn);
    thing.on<void (const std::string &)>("event1", fn_);
    
    thing.event1("hello");

    BOOST_CHECK_EQUAL(strs, vector<string>({"hello", "hello"}));

    int total = 0;

    auto fn2 = [&] (int i) { total += i; };

    thing.on<void (int i)>("event2", fn2);

    thing.event2(6);

    BOOST_CHECK_EQUAL(total, 6);
    
    {
        // Check that adding the wrong callback type throws
        JML_TRACE_EXCEPTIONS(false);
        BOOST_CHECK_THROW(thing.on<void (int i)>("event1", fn2), ML::Exception);
    }

    string str;

    auto fn3 = [&] (int i, string s)
        { str += ML::format("%s%i", s.c_str(), i); };

    thing.on<void (int, const std::string &)>("event3", fn3);

    thing.event3(4, "hello");

    BOOST_CHECK_EQUAL(str, "hello4");
}

