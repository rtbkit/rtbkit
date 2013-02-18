/* signal_slot_test_module.cc
   Jeremy Barnes, 17 November 2010
   Copyright (c) 2010 Datacratic.  All rights reserved.

   Module for testing of signals/slots mechanism.
*/


#include <signal.h>
#include "soa/js/js_wrapped.h"
#include "soa/js/js_utils.h"
#include "soa/js/js_registry.h"
#include "v8.h"
#include "jml/compiler/compiler.h"
#include "soa/sigslot/signal.h"
#include "soa/sigslot/slot_impl_js.h"
#include "soa/sigslot/slot_js.h"
#include "jml/arch/rtti_utils.h"
#include "jml/utils/string_functions.h"
#include <cxxabi.h>
#include "soa/js/js_call.h"
#include "soa/jsoncpp/json.h"

using namespace std;
using namespace Datacratic;

struct SignalSlotTest {

    SignalSlotTest()
    {
    }

    virtual ~SignalSlotTest()
    {
    }

    SlotDisconnector
    on(const std::string & event, const Slot & slot)
    {
        return signals().on(event, this, slot);
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
        signals_.add<Event1, &SignalSlotTest::onEvent1>("event1");
        signals_.add<Event2, &SignalSlotTest::onEvent2>("event2");
    }

    virtual std::string objectType() const
    {
        return "SignalSlotTest";
    }

    virtual const SignalRegistryBase & signals() const
    {
        return signals_;
    }

    void doEvent1(const std::string & s)
    {
        event1(s);
    }

    void doEvent2(int i)
    {
        event2(i);
    }

    static SignalRegistry<SignalSlotTest> signals_;
    static DoRegisterSignals reg;
};

RegisterJsOps<SignalSlotTest::Event1> reg1;
RegisterJsOps<SignalSlotTest::Event2> reg2;

SignalRegistry<SignalSlotTest> SignalSlotTest::signals_;
DoRegisterSignals SignalSlotTest::reg(addSignals);

using namespace Datacratic::JS;
using namespace v8;

const char * SignalSlotTestName = "SignalSlotTest";
const char * SignalSlotTestModule = "sig";

//struct SignalSlotTestJS
//    : public JS::JSWrapped3<SignalSlotTest, SignalSlotTestJS, JS::ObjectJS,
//                            SignalSlotTestName, SignalSlotTestModule>;

struct SignalSlotTestJS
    : public JS::JSWrapped2<SignalSlotTest, SignalSlotTestJS,
                            SignalSlotTestName, SignalSlotTestModule> {

    SignalSlotTestJS(const v8::Arguments & args)
    {
        std::shared_ptr<SignalSlotTest> obj
            (new SignalSlotTest());
        
        wrap(args.This(), obj);
    }

    static v8::Persistent<v8::FunctionTemplate>
    Initialize()
    {
        v8::Persistent<FunctionTemplate> t = Register(New, Setup);
        
        // Instance methods
        NODE_SET_PROTOTYPE_METHOD(t, "event1", event1);
        NODE_SET_PROTOTYPE_METHOD(t, "event2", event2);
        NODE_SET_PROTOTYPE_METHOD(t, "checkCast", checkCast);
        NODE_SET_PROTOTYPE_METHOD(t, "on", on);
        NODE_SET_PROTOTYPE_METHOD(t, "signalNames", signalNames);
        NODE_SET_PROTOTYPE_METHOD(t, "signalInfo", signalInfo);

        t->InstanceTemplate()
            ->SetAccessor(String::NewSymbol("lastIndex"), lastIndexGetter);
        
        return t;
    }

    static v8::Handle<v8::Value>
    New(const Arguments & args)
    {
        try {
            new SignalSlotTestJS(args);
            return args.This();
        } HANDLE_JS_EXCEPTIONS;
    }

    static v8::Handle<v8::Value>
    on(const Arguments & args)
    {
        try {
            string event = getArg(args, 0, "event");
            v8::Local<v8::Function> fn = getArg(args, 1, "callback");
            
            Slot s(fn); // = Slot::fromJs(fn, args.This());
            
            SlotDisconnector result = getShared(args)->on(event, s);
            
            return JS::toJS(Slot(result));
        } HANDLE_JS_EXCEPTIONS;
    }

    static v8::Handle<v8::Value>
    signalNames(const Arguments & args)
    {
        try {
            const SignalRegistryBase & signals
                = getShared(args)->signals();
            vector<string> names = signals.names();
            return JS::toJS(names);
        } HANDLE_JS_EXCEPTIONS;
    }
    
    static v8::Handle<v8::Value>
    signalInfo(const Arguments & args)
    {
        try {
            return JS::toJS(getShared(args)
                            ->signals()
                            .info(getArg(args, 0, "signalName"))
                            .toJson());
        } HANDLE_JS_EXCEPTIONS;
    }

    static v8::Handle<v8::Value>
    event1(const Arguments & args)
    {
        try {
            int ntimes = getArg(args, 1, 1, "numTimes");
            for (unsigned i = 0;  i < ntimes;  ++i) {
                getWrapper(args)->lastIndex = i;
                getShared(args)->event1(getArg(args, 0, "intArg"));
            }
            getWrapper(args)->lastIndex = ntimes;
            return NULL_HANDLE;
        } HANDLE_JS_EXCEPTIONS;
    }

    int lastIndex;

    static v8::Handle<v8::Value>
    lastIndexGetter(v8::Local<v8::String> property,
                    const AccessorInfo & info)
    {
        try {
            return v8::Integer::New(getWrapper(info.This())->lastIndex);
        } HANDLE_JS_EXCEPTIONS;
    }
    
    static v8::Handle<v8::Value>
    event2(const Arguments & args)
    {
        try {
            int ntimes = getArg(args, 1, 1, "numTimes");
            for (unsigned i = 0;  i < ntimes;  ++i)
                getShared(args)->event2(getArg(args, 0, "stringArg"));
            return NULL_HANDLE;
        } HANDLE_JS_EXCEPTIONS;
    }

    static v8::Handle<v8::Value>
    checkCast(const Arguments & args)
    {
        try {
            SignalSlotTest * obj = getShared(args);
            const void * converted1
                = ML::is_convertible(typeid(SignalSlotTest),
                                     typeid(SignalSlotTest),
                                     obj);
            
            SignalSlotTest * converted2
                = dynamic_cast<SignalSlotTest *>(obj);

            v8::Local<v8::Array> result(v8::Array::New(2));
            result->Set(v8::Uint32::New(0),
                        v8::String::New(ML::format("%p", converted1).c_str()));
            result->Set(v8::Uint32::New(1),
                        v8::String::New(ML::format("%p", converted2).c_str()));
            
            return result;
        } HANDLE_JS_EXCEPTIONS;
    }

};

typedef void (DoSomething) ();
RegisterJsOps<DoSomething> reg_void;

extern "C" void
init(Handle<v8::Object> target)
{
    registry.init(target, SignalSlotTestModule);
}
