/* slot.cc
   Jeremy Barnes, 16 November 2010
   Copyright (c) 2010 Datacratic.  All rights reserved.
   
   Code to deal with slots.
*/

#include "slot.h"
#if NODEJS_ENABLED
#include "v8.h"
#include "soa/js/js_utils.h"
#include "soa/js/js_call.h"
#endif // NODEJS_ENABLED
#include <boost/signals2.hpp>
#include <boost/bind.hpp>
#include "jml/arch/format.h"


using namespace std;
using namespace ML;


namespace Datacratic {

#if NODEJS_ENABLED
bool inJsContext()
{
    return true;
    return v8::Locker::IsLocked();
}

void enterJs(void * & locker)
{
    locker = 0;
    if (v8::Locker::IsLocked()) return;
    v8::Locker::StartPreemption(1);  // every one ms
    locker = new v8::Locker();
}

void exitJs(void * & locker_)
{
    if (!locker_) return;
    v8::Locker * locker = (v8::Locker *)locker_;
    delete locker;
    locker_ = 0;
}
#endif // NODEJS_ENABLED


/*****************************************************************************/
/* SLOT DISCONNECTOR                                                         */
/*****************************************************************************/

SlotDisconnector::
SlotDisconnector(const boost::signals2::connection & connection)
    : boost::function<void (void)>
      (boost::bind(&boost::signals2::connection::disconnect,
                   connection))
{
}


/*****************************************************************************/
/* SLOT                                                                      */
/*****************************************************************************/

#if NODEJS_ENABLED
Slot::
Slot(const Slot & other)
    : fn(other.fn), ops(other.ops), jsops(other.jsops), fntype(other.fntype)
{
    switch (fntype) {
    case EMPTY: break;
    case BOOST:
        if (ops && fn)
            fn = (boost::function_base *)(ops(3, fn));
        break;
    case JS:
        jsfn = new v8::Persistent<v8::Function>
            (v8::Persistent<v8::Function>::New(*jsfn));
        break;
    default:
        throw Exception("wrong fntype");
    }
}
#else // NODEJS_ENABLED
Slot::
Slot(const Slot & other)
    : fn(other.fn), ops(other.ops), fntype(other.fntype)
{
    switch (fntype) {
    case EMPTY: break;
    case BOOST:
        if (ops && fn)
            fn = (boost::function_base *)(ops(3, fn));
        break;
    default:
        throw Exception("wrong fntype");
    }
}
#endif // NODEJS_ENABLED

#if NODEJS_ENABLED
Slot::
Slot(Slot && other)
    : fn(other.fn), ops(other.ops), jsops(other.jsops), fntype(other.fntype)
{
    other.fntype = EMPTY;
}
#else // NODEJS_ENABLED
Slot::
Slot(Slot && other)
    : fn(other.fn), ops(other.ops), fntype(other.fntype)
{
    other.fntype = EMPTY;
}
#endif // NODEJS_ENABLED

#if NODEJS_ENABLED
Slot::
Slot(const v8::Handle<v8::Function> & fn)
    : jsfn(new v8::Persistent<v8::Function>
           (v8::Persistent<v8::Function>::New(fn))),
      fntype(JS)
{
    if (fn.IsEmpty() || jsfn->IsEmpty())
        throw Exception("invalid function");
}

Slot::
Slot(const v8::Handle<v8::Value> & fn)
    : jsfn(new v8::Persistent<v8::Function>
           (v8::Persistent<v8::Function>::New
            (v8::Handle<v8::Function>
             (v8::Function::Cast(*fn))))),
      fntype(JS)
{
    if (fn.IsEmpty())
        throw Exception("invalid function");
    if (jsfn->IsEmpty())
        throw Exception("Non-function JS value " + JS::cstr(fn) + " passed to slot");
}
#endif // NODEJS_ENABLED

Slot::
~Slot()
{
    free();
}
     
void
Slot::
free()
{
    switch (fntype) {
    case EMPTY: break;
    case BOOST:
        if (ops) ops(1, fn);
        break;
#if NODEJS_ENABLED
    case JS:
        if (jsfn) {
            jsfn->Dispose();
            jsfn->Clear();
            delete jsfn;
        }
        break;
#endif // NODEJS_ENABLED
    default:
        throw Exception("Slot::free(): wrong type");
    }

    fntype = EMPTY;
}

Slot &
Slot::
operator = (const Slot & other)
{
    Slot new_me(other);
    swap(new_me);
    return *this;
}

Slot &
Slot::
operator = (Slot && other)
{
    Slot new_me(other);
    swap(other);
    return *this;
}

void
Slot::
swap(Slot & other)
{
    std::swap(fntype, other.fntype);
    std::swap(fn, other.fn);
    std::swap(ops, other.ops);
#if NODEJS_ENABLED
    std::swap(jsops, other.jsops);
#endif // NODEJS_ENABLED
}

std::string
Slot::
print() const
{
    switch (fntype) {
    case EMPTY:
        return "(empty)";
    case BOOST:
        return "(c++) " + ML::demangle(type()) + "  as "
            + ML::demangle(fn->target_type());
#if NODEJS_ENABLED
    case JS:
        return "(js) " + JS::cstr(*jsfn);
#endif // NODEJS_ENABLED
    default:
        return "(invalid type) " + ML::format("%d", fntype);
    }
}
        
#if NODEJS_ENABLED
v8::Handle<v8::Value>
Slot::
call(const v8::Arguments & args) const
{
    switch (fntype) {
    case EMPTY:
        throw Exception("cannot call an empty function");
    case BOOST: {
        if (!jsops)
            throw ML::Exception("no javascript translator");
        JS::JSCallsBoost op = (JS::JSCallsBoost)jsops;

        v8::HandleScope scope;

        v8::Handle<v8::Value> result;
        op(0, *fn, args, result);
        return scope.Close(result);
    }
    case JS: {
        // Forward directly to JS
        v8::HandleScope scope;
        vector<v8::Local<v8::Value> > vals(args.Length());
        for (unsigned i = 0;  i < vals.size();  ++i)
            vals[i] = args[i];
        return scope.Close((*jsfn)->Call(args.This(), args.Length(),
                                         &vals[0]));
    }
    default:
        throw Exception("invalid fn type");
    }
}

v8::Handle<v8::Value>
Slot::
call(const v8::Handle<v8::Object> & This,
     int argc, v8::Handle<v8::Value> argv[]) const
{
    switch (fntype) {
    case EMPTY:
        throw Exception("cannot call an empty function");
    case BOOST: {
        if (!jsops)
            throw ML::Exception("no javascript translator");
        JS::JSCallsBoost op = (JS::JSCallsBoost)jsops;

        v8::HandleScope scope;

        v8::Handle<v8::Value> result;
        op(0, *fn, JS::JSArgs(This, argc, argv), result);

        return scope.Close(result);
    }
    case JS:
        // Forward directly to JS
        return (*jsfn)->Call(This, argc, argv);
    default:
        throw Exception("invalid fn type");
    }
}
#endif // NODEJS_ENABLED

const std::type_info &
Slot::
type() const
{
    switch (fntype) {
    case EMPTY:
        return typeid(void);
    case BOOST: {
        std::type_info * p = 0;
        ops(0, &p);
        return *p;
    }
#if NODEJS_ENABLED
    case JS:
        return typeid(v8::Function);
#endif // NODEJS_ENABLED
    default:
        throw Exception("Slot::type(): invalid type");
    }
}

#if NODEJS_ENABLED
namespace JS {

Slot from_js(const JSValue & val, Slot *)
{
    return Slot(from_js(val, (v8::Handle<v8::Function> *)0));
}

} // namespace JS
#endif // NODEJS_ENABLED

} // namespace Datacratic
