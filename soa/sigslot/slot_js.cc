/* slot_js.cc
   Jeremy Barnes, 16 November 2010
   Copyright (c) 2010 Datacratic.  All rights reserved.

   JS interface to slots.
*/

#include "slot_js.h"
#include "soa/sigslot/slot.h"

#include "node.h"

#include "soa/js/js_value.h"
#include "soa/js/js_utils.h"
#include "soa/js/js_wrapped.h"

#include "jml/utils/smart_ptr_utils.h"


using namespace std;
using namespace ML;


namespace Datacratic {
namespace JS {


/*****************************************************************************/
/* SLOTJS                                                                    */
/*****************************************************************************/

const char * SlotName = "Slot";

struct SlotJS : public JSWrapped2<Slot, SlotJS, SlotName, sigslotModule> {

    SlotJS(v8::Handle<v8::Object> This,
           const std::shared_ptr<Slot> & slot
               = std::shared_ptr<Slot>())
    {
        wrap(This, slot);
    }

    SlotJS(v8::Handle<v8::Object> This,
           const Slot & slot)
    {
        wrap(This, make_std_sp(new Slot(slot)));
    }

    static v8::Handle<v8::Value>
    New(const v8::Arguments & args)
    {
        try {
            new SlotJS(args.This());
            return args.This();
        } HANDLE_JS_EXCEPTIONS;
    }

    static v8::Handle<v8::Value>
    call(const v8::Arguments & args)
    {
        try {
            return getShared(args)->call(args);
        } HANDLE_JS_EXCEPTIONS;
    }

    static void Initialize()
    {
        using namespace v8;
        v8::Persistent<v8::FunctionTemplate> t = Register(New);
        NODE_SET_PROTOTYPE_METHOD(t, "toString", toString);
        NODE_SET_PROTOTYPE_METHOD(t, "inspect", toString);
        NODE_SET_PROTOTYPE_METHOD(t, "call", call);
        t->InstanceTemplate()->SetCallAsFunctionHandler(call);
    }

    static v8::Handle<v8::Value>
    toString(const v8::Arguments & args)
    {
        try {
            return JS::toJS("[Slot " + getShared(args)->print() + "]");
        } HANDLE_JS_EXCEPTIONS;
    }
};


Slot * from_js(const JSValue & val, Slot **)
{
    return SlotJS::fromJS(val).get();
}

std::shared_ptr<Slot>
from_js(const JSValue & val, std::shared_ptr<Slot>*)
{
    return SlotJS::fromJS(val);
}

void to_js(JS::JSValue & value, const std::shared_ptr<Slot> & slot)
{
    if (slot->isEmpty())
        value = v8::Null();
    else value = SlotJS::toJS(slot);
}

void to_js(JS::JSValue & value, const Slot & slot)
{
    if (slot.isEmpty())
        value = v8::Null();
    else to_js(value, make_std_sp(new Slot(slot)));
}

const char * const sigslotModule = "sigslot";

} // namespace JS

v8::Local<v8::Function>
Slot::
toJsFunction() const
{
    v8::HandleScope scope;

    std::shared_ptr<Slot> s(new Slot(*this));
    v8::Local<v8::Value> res = JS::SlotJS::toJS(s);

    v8::Local<v8::Function> cast = v8::Function::Cast(*res);

    if (cast.IsEmpty() || cast->IsUndefined() || cast->IsNull())
        throw Exception("Slot is not a function");

    return cast;
}

} // namespace Datacratic
