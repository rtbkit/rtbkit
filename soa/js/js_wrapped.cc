/* js_wrapped.cc
   Jeremy Barnes, 15 July 2010
   Copyright (c) 2010 Datacratic.  All rights reserved.

   Implementation of javascript wrapper.

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

#include "js_wrapped.h"
#include <iostream>


using namespace std;
using namespace v8;


namespace Datacratic {
namespace JS {


const std::type_info & getWrapperTypeInfo(v8::Handle<v8::Value> handle)
{
    ExcAssert(!handle.IsEmpty());
    ExcAssert(handle->IsObject());

    v8::Handle<v8::Object> object(v8::Object::Cast(*handle));

    ExcAssert(object->InternalFieldCount() == 2);

    return *static_cast<const std::type_info *>
        (v8::Handle<v8::External>::Cast
         (object->GetInternalField(1))->Value());
}



/*****************************************************************************/
/* JSWRAPPEDBASE                                                             */
/*****************************************************************************/

JSWrappedBase::
~JSWrappedBase ()
{
    if (js_object_.IsEmpty()) return;
    ExcAssert(js_object_.IsNearDeath());
    js_object_->SetInternalField(0, v8::Undefined());
    js_object_->SetInternalField(1, v8::Undefined());
    js_object_.Dispose();
    js_object_.Clear();
}

void
JSWrappedBase::
wrap(v8::Handle<v8::Object> handle,
     size_t size_in_bytes,
     const std::type_info & wrappedType)
{
    size_in_bytes_ = size_in_bytes;
    ExcAssert(js_object_.IsEmpty());

    if (handle->InternalFieldCount() == 0) {
        throw ML::Exception("InternalFieldCount is zero; are you forgetting "
                            "to use 'new' for " + getJsTypeName());
    }

    ExcAssert(handle->InternalFieldCount() == 2);

    js_object_ = v8::Persistent<v8::Object>::New(handle);
    js_object_->SetInternalField(0, v8::External::New(this));
    js_object_->SetInternalField(1, v8::External::New((void *)&wrappedType));
    v8::V8::AdjustAmountOfExternalAllocatedMemory(size_in_bytes);
    registerForGarbageCollection();
}

void
JSWrappedBase::
registerForGarbageCollection()
{
    js_object_.MakeWeak(this, getGarbageCollectionCallback());
}

void
JSWrappedBase::
ref()
{
    ExcAssert(!js_object_.IsEmpty());
    refs_++;
    js_object_.ClearWeak();
}

void
JSWrappedBase::
unref()
{
    ExcAssert(!js_object_.IsEmpty());
    ExcAssert(!js_object_.IsWeak());
    ExcAssert(refs_ > 0);
    if (--refs_ == 0) { registerForGarbageCollection(); }
}

void
JSWrappedBase::
dispose()
{
    //cerr << "disposing of " << ML::type_name(*this) << " at " << this
    //     << endl;
    delete this;
}

v8::WeakReferenceCallback
JSWrappedBase::
getGarbageCollectionCallback() const
{
    return garbageCollectionCallback;
}
   
void
JSWrappedBase::
garbageCollectionCallback(v8::Persistent<v8::Value> value, void * data)
{
    try {
        JSWrappedBase * obj = reinterpret_cast<JSWrappedBase *>(data);
        ExcAssert(value == obj->js_object_);
        ExcAssert(!obj->refs_);
        if (value.IsNearDeath()) {
            v8::V8::AdjustAmountOfExternalAllocatedMemory(-obj->size_in_bytes_);
            obj->size_in_bytes_ = 0;
            obj->dispose();
        }
    } catch (const std::exception & exc) {
        cerr << "WARNING: exception thrown in GC: " << exc.what() << endl;
    } catch (...) {
        cerr << "WARNING: exception thrown in GC: unknown" << endl;
    }
}

} // namespace JS
} // namespace Datacratic
