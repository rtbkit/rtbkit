/* js_value.h                                                      -*- C++ -*-
   Jeremy Barnes, 21 July 2010
   Copyright (c) 2010 Datacratic.  All rights reserved.

   Definition of what a javascript value is.
*/

#pragma once

#include "js_value_fwd.h"
#include "v8/v8.h"

namespace Datacratic {
namespace JS {


/*****************************************************************************/
/* JSVALUE                                                                   */
/*****************************************************************************/

/** Opaque type used to represent a Javascript value
*/

struct JSValue : public v8::Handle<v8::Value> {
    JSValue()
    {
    }

    template<typename T>
    JSValue(const v8::Local<T> & val)
        : v8::Handle<v8::Value>(val)
    {
    }

    template<typename T>
    JSValue(const v8::Handle<T> & val)
        : v8::Handle<v8::Value>(val)
    {
    }

    operator v8::Handle<v8::Object>() const;
};


/*****************************************************************************/
/* JSVALUE                                                                   */
/*****************************************************************************/

/** Opaque type used to represent a Javascript object.  Convertible to a
    Javscript value.
*/

struct JSObject : public v8::Handle<v8::Object> {
    JSObject()
    {
    }

    template<typename T>
    JSObject(const v8::Local<T> & val)
        : v8::Handle<v8::Object>(val)
    {
    }

    template<typename T>
    JSObject(const v8::Handle<T> & val)
        : v8::Handle<v8::Object>(val)
    {
    }

    void initialize();

    void add(const std::string & key, const std::string & value);
    void add(const std::string & key, const JSValue & value);
};

} // namespace JS
} // namespace Datacratic
