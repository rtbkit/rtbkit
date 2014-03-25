/* js_value_fwd.h                                                  -*- C++ -*-
   Jeremy Barnes, 27 October 2010
   Copyright (c) 2010 Datacratic.  All rights reserved.

   Forward definitions for js_value.
*/

#ifndef __js__js_value_fwd_h__
#define __js__js_value_fwd_h__

#include <string>
#include "soa/types/string.h"
#include <boost/utility/enable_if.hpp>
#include <boost/type_traits/is_enum.hpp>
#include <boost/type_traits/is_same.hpp>

namespace Json {

struct Value;

} // namespace Json

namespace Datacratic {
namespace JS {

struct JSValue;
struct JSObject;

void to_js(JSValue & jsval, signed int value);
void to_js(JSValue & jsval, unsigned int value);
void to_js(JSValue & jsval, signed long value);
void to_js(JSValue & jsval, unsigned long value);
void to_js(JSValue & jsval, signed long long value);
void to_js(JSValue & jsval, unsigned long long value);
void to_js(JSValue & jsval, float value);
void to_js(JSValue & jsval, double value);
void to_js_bool(JSValue & jsval, bool value);
void to_js(JSValue & jsval, const char * value);
void to_js(JSValue & jsval, const std::string & value);
void to_js(JSValue & jsval, const Utf8String & value);
void to_js(JSValue & jsval, const Utf32String & value);
void to_js(JSValue & jsval, const Json::Value & value);
void to_js(JSValue & jsval, const JSValue & value);

/** Avoid implicit conversion from pointers to bool */
template<typename T>
void to_js(JSValue & jsval, T value,
           typename boost::enable_if<typename boost::is_same<T, bool>::type, void *>::type = 0)
{
    to_js_bool(jsval, value);
}

signed int from_js(const JSValue & val, signed int *);
unsigned int from_js(const JSValue & val, unsigned *);
signed long from_js(const JSValue & val, signed long *);
unsigned long from_js(const JSValue & val, unsigned long *);
signed long long from_js(const JSValue & val, signed long long *);
unsigned long long from_js(const JSValue & val, unsigned long long *);
float from_js(const JSValue & val, float *);
double from_js(const JSValue & val, double *);
std::string from_js(const JSValue & val, std::string *);
Utf8String from_js(const JSValue & val, Utf8String *);
Utf32String from_js(const JSValue & val, Utf32String *);
bool from_js(const JSValue & val, bool *);
Json::Value from_js(const JSValue & val, Json::Value *);
Json::Value from_js_ref(const JSValue & val, Json::Value *);

inline const JSValue & from_js(const JSValue & val, JSValue *) { return val; }

template<typename T>
typename boost::enable_if<typename boost::is_enum<T>, T>::type
from_js(const JSValue & val, T *)
{
    int ival = from_js(val, (int *)0);
    return static_cast<T>(ival);
}

} // namespace JS
} // namespace Datacratic


#endif /* __js__js_value_fwd_h__ */
