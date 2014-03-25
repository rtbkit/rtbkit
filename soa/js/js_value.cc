/* js_value.cc
   Jeremy Barnes, 21 July 2010
   Copyright (c) 2010 Datacratic.  All rights reserved.

   Javascript value handling.
*/

#include "js_value.h"
#include "js_utils.h"
#include "jml/utils/string_functions.h"
#include "jml/arch/demangle.h"
#include "jml/arch/backtrace.h"
#include "soa/types/date.h"
#include "soa/types/string.h"
#include <boost/lexical_cast.hpp>
#include <boost/foreach.hpp>
#include "soa/jsoncpp/json.h"
#include "node/node_buffer.h"
#include <cxxabi.h>
using namespace std;
using namespace ML;
using namespace Datacratic::JS;

namespace node {

// Define as a weak symbol to avoid linker errors when linking without node
__attribute__((__weak__))
Buffer * Buffer::New(size_t) 
{
    throw Exception("node needs to be linked in to use non-ASCII strings");
}

__attribute__((__weak__))
bool Buffer::HasInstance(v8::Handle<v8::Value> val) 
{
    return false;  // if node isn't linked in, then it can't be a buffer
}
 
} // namespace node

namespace Datacratic {
namespace JS {


/*****************************************************************************/
/* JSVALUE                                                                   */
/*****************************************************************************/

JSValue::operator v8::Handle<v8::Object>() const
{
    return toObject(*this);
}


/*****************************************************************************/
/* JSOBJECT                                                                  */
/*****************************************************************************/

void
JSObject::
initialize()
{
    *this = v8::Object::New();
}

void
JSObject::
add(const std::string & key, const std::string & value)
{
    (*this)->Set(v8::String::NewSymbol(key.c_str()),
                 v8::String::New(value.c_str(), value.length()));
}

void
JSObject::
add(const std::string & key, const JSValue & value)
{
    (*this)->Set(v8::String::NewSymbol(key.c_str()),
                 value);
}


/*****************************************************************************/
/* CONVERSIONS                                                               */
/*****************************************************************************/

void to_js(JSValue & jsval, signed int value)
{
    jsval = v8::Integer::New(value);
}

void to_js(JSValue & jsval, unsigned int value)
{
    jsval = v8::Integer::NewFromUnsigned(value);
}

void to_js(JSValue & jsval, signed long value)
{
    if (value <= INT_MAX && value >= INT_MIN)
        jsval = v8::Integer::New(value);
    else jsval = v8::Number::New((double) value);
}

void to_js(JSValue & jsval, unsigned long value)
{
    if (value <= UINT_MAX)
        jsval = v8::Integer::NewFromUnsigned(value);
    else jsval = v8::Number::New((double) value);
}

void to_js(JSValue & jsval, signed long long value)
{
    if (value <= INT_MAX && value >= INT_MIN)
        jsval = v8::Integer::New(value);
    else jsval = v8::Number::New((double) value);
}

void to_js(JSValue & jsval, unsigned long long value)
{
    if (value <= UINT_MAX)
        jsval = v8::Integer::NewFromUnsigned(value);
    else jsval = v8::Number::New((double) value);
}

void to_js(JSValue & jsval, float value)
{
    jsval = v8::Number::New(value);
}

void to_js(JSValue & jsval, double value)
{
    jsval = v8::Number::New(value);
}

void to_js_bool(JSValue & jsval, bool value)
{
    jsval = v8::Boolean::New(value);
}

void to_js(JSValue & jsval, const std::string & value)
{
    bool isAscii = true;
    for (unsigned i = 0;  i < value.size() && isAscii;  ++i)
        if (value[i] == 0 || value[i] > 127)
            isAscii = false;
    if (isAscii)
        jsval = v8::String::New(value.c_str(), value.length());
    else {
        // We can't represent this in ASCII.  In this case, we need to use a
        // buffer.
        node::Buffer * buffer
            = node::Buffer::New(value.size());
        std::copy(value.begin(), value.end(), node::Buffer::Data(buffer));
        jsval = buffer->handle_;
    }
}

void to_js(JSValue & jsval, const Utf8String & value)
{
	jsval = v8::String::New(value.rawData(), value.rawLength());
}

void to_js(JSValue & jsval, const Utf32String & value)
{
    std::string utf8Str { value.utf8String() };
	jsval = v8::String::New(utf8Str.c_str(), utf8Str.size());
}
void to_js(JSValue & jsval, const char * value)
{
    jsval = v8::String::New(value);
}

void to_js(JSValue & jsval, const Json::Value & value)
{
    switch(value.type())
    {
    case Json::objectValue:
    {
        v8::HandleScope scope;
        v8::Local<v8::Object> obj = v8::Object::New();
        BOOST_FOREACH(string key, value.getMemberNames())
        {
            JSValue member;
            to_js(member, value[key]);
            obj->Set(v8::String::NewSymbol(key.c_str()), member);
        }
        jsval = scope.Close(obj);
    }
        break;
    case Json::arrayValue:
    {
        v8::HandleScope scope;
        v8::Local<v8::Array> arr = v8::Array::New();
        for(int i=0;i< value.size(); ++i)
        {
            JSValue elem;
            to_js(elem, value[i]);
            arr->Set(i, elem);
        }
        jsval = scope.Close(arr);
    }
        break;
    case Json::realValue:
        to_js(jsval, value.asDouble());
        break;
    case Json::stringValue:
        to_js(jsval, value.asString());
        break;
    case Json::intValue:
        to_js(jsval, value.asInt());
        break;
    case Json::uintValue:
        to_js(jsval, value.asUInt());
        break;
    case Json::booleanValue:
        to_js(jsval, value.asBool());
        break;
    case Json::nullValue:
        jsval = v8::Null();
        break;
    default:
        throw ML::Exception("Can't convert from JsonCpp to JSValue");
        break;
    }
}

void to_js(JSValue & jsval, Date value)
{
    jsval = v8::Date::New(value.secondsSinceEpoch() * 1000.0);
}

namespace {

int64_t check_to_int2(const JSValue & val)
{
    //cerr << "check_to_int " << cstr(val) << endl;

    int64_t ival = val->IntegerValue();
    double dval = val->NumberValue();

    //cerr << "  ival = " << ival << endl;
    //cerr << "  dval = " << dval << endl;

    if (ival != 0 && ival == dval) return ival;

    if (dval > std::numeric_limits<uint64_t>::max()
        || dval < std::numeric_limits<uint64_t>::min())
        throw ML::Exception("Cannot fit " + cstr(val) + " into an integer");
        
    v8::Local<v8::Number> num;

    if (val->IsArray())
        throw Exception("cannot convert array to integer");

    //bool debug = val->IsArray();
    //if (debug) cerr << "is array" << endl;

    if (val->IsNumber() || v8::Number::Cast(*val)) {
        //if (debug)
        //cerr << "is number" << endl;
        double d = val->NumberValue();
        if (!isfinite(d))
            throw Exception("cannot convert double value "
                            + cstr(val) + " to integer");
        return d;
    }
    if (val->IsString()) {
        //if (debug)
        //cerr << "is string" << endl;

        int64_t ival = val->IntegerValue();
        if (ival != 0) return ival;
        string s = lowercase(cstr(val));
        try {
            return boost::lexical_cast<int64_t>(s);
        } catch (const boost::bad_lexical_cast & error) {
            throw Exception("cannot convert string value \""
                            + cstr(val) + "\" (\"" + s + "\") to integer");
        }
    }

#define try_type(x) if (val->x()) cerr << #x << endl;

    try_type(IsUndefined);
    try_type(IsNull);
    try_type(IsTrue);
    try_type(IsFalse);
    try_type(IsString);
    try_type(IsFunction);
    try_type(IsArray);
    try_type(IsObject);
    try_type(IsBoolean);
    try_type(IsNumber);
    try_type(IsExternal);
    try_type(IsInt32);
    try_type(IsDate);

    if (val->IsObject()) {
        cerr << "object: " << cstr(val->ToObject()->ObjectProtoToString())
             << endl;
        cerr << "val->NumberValue() = " << val->NumberValue() << endl;
    }

    backtrace();

    throw Exception("cannot convert value \""
                    + cstr(val) + "\" to integer");
}

template<typename T>
T check_to_int(const JSValue & val)
{
    if (val.IsEmpty())
        throw Exception("from_js: value is empty");

    

    int64_t result1 = check_to_int2(val);
    T result2 = result1;
    if (result1 != result2)
        throw Exception("value " + cstr(val) + " does not fit in type "
                        + ML::type_name<T>());
    return result2;
}

} // file scope

signed int from_js(const JSValue & val, signed int *)
{
    //cerr << "from_js signed int" << endl;
    return check_to_int<signed int>(val);
}

unsigned int from_js(const JSValue & val, unsigned *)
{
    //cerr << "from_js unsigned int" << endl;
    return check_to_int<unsigned int>(val);
}

signed long from_js(const JSValue & val, signed long *)
{
    //cerr << "from_js signed long" << endl;
    return check_to_int<signed long>(val);
}

unsigned long from_js(const JSValue & val, unsigned long *)
{
    //cerr << "from_js unsigned long" << endl;
    return check_to_int<unsigned long>(val);
}

signed long long from_js(const JSValue & val, signed long long *)
{
    //cerr << "from_js signed long" << endl;
    return check_to_int<signed long long>(val);
}

unsigned long long from_js(const JSValue & val, unsigned long long *)
{
    //cerr << "from_js unsigned long" << endl;
    return check_to_int<unsigned long long>(val);
}

float from_js(const JSValue & val, float *)
{
    //cerr << "from_js float" << endl;
    return from_js(val, (double *)0);
}

double from_js(const JSValue & val, double *)
{
    //cerr << "from_js double" << endl;
    const double result = val->NumberValue();
    if (std::isnan(result)) {
        if (val->IsNumber()) return result;
        if (val->IsString()) {
            string s = lowercase(cstr(val));
            if (s == "nan" || s == "-nan")
                return result;
            throw ML::Exception("string value \"%s\" is not converible to "
                                "floating point",
                                s.c_str());
        }
        throw Exception("value \"%s\" not convertible to floating point",
                        cstr(val).c_str());
    }
    return result;
}

bool from_js(const JSValue & val, bool *)
{
    bool result = val->BooleanValue();
    return result;
}

std::string from_js(const JSValue & val, std::string *)
{
    if (node::Buffer::HasInstance(val)) {
        //cerr << "from_js with buffer" << endl;
        return string(node::Buffer::Data(val), node::Buffer::Length(val));
    }
    else
    {
    	return *v8::String::AsciiValue(val);
    }
}

Json::Value from_js(const JSValue & val, Json::Value *)
{
    if (val.IsEmpty())
        throw ML::Exception("empty val");

    //cerr << cstr(val) << endl;

    if(val->IsObject())
    {
        if(v8::Date::Cast(*val)->IsDate())
        {
            return from_js(val, (Datacratic::Date*)(0)).secondsSinceEpoch();
        }
        if(val->IsArray())
        {
            Json::Value result (Json::arrayValue);

            auto arrPtr = v8::Array::Cast(*val);
            for(int i=0; i<arrPtr->Length(); ++i)
            {
                result[i] = from_js(arrPtr->Get(i), (Json::Value *)0);
            }

            return result;
        }
        else
        {
            Json::Value result (Json::objectValue);
            auto objPtr = v8::Object::Cast(*val);
            v8::Handle<v8::Array> prop_names = objPtr->GetPropertyNames();

            for (unsigned i = 0;  i < prop_names->Length();  ++i)
            {
                v8::Handle<v8::String> key
                    = prop_names->Get(v8::Uint32::New(i))->ToString();
                if (!objPtr->HasOwnProperty(key)) continue;
                result[from_js(key, (string *)0)] =
                        from_js(objPtr->Get(key), (Json::Value *)0);
            }

            return result;
        }
    }
    if(val->IsBoolean())
    {
        return from_js(val, (bool *)0);
    }
    if(val->IsString())
    {
       	return from_js(val, (Utf8String *)0);
    }
    if(val->IsInt32())
    {
        return from_js(val, (int32_t *)0);
    }
    if(val->IsUint32())
    {
        return from_js(val, (uint32_t *)0);
    }
    if(val->IsNumber())
    {
        return from_js(val, (double *)0);
    }
    if (val->IsNull() || val->IsUndefined())
        return Json::Value();
    throw ML::Exception("can't convert from JSValue %s to Json::Value",
                        cstr(val).c_str());
}

Date from_js(const JSValue & val, Date *)
{
    if(!v8::Date::Cast(*val)->IsDate())
        throw ML::Exception("Couldn't convert from " + cstr(val) + " to Datacratic::Date");
    return Date::fromSecondsSinceEpoch(v8::Date::Cast(*val)->NumberValue()
                                       / 1000.0);
}

Utf8String from_js(const JSValue & val, Utf8String *)
{
	return Utf8String(*v8::String::Utf8Value(val)) ;
}

Utf32String from_js(const JSValue & val, Utf32String *)
{
    return Utf32String(*v8::String::Utf8Value(val));
}

Json::Value from_js_ref(const JSValue & val, Json::Value *)
{
    return from_js(val, (Json::Value *)0);
}


} // namespace JS
} // namespace Datacratic
