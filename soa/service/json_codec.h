/* json_codec.h                                                    -*- C++ -*-
   Jeremy Banres, 26 November 2012
   Copyright (c) 2012 Datacratic.  All rights reserved.

   JSON encoding/decodign code.
*/

#pragma once

#include "soa/jsoncpp/json.h"
#include <vector>
#include <map>
#include <unordered_map>
#include "jml/utils/exc_assert.h"
#include "soa/types/basic_value_descriptions.h"

namespace Datacratic {

// Define JSON encodability for default types

#define DO_JSON_ENCODABLE(T, extract) \
    inline Json::Value jsonEncode(const T & obj) { return Json::Value(obj); } \
    inline T jsonDecode(const Json::Value & j, T * = 0) { return j.extract(); }

DO_JSON_ENCODABLE(std::string, asString);
DO_JSON_ENCODABLE(unsigned int, asInt);
DO_JSON_ENCODABLE(signed int, asUInt);
DO_JSON_ENCODABLE(short unsigned int, asInt);
DO_JSON_ENCODABLE(short signed int, asUInt);
DO_JSON_ENCODABLE(long unsigned int, asInt);
DO_JSON_ENCODABLE(long signed int, asUInt);
DO_JSON_ENCODABLE(long long unsigned int, asInt);
DO_JSON_ENCODABLE(long long signed int, asUInt);
DO_JSON_ENCODABLE(bool, asBool);
DO_JSON_ENCODABLE(double, asDouble);
DO_JSON_ENCODABLE(float, asDouble);

inline Json::Value jsonEncode(Json::Value v)
{
    return v;
}

inline Json::Value jsonDecode(Json::Value v, Json::Value * = 0)
{
    return v;
}

// Anything with a toJson() method gets to be jsonEncoded
template<typename T>
Json::Value jsonEncode(const T & obj,
                       decltype(std::declval<T>().toJson()) * = 0)
{
    return obj.toJson();
}

// Anything with a static fromJson() method gets to be jsonDecoded
template<typename T>
T jsonDecode(const Json::Value & json, T * = 0,
             decltype(T::fromJson(std::declval<Json::Value>())) * = 0)
{
    return T::fromJson(json);
}

template<typename T>
Json::Value jsonEncode(const std::vector<T> & vec)
{
    Json::Value result(Json::arrayValue);
    for (unsigned i = 0;  i < vec.size();  ++i)
        result[i] = jsonEncode(vec[i]);
    return result;
}

template<typename T>
std::vector<T> jsonDecode(const Json::Value & val, std::vector<T> *)
{
    ExcAssert(val.isArray());
    std::vector<T> res;
    res.reserve(val.size());
    for (unsigned i = 0;  i < val.size();  ++i)
        res.push_back(jsonDecode(val[i], (T*)0));
    return res;
}

template<typename T>
Json::Value jsonEncode(const std::map<std::string, T> & m)
{
    Json::Value result(Json::objectValue);
    for (auto & e: m)
        result[e.first] = jsonEncode(e.second);
    return result;
}

template<typename T>
std::map<std::string, T>
jsonDecode(const Json::Value & val, std::map<std::string, T> *)
{
    std::map<std::string, T> res;
    if (val.isNull())
        return res;

    ExcAssert(val.isObject());

    for (auto it = val.begin(), end = val.end();  it != end;  ++it)
        res.emplace(it.memberName(), jsonDecode(*it, (T *)0));

    return res;
}

template<typename T>
Json::Value jsonEncode(const std::unordered_map<std::string, T> & m)
{
    Json::Value result(Json::objectValue);
    for (auto & e: m)
        result[m.first] = jsonEncode(m.second);
    return result;
}

template<typename T>
std::unordered_map<std::string, T>
jsonDecode(const Json::Value & val, std::unordered_map<std::string, T> *)
{
    std::unordered_map<std::string, T> res;
    if (val.isNull())
        return res;

    ExcAssert(val.isObject());

    for (auto it = val.begin(), end = val.end();  it != end;  ++it)
        res.insert(std::make_pair(it.memberName(), jsonDecode(*it, (T *)0)));

    return res;
}

template<typename T, typename Enable = void>
struct JsonCodec {
    static T decode(const Json::Value & val)
    {
        return jsonDecode(val, (T *)0);
    }

    static Json::Value encode(const T & val)
    {
        return jsonEncode(val);
    }
};

template<typename T>
void getParam(const Json::Value & parameters,
              T & val,
              const std::string & name)
{
    if (parameters.isMember(name)) {
        Json::Value j = parameters[name];
        if (j.isNull())
            return;
        val = jsonDecode(j, &val);
    }
}

} // namespace Datacratic
