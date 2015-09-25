/* include_exclude.h                                               -*- C++ -*-
   Jeremy Barnes, 8 March 2012
   Copyright (c) 2012 Datacratic.  All rights reserved.

   Deals with lists of include/exclude items.
*/

#ifndef __rtb_router__include_exclude_h__
#define __rtb_router__include_exclude_h__

#include "jml/arch/exception.h"
#include "jml/utils/lightweight_hash.h"
#include "soa/types/url.h"
#include "soa/jsoncpp/value.h"
#include <boost/regex.hpp>
#include <boost/regex/icu.hpp>
#include "soa/types/string.h"
#include <vector>
#include <set>
#include <iostream>


namespace RTBKIT {

using namespace Datacratic;

struct SegmentList;

template<typename T>
inline bool matches(const T & t1, const T & t2)
{
    return t1 == t2;
}

inline bool matches(const boost::u32regex & rex, const Utf8String & val)
{
    std::string raw(val.rawData(), val.rawLength());
    boost::match_results<std::string::const_iterator> matches;
    bool result = boost::u32regex_search(raw, matches, rex) ;
    return result;
}

inline bool matches(const boost::u32regex & rex, const Utf32String & val)
{
    std::string raw;
    utf8::utf32to8(val.begin(), val.end(), std::back_inserter(raw));
    boost::match_results<std::string::const_iterator> matches;
    bool result = boost::u32regex_search(raw, matches, rex) ;
    return result;
}

inline bool matches(const boost::regex & rex, const std::string & val)
{
    //using namespace std;
    //cerr << "matching " << val << " with rex " << rex.str() << endl;
    return boost::regex_search(val, rex);
}
#if 0
inline bool matches(const std::string & str, const std::string & val)
{
    return str == val;
}

inline bool matches(int i, int j)
{
    return i == j;
}
#endif

void jsonParse(const Json::Value & value, boost::regex & reg);
void jsonParse(const Json::Value & value, boost::u32regex & reg);
void jsonParse(const Json::Value & value, std::string & str);
void jsonParse(const Json::Value & value, int & i);

inline Json::Value jsonPrint(const boost::regex & rex)
{
    return rex.str();
}

inline Json::Value jsonPrint(const boost::u32regex & rex)
{
    std::vector<unsigned char> utf8result;
    std::basic_string<int,std::char_traits<int>, std::allocator<int> >
        unicodeStr = rex.str();
    utf8::utf32to8(unicodeStr.begin(),
                   unicodeStr.begin() + unicodeStr.length(),
                   std::back_inserter(utf8result));
    Utf8String utf8str(std::string(utf8result.begin(), utf8result.end()));
    return utf8str;
}


inline Json::Value jsonPrint(const std::string & str)
{
    return str;
}

inline Json::Value jsonPrint(int i)
{
    return i;
}

struct JsonPrint {
    template<typename T>
    Json::Value operator () (const T & t) const
    {
        return jsonPrint(t);
    }
};

inline uint64_t hashString(const std::string & str)
{
    uint64_t res = std::hash<std::string>()(str);
    //cerr << "hashString of " << str << " returned " << res << endl;
    return res;
}

inline uint64_t hashString(const Utf8String & str)
{
    return std::hash<std::string>()(std::string(str.rawData(), str.rawLength()));
}


inline uint64_t hashString(const Utf32String & str)
{
    return std::hash<std::u32string>()(str.rawString());
}

inline uint64_t hashString(const std::wstring & str)
{
    uint64_t res = std::hash<std::wstring>()(str);
    return res;
}

inline void createRegex(boost::u32regex & regex, const wchar_t * str)
{
    regex = boost::make_u32regex(str);
}

inline void createRegex(boost::regex & regex, const std::string & str)
{
    regex = boost::regex(str);
}

/*****************************************************************************/
/* CACHED REGEX                                                              */
/*****************************************************************************/

template<typename Base, typename Str>
struct CachedRegex {
    Base base;
    uint64_t hash;
    
    CachedRegex()
        : hash(0)
    {
    }

    template<typename InitStr>
    CachedRegex(const InitStr & val)
        : hash(hashString(val))
    {
        createRegex(base, val);
    }

    void jsonParse(const Json::Value & val)
    {
        RTBKIT::jsonParse(val, base);
        hash = std::hash<std::string>() (val.asString());
    }

    bool operator < (const CachedRegex & other) const
    {
        return base < other.base;
    }

    bool matches(const Str & str) const
    {
        return RTBKIT::matches(base, str);
    }

    bool matches(const Str & str, uint64_t strHash,
                 ML::Lightweight_Hash<uint64_t, int> & cache) const
    {
        uint64_t bucket = hash ^ (strHash >> 1);
        bucket += (bucket == 0);
        int & cached = cache[bucket];
        if (cached == 0)
            cached = RTBKIT::matches(base, str) + 1;
        return cached - 1;
    }
};

template<typename Base, typename Str>
void jsonParse(const Json::Value & value, CachedRegex<Base, Str> & rex)
{
    rex.jsonParse(value);
}

template<typename Base, typename Str>
inline Json::Value jsonPrint(const CachedRegex<Base, Str> & rex)
{
    return jsonPrint(rex.base);
}

template<typename Base, typename Str, typename Cache>
inline bool matches(const CachedRegex<Base, Str> & rex,
                    const Str & str, uint64_t strHash,
                    Cache & cache)
{
    if (strHash == 0)
        throw ML::Exception("zero string hash");
    return rex.matches(str, strHash, cache);
}

template<typename Base, typename Str>
inline bool matches(const CachedRegex<Base, Str> & rex, const Str & str)
{
    return rex.matches(str);
}


/*****************************************************************************/
/* URL MATCHER                                                               */
/*****************************************************************************/

struct DomainMatcher {
    boost::regex rex;
    bool isLiteral;
    std::string str;
    uint64_t hash;
    
    DomainMatcher()
        : hash(0)
    {
    }

    DomainMatcher(const std::string & val)
        : str(val), hash(std::hash<std::string>() (val))
    {
        isLiteral = true;
        return;
        if (false
            && val.find('*') == std::string::npos
            && val.find('/') == std::string::npos
            && val.find('?') == std::string::npos
            && val.find(':') == std::string::npos
            && val.find('.') != std::string::npos) {
            isLiteral = true;
        }
        else {
            rex = boost::regex(val);
            isLiteral = false;
        }
    }

    void jsonParse(const Json::Value & val)
    {
        std::string s = val.asString();
        *this = DomainMatcher(s);
    }

    bool operator < (const DomainMatcher & other) const
    {
        return str < other.str;
    }

    bool matches(const Url & url) const
    {
        if (isLiteral)
            return url.domainMatches(str);
        else return RTBKIT::matches(rex, url.host());
    }

    bool matches(const Url & url, uint64_t urlHash,
                 ML::Lightweight_Hash<uint64_t, int> & cache) const
    {
        uint64_t bucket = hash ^ (urlHash >> 1);
        bucket += (bucket == 0);
        int & cached = cache[bucket];
        if (cached == 0)
            cached = matches(url) + 1;
        return cached - 1;
    }
};

inline void jsonParse(const Json::Value & value, DomainMatcher & rex)
{
    rex.jsonParse(value);
}

inline Json::Value jsonPrint(const DomainMatcher & rex)
{
    return rex.str;
}

template<typename Cache>
inline bool matches(const DomainMatcher & rex, const Url & url,
                    uint64_t urlHash,
                    Cache & cache)
{
    return rex.matches(url, urlHash, cache);
}

inline bool matches(const DomainMatcher & rex, const Url & url)
{
    return rex.matches(url);
}


/*****************************************************************************/
/* MATCHING AND PARSING FUNCTIONS                                            */
/*****************************************************************************/

template<typename T, typename Fn>
Json::Value
collectionToJson(const std::vector<T> & vec, Fn fn)
{
    Json::Value result;
    for (unsigned i = 0;  i < vec.size();  ++i)
        result[i] = fn(vec[i]);
    return result;
}

template<typename T, typename Fn>
Json::Value
collectionToJson(const std::set<T> & s, Fn fn)
{
    Json::Value result;
    unsigned i = 0;
    for (auto it = s.begin(), end = s.end();  it != end;  ++it, ++i)
        result[i] = fn(*it);
    return result;
}

template<typename Collection, typename Fn>
Json::Value
includeExcludeToJson(const Collection & include,
                     const Collection & exclude,
                     Fn fn)
{
    Json::Value result;
    if (!include.empty())
        result["include"] = collectionToJson(include, fn);
    if (!exclude.empty())
        result["exclude"] = collectionToJson(exclude, fn);
    return result;
}

template<typename T, typename U>
bool matchesAny(const std::vector<T> & values, const U & key, bool matchIfEmpty)
{
    if (values.empty())
    {
    	return matchIfEmpty;
    }
    for (unsigned i = 0;  i < values.size();  ++i)
    {
        if (matches(values[i], key)) return true;
    }
    return false;
}

template<typename T, typename U, typename Cache>
bool matchesAny(const std::vector<T> & values,
                const U & key, uint64_t keyHash,
                bool matchIfEmpty,
                Cache & cache)
{
    if (values.empty())
    {
    	return matchIfEmpty;
    }
    for (unsigned i = 0;  i < values.size();  ++i)
    {
        if (matches(values[i], key, keyHash, cache)) return true;
    }
    return false;
}

// TODO: this is O(mn) but could be O(n+m) since they are sorted
template<typename T, class Vec>
bool matchesAnyAny(const std::vector<T> & values, const Vec & vec,
                   bool matchIfEmpty)
{
    if (values.empty()) return matchIfEmpty;
    
    for (auto it = vec.begin(), end = vec.end(); it != end; ++it)
        if (matchesAny(values, *it, matchIfEmpty)) return true;
    return false;
}

bool matchesAnyAny(const std::vector<int> & values, const SegmentList & vals,
                   bool matchIfEmpty);

enum IncludeExcludeResult {
    IE_NO_DATA,
    IE_NOT_INCLUDED,
    IE_EXCLUDED,
    IE_PASSED
};

template<typename U, typename IE>
bool isIncludedImpl(const U & value, const IE & include, const IE & exclude)
{
    if (!matchesAny(include, value, true)) return false;
    if (matchesAny(exclude, value, false)) return false;
    return true;
}

template<typename U, typename IE, typename Cache>
bool isIncludedImpl(const U & value, uint64_t hash,
                    const IE & include, const IE & exclude,
                    Cache & cache)
{
    if (!matchesAny(include, value, hash, true, cache)) return false;
    if (matchesAny(exclude, value, hash, false, cache)) return false;
    return true;
}

template<typename Vec, typename IE>
bool anyIsIncludedImpl(const Vec & vec, const IE & include, const IE & exclude)
{
    if (!matchesAnyAny(include, vec, true)) return false;
    if (matchesAnyAny(exclude, vec, false)) return false;
    return true;
}



/*****************************************************************************/
/* INCLUDE EXCLUDE                                                           */
/*****************************************************************************/

template<typename T, typename IE = std::vector<T> >
struct IncludeExclude {
    IE include;
    IE exclude;

    static IncludeExclude
    createFromJson(const Json::Value & val,
                   const std::string & name)
    {
        IncludeExclude result;

        for (auto jt = val.begin(), jend = val.end();  jt != jend;  ++jt) {
            if (jt.memberName() != "include"
                && jt.memberName() != "exclude")
                throw ML::Exception("filter %s has invalid key: %s",
                                    name.c_str(), jt.memberName().c_str());
            
            const Json::Value & val = *jt;
            if(!val.isNull() && val.isArray()) {

                for (unsigned i = 0;  i != val.size();  ++i) {
                    try {
                        T t;
                        jsonParse(val[i], t);
                        if (jt.memberName() == "include")
                            result.include.push_back(t);
                        else result.exclude.push_back(t);
                    } catch (...) {
                        throw ML::Exception("error parsing include/exclude %s in %s",
                                            val[i].toString().c_str(), name.c_str());
                    }
                }
            }else {
                throw ML::Exception("error parsing include/exclude : "
                        "include/exclude must be an array");
            }
        }

        std::sort(result.include.begin(), result.include.end());
        std::sort(result.exclude.begin(), result.exclude.end());

        return result;
    }

    void fromJson(const Json::Value & val, const std::string & name)
    {
        *this = createFromJson(val, name);
    }

    Json::Value toJson() const
    {
        Json::Value result = includeExcludeToJson(include, exclude, JsonPrint());
        return result;
    }

    bool empty() const { return include.empty() && exclude.empty(); }

    template<typename U>
    bool isIncluded(const U & value) const
    {
        return isIncludedImpl(value, include, exclude);
    }
    
    template<typename U, typename Cache>
    bool isIncluded(const U & value, uint64_t hash, Cache & cache) const
    {
        return isIncludedImpl(value, hash, include, exclude, cache);
    }
    
    template<typename Vec>
    bool anyIsIncluded(const Vec & vec) const
    {
        return anyIsIncludedImpl(vec, include, exclude);
    }
};

extern template class IncludeExclude<std::string>;
extern template class IncludeExclude<boost::regex>;
extern template class IncludeExclude<boost::u32regex>;
extern template class IncludeExclude<int>;

} // namespace RTBKIT

#endif /* __rtb_router__include_exclude_h__ */
