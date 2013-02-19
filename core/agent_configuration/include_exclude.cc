/* include_exclude.cc
   Jeremy Barnes, 8 March 2012
   Copyright (c) 2012 Datacratic.  All rights reserved.

*/

#include "include_exclude.h"
#include "rtbkit/common/segments.h"

namespace RTBKIT {

std::string jsonToString(const Json::Value & value)
{
    return value.asString();
}

void jsonParse(const Json::Value & value, boost::u32regex & rex)
{
    rex = boost::make_u32regex(value.asString());
}

void jsonParse(const Json::Value & value, boost::regex & rex)
{
    rex = value.asString();
}

void jsonParse(const Json::Value & value, std::string & str)
{
    str = value.asString();
}

void jsonParse(const Json::Value & value, int & i)
{
    i = value.asInt();
}

bool matchesAnyAny(const std::vector<int> & values, const SegmentList & vals,
                   bool matchIfEmpty)
{
    if (values.empty()) return matchIfEmpty;
    return vals.match(values);
}

template class IncludeExclude<std::string>;
template class IncludeExclude<boost::regex>;
template class IncludeExclude<int>;
template class IncludeExclude<boost::u32regex>;

#if 0

    // Structure to match a value against an include and exclude list of
    // regular expressions, caching the result of the computations.
    struct RegexMatcher {
        RegexMatcher()
        {
        }

        bool isIncluded(const std::string & val,
                        const std::vector<boost::regex> & include,
                        const std::vector<boost::regex> & exclude)
        {
            bool included = include.empty();
            for (unsigned i = 0;  !included && i < include.size();  ++i) {
                const boost::regex & rex = include[i];
                string rexStr = rex.str();
                bool regexResult;
                if (regexCache.count(rexStr))
                    regexResult = regexCache[rexStr];
                else {
                    regexResult = regexCache[rexStr]
                        = boost::regex_search(val, rex);
                }
                    
                included = regexResult;
            }
                
            if (!included) return false;
                
            bool excluded = false;
            for (unsigned i = 0;  !excluded && i < exclude.size();  ++i) {
                const boost::regex & rex = exclude[i];
                string rexStr = rex.str();
                bool regexResult;
                if (regexCache.count(rexStr))
                    regexResult = regexCache[rexStr];
                else {
                    regexResult = regexCache[rexStr]
                        = boost::regex_search(val, rex);
                }
                    
                excluded = regexResult;
            }

            if (excluded) return false;

            return true;
        }  

        std::map<std::string, bool> regexCache;
    };

    RegexMatcher urlMatcher, locationMatcher, languageMatcher;

#endif

} // namespace RTBKIT
