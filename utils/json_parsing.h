/* json_parsing.h                                                  -*- C++ -*-
   Jeremy Barnes, 1 February 2012
   Copyright (c) 2012 Datacratic Inc.  All rights reserved.

   Released under the MIT license.

   Functionality to ease parsing of JSON from within a parse_function.
*/

#ifndef __jml__utils__json_parsing_h__
#define __jml__utils__json_parsing_h__

#include <string>
#include <functional>
#include "parse_context.h"
#include <boost/lexical_cast.hpp>


namespace ML {

/*****************************************************************************/
/* JSON UTILITIES                                                            */
/*****************************************************************************/

std::string jsonEscape(const std::string & str);

void jsonEscape(const std::string & str, std::ostream & out);

/*
 * If non-ascii characters are found an exception is thrown
 */
std::string expectJsonStringAscii(Parse_Context & context);

/*
 * If non-ascii characters are found an exception is thrown.
 * Output goes into the given buffer, of the given maximum length.
 * If it doesn't fit, then return zero.
 */
ssize_t expectJsonStringAscii(Parse_Context & context, char * buf,
                             size_t maxLength);

/*
 * if non-ascii characters are found we replace them by an ascii character that is supplied
 */
std::string expectJsonStringAsciiPermissive(Parse_Context & context, char c);

bool matchJsonString(Parse_Context & context, std::string & str);

bool matchJsonNull(Parse_Context & context);

void
expectJsonArray(Parse_Context & context,
                const std::function<void (int, Parse_Context &)> & onEntry);

void
expectJsonObject(Parse_Context & context,
                 const std::function<void (std::string, Parse_Context &)> & onEntry);

/** Expect a Json object and call the given callback.  The keys are assumed
    to be ASCII which means no embedded nulls, and so the key can be passed
    as a const char *.
*/
void
expectJsonObjectAscii(Parse_Context & context,
                      const std::function<void (const char *, Parse_Context &)> & onEntry);

bool
matchJsonObject(Parse_Context & context,
                const std::function<bool (std::string, Parse_Context &)> & onEntry);

void skipJsonWhitespace(Parse_Context & context);

inline bool expectJsonBool(Parse_Context & context)
{
    if (context.match_literal("true"))
        return true;
    else if (context.match_literal("false"))
        return false;
    context.exception("expected bool (true or false)");
}

#ifdef CPPTL_JSON_H_INCLUDED

inline Json::Value
expectJson(Parse_Context & context)
{
    context.skip_whitespace();
    if (*context == '"')
        return expectJsonStringAscii(context);
    else if (context.match_literal("null"))
        return Json::Value();
    else if (context.match_literal("true"))
        return Json::Value(true);
    else if (context.match_literal("false"))
        return Json::Value(false);
    else if (*context == '[') {
        Json::Value result(Json::arrayValue);
        expectJsonArray(context,
                        [&] (int i, Parse_Context & context)
                        {
                            result[i] = expectJson(context);
                        });
        return result;
    } else if (*context == '{') {
        Json::Value result(Json::objectValue);
        expectJsonObject(context,
                         [&] (std::string key, Parse_Context & context)
                         {
                             result[key] = expectJson(context);
                         });
        return result;
    } else {
        Json::Value result;

        std::string number;
        bool negative = false;
        bool doublePrecision = false;

        if (context.match_literal('-')) {
            number += '-';
            negative = true;
        }

        while (context && isdigit(*context)) {
            number += *context++;
        }

        if (context.match_literal('.')) {
            doublePrecision = true;
            number += '.';

            while (context && isdigit(*context)) {
                number += *context++;
            }
        }

        char sci = context ? *context : '\0';
        if (sci == 'e' || sci == 'E') {
            doublePrecision = true;
            number += *context++;

            char sign = context ? *context : '\0';
            if (sign == '+' || sign == '-') {
                number += *context++;
            }

            while (context && isdigit(*context)) {
                number += *context++;
            }
        }

        if (doublePrecision) {
            result = boost::lexical_cast<double>(number);
        } else if (negative) {
            result = boost::lexical_cast<long long>(number);
        } else {
            result = boost::lexical_cast<unsigned long long>(number);
        }

        return result;
    }
}

#endif

} // namespace ML


#endif /* __jml__utils__json_parsing_h__ */

