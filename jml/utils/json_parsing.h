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

std::string expectJsonString(Parse_Context & context);

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
                 const std::function<void (const std::string &, Parse_Context &)> & onEntry);

/** Expect a Json object and call the given callback.  The keys are assumed
    to be ASCII which means no embedded nulls, and so the key can be passed
    as a const char *.
*/
void
expectJsonObjectAscii(Parse_Context & context,
                      const std::function<void (const char *, Parse_Context &)> & onEntry);

bool
matchJsonObject(Parse_Context & context,
                const std::function<bool (const std::string &, Parse_Context &)> & onEntry);

void skipJsonWhitespace(Parse_Context & context);

inline bool expectJsonBool(Parse_Context & context)
{
    if (context.match_literal("true"))
        return true;
    else if (context.match_literal("false"))
        return false;
    context.exception("expected bool (true or false)");
}

/** Representation of a numeric value in JSON.  It's designed to allow
    it to be stored the same way it was written (as an integer versus
    floating point, signed vs unsigned) without losing precision.
*/
struct JsonNumber {
    enum Type {
        NONE,
        UNSIGNED_INT,
        SIGNED_INT,
        FLOATING_POINT
    } type;

    union {
        unsigned long long uns;
        long long sgn;
        double fp;
    };    
};

/** Expect a JSON number.  This function is written in this strange way
    because JsonCPP is not a require dependency of jml, but the function
    needs to be out-of-line.
*/
JsonNumber expectJsonNumber(Parse_Context & context);

/** Match a JSON number. */
bool matchJsonNumber(Parse_Context & context, JsonNumber & num);

#ifdef CPPTL_JSON_H_INCLUDED

inline Json::Value
expectJson(Parse_Context & context)
{
    context.skip_whitespace();
    if (*context == '"')
        return expectJsonString(context);
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
                         [&] (const std::string & key, Parse_Context & context)
                         {
                             result[key] = expectJson(context);
                         });
        return result;
    } else {
        JsonNumber number = expectJsonNumber(context);
        switch (number.type) {
        case JsonNumber::UNSIGNED_INT:
            return number.uns;
        case JsonNumber::SIGNED_INT:
            return number.sgn;
        case JsonNumber::FLOATING_POINT:
            return number.fp;
        default:
            throw ML::Exception("logic error in expectJson");
        }
    }
}

inline Json::Value
expectJsonAscii(Parse_Context & context)
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
                            result[i] = expectJsonAscii(context);
                        });
        return result;
    } else if (*context == '{') {
        Json::Value result(Json::objectValue);
        expectJsonObjectAscii(context,
                         [&] (const char * key, Parse_Context & context)
                         {
                             result[key] = expectJsonAscii(context);
                         });
        return result;
    } else {
        JsonNumber number = expectJsonNumber(context);
        switch (number.type) {
        case JsonNumber::UNSIGNED_INT:
            return number.uns;
        case JsonNumber::SIGNED_INT:
            return number.sgn;
        case JsonNumber::FLOATING_POINT:
            return number.fp;
        default:
            throw ML::Exception("logic error in expectJson");
        }
    }
}

#endif

} // namespace ML


#endif /* __jml__utils__json_parsing_h__ */

