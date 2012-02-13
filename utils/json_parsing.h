/* json_parsing.h                                                  -*- C++ -*-
   Jeremy Barnes, 1 February 2012
   Copyright (c) 2012 Recoset Inc.  All rights reserved.

   Released under the MIT license.

   Functionality to ease parsing of JSON from within a parse_function.
*/

#ifndef __jml__utils__json_parsing_h__
#define __jml__utils__json_parsing_h__

#include <boost/function.hpp>
#include <string>
#include "parse_context.h"

namespace ML {

/*****************************************************************************/
/* JSON UTILITIES                                                            */
/*****************************************************************************/

std::string jsonEscape(const std::string & str);

std::string expectJsonString(Parse_Context & context);
bool matchJsonString(Parse_Context & context, std::string & str);

void
expectJsonArray(Parse_Context & context,
                boost::function<void (int, Parse_Context &)> onEntry);

void
expectJsonObject(Parse_Context & context,
                 boost::function<void (std::string, Parse_Context &)> onEntry);

bool
matchJsonObject(Parse_Context & context,
                boost::function<bool (std::string, Parse_Context &)> onEntry);


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
                         [&] (std::string key, Parse_Context & context)
                         {
                             result[key] = expectJson(context);
                         });
        return result;
    } else return context.expect_double();
}

#endif

} // namespace ML


#endif /* __jml__utils__json_parsing_h__ */

