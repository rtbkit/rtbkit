/* json_parsing.cc
   Jeremy Barnes, 1 February 2012
   Copyright (c) 2012 Recoset.  All rights reserved.

   Released under the MIT license.
*/

#include "json_parsing.h"
#include "jml/arch/format.h"


using namespace std;


namespace ML {

/*****************************************************************************/
/* JSON UTILITIES                                                            */
/*****************************************************************************/

std::string
jsonEscape(const std::string & str)
{
    std::string result;
    result.reserve(str.size() * 2);

    for (unsigned i = 0;  i < str.size();  ++i) {
        char c = str[i];
        if (c >= ' ' && c < 127 && c != '\"' && c != '\\')
            result.push_back(c);
        else {
            result.push_back('\\');
            switch (c) {
            case '\t': result.push_back('t');  break;
            case '\n': result.push_back('n');  break;
            case '\r': result.push_back('r');  break;
            case '\b': result.push_back('b');  break;
            case '\f': result.push_back('f');  break;
            case '/':
            case '\\':
            case '\"': result.push_back(c);  break;
            default:
                throw Exception("invalid character in Json string");
            }
        }
    }

    return result;
}

std::string
expectJsonString(Parse_Context & context)
{
    context.skip_whitespace();
    context.expect_literal('"');

    std::string result;

    while (!context.match_literal('"')) {
        int c = *context++;
        //if (c < 0 || c >= 127)
        //    context.exception("invalid JSON string character");
        if (c != '\\') {
            result.push_back(c);
            continue;
        }
        c = *context++;
        switch (c) {
        case 't': result.push_back('\t');  break;
        case 'n': result.push_back('\n');  break;
        case 'r': result.push_back('\r');  break;
        case 'f': result.push_back('\f');  break;
        case '/': result.push_back('/');   break;
        case '\\':result.push_back('\\');  break;
        case '"': result.push_back('"');   break;
        case 'u': {
            int code = context.expect_int();
            if (code<0 || code>255)
            {
                context.exception(format("non 8bit char %d", code));
            }
            result.push_back(code);
            break;
        }
        default:
            context.exception("invalid escaped char");
        }
    }

    return result;
}

void
expectJsonArray(Parse_Context & context,
                boost::function<void (int, Parse_Context &)> onEntry)
{
    context.skip_whitespace();

    if (context.match_literal("null"))
        return;

    context.expect_literal('[');
    context.skip_whitespace();
    if (context.match_literal(']')) return;

    for (int i = 0;  ; ++i) {
        context.skip_whitespace();

        onEntry(i, context);

        context.skip_whitespace();

        if (!context.match_literal(',')) break;
    }

    context.skip_whitespace();
    context.expect_literal(']');
}

void
expectJsonObject(Parse_Context & context,
                 boost::function<void (std::string, Parse_Context &)> onEntry)
{
    context.skip_whitespace();

    if (context.match_literal("null"))
        return;

    context.expect_literal('{');
    context.skip_whitespace();
    if (context.match_literal('}')) return;

    for (;;) {
        context.skip_whitespace();

        string key = expectJsonString(context);

        context.skip_whitespace();
        context.expect_literal(':');
        context.skip_whitespace();

        onEntry(key, context);

        context.skip_whitespace();

        if (!context.match_literal(',')) break;
    }

    context.skip_whitespace();
    context.expect_literal('}');
}

} // namespace ML
