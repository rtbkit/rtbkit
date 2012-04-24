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

bool matchJsonString(Parse_Context & context, std::string & str)
{
    Parse_Context::Revert_Token token(context);

    context.skip_whitespace();
    if (!context.match_literal('"')) return false;

    std::string result;

    while (!context.match_literal('"')) {
        if (context.eof()) return false;
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
                return false;
            }
            result.push_back(code);
            break;
        }
        default:
            return false;
        }
    }

    token.ignore();
    str = result;
    return true;
}

std::string expectJsonString(Parse_Context & context)
{
    context.skip_whitespace();
    context.expect_literal('"');

    char internalBuffer[4096];

    char * buffer = internalBuffer;
    size_t bufferSize = 4096;
    size_t pos = 0;

    // Try multiple times to make it fit
    while (!context.match_literal('"')) {
        int c = *context++;
        //if (c < 0 || c >= 127)
        //    context.exception("invalid JSON string character");
        if (c == '\\') {
            c = *context++;
            switch (c) {
            case 't': c = '\t';  break;
            case 'n': c = '\n';  break;
            case 'r': c = '\r';  break;
            case 'f': c = '\f';  break;
            case '/': c = '/';   break;
            case '\\':c = '\\';  break;
            case '"': c = '"';   break;
            case 'u': {
                int code = context.expect_int();
                if (code<0 || code>255) {
                    context.exception(format("non 8bit char %d", code));
                }
                c = code;
                break;
            }
            default:
                context.exception("invalid escaped char");
            }
        }
        if (pos == bufferSize) {
            size_t newBufferSize = bufferSize * 8;
            char * newBuffer = new char[newBufferSize];
            std::copy(buffer, buffer + bufferSize, newBuffer);
            if (buffer != internalBuffer)
                delete[] buffer;
            buffer = newBuffer;
            bufferSize = newBufferSize;
        }
        buffer[pos++] = c;
    }

    string result(buffer, buffer + pos);
    if (buffer != internalBuffer)
        delete[] buffer;
    
    return result;
}

void
expectJsonArray(Parse_Context & context,
                std::function<void (int, Parse_Context &)> onEntry)
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
                 std::function<void (std::string, Parse_Context &)> onEntry)
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

bool
matchJsonObject(Parse_Context & context,
                std::function<bool (std::string, Parse_Context &)> onEntry)
{
    context.skip_whitespace();

    if (context.match_literal("null"))
        return true;

    if (!context.match_literal('{')) return false;
    context.skip_whitespace();
    if (context.match_literal('}')) return true;

    for (;;) {
        context.skip_whitespace();

        string key = expectJsonString(context);

        context.skip_whitespace();
        if (!context.match_literal(':')) return false;
        context.skip_whitespace();

        if (!onEntry(key, context)) return false;

        context.skip_whitespace();

        if (!context.match_literal(',')) break;
    }

    context.skip_whitespace();
    if (!context.match_literal('}')) return false;

    return true;
}

} // namespace ML
