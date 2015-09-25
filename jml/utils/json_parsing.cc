/* json_parsing.cc
   Jeremy Barnes, 1 February 2012
   Copyright (c) 2012 Datacratic.  All rights reserved.

   Released under the MIT license.
*/

#include "json_parsing.h"
#include "jml/arch/format.h"


using namespace std;


namespace ML {

/*****************************************************************************/
/* JSON UTILITIES                                                            */
/*****************************************************************************/

void skipJsonWhitespace(Parse_Context & context)
{
    // Fast-path for the usual case for not EOF and no whitespace
    if (JML_LIKELY(!context.eof())) {
        char c = *context;
        if (c > ' ') {
            return;
        }
        if (c != ' ' && c != '\t' && c != '\n' && c != '\r')
            return;
    }

    while (!context.eof()
           && (context.match_whitespace() || context.match_eol()));
}

char * jsonEscapeCore(const std::string & str, char * p, char * end)
{
    for (unsigned i = 0;  i < str.size();  ++i) {
        if (p + 4 >= end)
            return 0;

        char c = str[i];
        if (c >= ' ' && c < 127 && c != '\"' && c != '\\')
            *p++ = c;
        else {
            *p++ = '\\';
            switch (c) {
            case '\t': *p++ = ('t');  break;
            case '\n': *p++ = ('n');  break;
            case '\r': *p++ = ('r');  break;
            case '\f': *p++ = ('f');  break;
            case '\b': *p++ = ('b');  break;
            case '/':
            case '\\':
            case '\"': *p++ = (c);  break;
            default:
                throw Exception("Invalid character in JSON string: " + str);
            }
        }
    }

    return p;
}

std::string
jsonEscape(const std::string & str)
{
    size_t sz = str.size() * 4 + 4;
    char buf[sz];
    char * p = buf, * end = buf + sz;

    p = jsonEscapeCore(str, p, end);

    if (!p)
        throw ML::Exception("To fix: logic error in JSON escaping");

    return string(buf, p);
}

void jsonEscape(const std::string & str, std::ostream & stream)
{
    size_t sz = str.size() * 4 + 4;
    char buf[sz];
    char * p = buf, * end = buf + sz;

    p = jsonEscapeCore(str, p, end);

    if (!p)
        throw ML::Exception("To fix: logic error in JSON escaping");

    stream.write(buf, p - buf);
}

bool matchJsonString(Parse_Context & context, std::string & str)
{
    Parse_Context::Revert_Token token(context);

    skipJsonWhitespace(context);
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
        case 'b': result.push_back('\b');  break;
        case '/': result.push_back('/');   break;
        case '\\':result.push_back('\\');  break;
        case '"': result.push_back('"');   break;
        case 'u': {
            int code = context.expect_hex4();
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

std::string expectJsonStringAsciiPermissive(Parse_Context & context, char sub)
{
    skipJsonWhitespace(context);
    context.expect_literal('"');

    char internalBuffer[4096];

    char * buffer = internalBuffer;
    size_t bufferSize = 4096;
    size_t pos = 0;

    // Try multiple times to make it fit
    while (!context.match_literal('"')) {
        int c = *context++;
        if (c == '\\') {
            c = *context++;
            switch (c) {
            case 't': c = '\t';  break;
            case 'n': c = '\n';  break;
            case 'r': c = '\r';  break;
            case 'f': c = '\f';  break;
            case 'b': c = '\b';  break;
            case '/': c = '/';   break;
            case '\\':c = '\\';  break;
            case '"': c = '"';   break;
            case 'u': {
                int code = context.expect_hex4();
                c = code;
                break;
            }
            default:
                context.exception("invalid escaped char");
            }
        }
        if (c < ' ' || c >= 127)
            c = sub;
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

std::string expectJsonString(Parse_Context & context)
{
    skipJsonWhitespace(context);
    context.expect_literal('"');

    char internalBuffer[4096];

    char * buffer = internalBuffer;
    size_t bufferSize = 4096;
    size_t pos = 0;

    auto encode = [&] (int code, unsigned pos, uint32_t mask, uint32_t head) -> char {
        return ((code >> (6 * pos)) & mask) | head;
    };

    // Try multiple times to make it fit
    while (!context.match_literal('"')) {

        int c = *context++;
        if (c == '\\') {
            c = *context++;
            switch (c) {
            case 't': c = '\t';  break;
            case 'n': c = '\n';  break;
            case 'r': c = '\r';  break;
            case 'f': c = '\f';  break;
            case 'b': c = '\b';  break;
            case '/': c = '/';   break;
            case '\\':c = '\\';  break;
            case '"': c = '"';   break;
            case 'u': {
                c = context.expect_hex4();
                break;
            }
            default:
                context.exception("invalid escaped char");
            }
        }

        if ((pos + 6) == bufferSize) {
            size_t newBufferSize = bufferSize * 8;
            char * newBuffer = new char[newBufferSize];

            std::copy(buffer, buffer + bufferSize, newBuffer);

            if (buffer != internalBuffer)
                delete[] buffer;

            buffer = newBuffer;
            bufferSize = newBufferSize;
        }

        if (c <= 0x7f) {
            buffer[pos++] = (char) c;
        }

        else if (c <= 0x7FF) {
            buffer[pos++] = encode(c, 1, 0x1F, 0xC0);
            buffer[pos++] = encode(c, 0, 0x3F, 0x80);
        }

        else if (c <= 0xFFFF) {
            buffer[pos++] = encode(c, 2, 0x0F, 0xE0);
            buffer[pos++] = encode(c, 1, 0x3F, 0x80);
            buffer[pos++] = encode(c, 0, 0x3F, 0x80);
        }

        else if (c <= 0x1FFFFF) {
            buffer[pos++] = encode(c, 3, 0x07, 0xF0);
            buffer[pos++] = encode(c, 2, 0x3F, 0x80);
            buffer[pos++] = encode(c, 1, 0x3F, 0x80);
            buffer[pos++] = encode(c, 0, 0x3F, 0x80);
        }

        else if (c <= 0x3FFFFFFF) {
            buffer[pos++] = encode(c, 4, 0x03, 0xF8);
            buffer[pos++] = encode(c, 3, 0x3F, 0x80);
            buffer[pos++] = encode(c, 2, 0x3F, 0x80);
            buffer[pos++] = encode(c, 1, 0x3F, 0x80);
            buffer[pos++] = encode(c, 0, 0x3F, 0x80);
        }

        else {
            buffer[pos++] = encode(c, 5, 0x01, 0xFC);
            buffer[pos++] = encode(c, 4, 0x3F, 0x80);
            buffer[pos++] = encode(c, 3, 0x3F, 0x80);
            buffer[pos++] = encode(c, 2, 0x3F, 0x80);
            buffer[pos++] = encode(c, 1, 0x3F, 0x80);
            buffer[pos++] = encode(c, 0, 0x3F, 0x80);
        }
    }

    string result(buffer, buffer + pos);

    if (buffer != internalBuffer)
        delete[] buffer;

    return result;
}

ssize_t expectJsonStringAscii(Parse_Context & context, char * buffer, size_t maxLength)
{
    skipJsonWhitespace(context);
    context.expect_literal('"');

    size_t bufferSize = maxLength - 1;
    size_t pos = 0;

    // Try multiple times to make it fit
    while (!context.match_literal('"')) {
        int c = *context++;
        if (c == '\\') {
            c = *context++;
            switch (c) {
            case 't': c = '\t';  break;
            case 'n': c = '\n';  break;
            case 'r': c = '\r';  break;
            case 'f': c = '\f';  break;
            case 'b': c = '\b';  break;
            case '/': c = '/';   break;
            case '\\':c = '\\';  break;
            case '"': c = '"';   break;
            case 'u': {
                int code = context.expect_hex4();
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
        if (c < 0 || c >= 127)
           context.exception("invalid JSON ASCII string character");
        if (pos == bufferSize) {
            return -1;
        }
        buffer[pos++] = c;
    }

    buffer[pos] = 0; // null terminator

    return pos;
}

std::string expectJsonStringAscii(Parse_Context & context)
{
    skipJsonWhitespace(context);
    context.expect_literal('"');

    char internalBuffer[4096];

    char * buffer = internalBuffer;
    size_t bufferSize = 4096;
    size_t pos = 0;

    // Try multiple times to make it fit
    while (!context.match_literal('"')) {

#if 0 // attempt to do it a block at a time
        char * bufferEnd = cbuffer + bufferSize;

        int charsMatched
            = context.match_text(buffer + pos, buffer + bufferSize,
                                 "\"\\");
        pos += charsMatched;
#endif

        int c = *context++;
        if (c == '\\') {
            c = *context++;
            switch (c) {
            case 't': c = '\t';  break;
            case 'n': c = '\n';  break;
            case 'r': c = '\r';  break;
            case 'f': c = '\f';  break;
            case 'b': c = '\b';  break;
            case '/': c = '/';   break;
            case '\\':c = '\\';  break;
            case '"': c = '"';   break;
            case 'u': {
                int code = context.expect_hex4();
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
        if (c < 0 || c >= 127)
           context.exception("invalid JSON ASCII string character");
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

bool
matchJsonNull(Parse_Context & context)
{
    skipJsonWhitespace(context);
    return context.match_literal("null");
}

void
expectJsonArray(Parse_Context & context,
                const std::function<void (int, Parse_Context &)> & onEntry)
{
    skipJsonWhitespace(context);

    if (context.match_literal("null"))
        return;

    context.expect_literal('[');
    skipJsonWhitespace(context);
    if (context.match_literal(']')) return;

    for (int i = 0;  ; ++i) {
        skipJsonWhitespace(context);

        onEntry(i, context);

        skipJsonWhitespace(context);

        if (!context.match_literal(',')) break;
    }

    skipJsonWhitespace(context);
    context.expect_literal(']');
}

void
expectJsonObject(Parse_Context & context,
                 const std::function<void (const std::string &, Parse_Context &)> & onEntry)
{
    skipJsonWhitespace(context);

    if (context.match_literal("null"))
        return;

    context.expect_literal('{');

    skipJsonWhitespace(context);

    if (context.match_literal('}')) return;

    for (;;) {
        skipJsonWhitespace(context);

        string key = expectJsonString(context);

        skipJsonWhitespace(context);

        context.expect_literal(':');

        skipJsonWhitespace(context);

        onEntry(key, context);

        skipJsonWhitespace(context);

        if (!context.match_literal(',')) break;
    }

    skipJsonWhitespace(context);
    context.expect_literal('}');
}

void
expectJsonObjectAscii(Parse_Context & context,
                      const std::function<void (const char *, Parse_Context &)> & onEntry)
{
    skipJsonWhitespace(context);

    if (context.match_literal("null"))
        return;

    context.expect_literal('{');

    skipJsonWhitespace(context);

    if (context.match_literal('}')) return;

    for (;;) {
        skipJsonWhitespace(context);

        char keyBuffer[1024];

        ssize_t done = expectJsonStringAscii(context, keyBuffer, 1024);
        if (done == -1)
            context.exception("JSON key is too long");

        skipJsonWhitespace(context);

        context.expect_literal(':');

        skipJsonWhitespace(context);

        onEntry(keyBuffer, context);

        skipJsonWhitespace(context);

        if (!context.match_literal(',')) break;
    }

    skipJsonWhitespace(context);
    context.expect_literal('}');
}

bool
matchJsonObject(Parse_Context & context,
                const std::function<bool (const std::string &, Parse_Context &)> & onEntry)
{
    skipJsonWhitespace(context);

    if (context.match_literal("null"))
        return true;

    if (!context.match_literal('{')) return false;
    skipJsonWhitespace(context);
    if (context.match_literal('}')) return true;

    for (;;) {
        skipJsonWhitespace(context);

        string key = expectJsonString(context);

        skipJsonWhitespace(context);
        if (!context.match_literal(':')) return false;
        skipJsonWhitespace(context);

        if (!onEntry(key, context)) return false;

        skipJsonWhitespace(context);

        if (!context.match_literal(',')) break;
    }

    skipJsonWhitespace(context);
    if (!context.match_literal('}')) return false;

    return true;
}

JsonNumber expectJsonNumber(Parse_Context & context)
{
    JsonNumber result;

    std::string number;
    number.reserve(32);

    bool negative = false;
    bool doublePrecision = false;

    if (context.match_literal('-')) {
        number += '-';
        negative = true;
    }

    // EXTENSION: accept NaN and positive or negative infinity
    if (context.match_literal('N')) {
        context.expect_literal("aN");
        result.fp = negative ? -NAN : NAN;
        result.type = JsonNumber::FLOATING_POINT;
        return result;
    }
    else if (context.match_literal('n')) {
        context.expect_literal("an");
        result.fp = negative ? -NAN : NAN;
        result.type = JsonNumber::FLOATING_POINT;
        return result;
    }
    else if (context.match_literal('I') || context.match_literal('i')) {
        context.expect_literal("nf");
        result.fp = negative ? -INFINITY : INFINITY;
        result.type = JsonNumber::FLOATING_POINT;
        return result;
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

    try {
        JML_TRACE_EXCEPTIONS(false);
        if (number.empty())
            context.exception("expected number");

        if (doublePrecision) {
            char * endptr = 0;
            errno = 0;
            result.fp = strtod(number.c_str(), &endptr);
            if (errno || endptr != number.c_str() + number.length())
                context.exception(ML::format("failed to convert '%s' to long long",
                                             number.c_str()));
            result.type = JsonNumber::FLOATING_POINT;
        } else if (negative) {
            char * endptr = 0;
            errno = 0;
            result.sgn = strtol(number.c_str(), &endptr, 10);
            if (errno || endptr != number.c_str() + number.length())
                context.exception(ML::format("failed to convert '%s' to long long",
                                             number.c_str()));
            result.type = JsonNumber::SIGNED_INT;
        } else {
            char * endptr = 0;
            errno = 0;
            result.uns = strtoull(number.c_str(), &endptr, 10);
            if (errno || endptr != number.c_str() + number.length())
                context.exception(ML::format("failed to convert '%s' to unsigned long long",
                                             number.c_str()));
            result.type = JsonNumber::UNSIGNED_INT;
        }
    } catch (const std::exception & exc) {
        context.exception("expected number");
    }

    return result;
}

} // namespace ML
