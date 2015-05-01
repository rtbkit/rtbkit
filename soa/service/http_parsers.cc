/* http_parsers.h                                                  -*- C++ -*-
   Wolfgang Sourdeau, January 2014
   Copyright (c) 2014 Datacratic.  All rights reserved.

*/

#include <string.h>

#include <iostream>
#include "jml/arch/exception.h"
#include "jml/utils/string_functions.h"

#include "http_parsers.h"

using namespace std;
using namespace Datacratic;


/****************************************************************************/
/* HTTP RESPONSE PARSER                                                     */
/****************************************************************************/

void
HttpResponseParser::
clear()
    noexcept
{
    expectBody_ = true;
    stage_ = 0;
    buffer_.clear();
    remainingBody_ = 0;
    useChunkedEncoding_ = false;
    requireClose_ = false;
}

void
HttpResponseParser::
feed(const char * bufferData)
{
    // cerr << "feed: /" + ML::hexify_string(string(bufferData)) + "/\n";
    feed(bufferData, strlen(bufferData));
}

HttpResponseParser::BufferState
HttpResponseParser::
prepareParsing(const char * bufferData, size_t bufferSize)
{
    BufferState state;

    if (buffer_.size() > 0) {
        buffer_.append(bufferData, bufferSize);
        state.data = buffer_.c_str();
        state.dataSize = buffer_.size();
        state.fromBuffer = true;
    }
    else {
        state.data = bufferData;
        state.dataSize = bufferSize;
        state.fromBuffer = false;
    }

    return state;
}

bool
HttpResponseParser::
BufferState::
skipToChar(char c, bool throwOnEol)
{
    while (ptr < dataSize) {
        if (data[ptr] == c) {
            return true;
        }
        else if (throwOnEol
                 && (data[ptr] == '\r' || data[ptr] == '\n')) {
            throw ML::Exception("unexpected end of line");
        }
        ptr++;
    }

    return false;
}

void
HttpResponseParser::
feed(const char * bufferData, size_t bufferSize)
{
    // std::cerr << ("data: /"
    //          + ML::hexify_string(string(bufferData, bufferSize))
    //          + "/\n");
    BufferState state = prepareParsing(bufferData, bufferSize);

    // cerr << ("state: " + to_string(stage_)
    //          + "; dataSize: " + to_string(dataSize) + "\n");

    /* We loop as long as there are bytes available for parsing and as long as
       the parsing stages change. */
    bool stageDone(true);
    while (stageDone && state.remaining() > 0) {
        if (stage_ == 0) {
            stageDone = parseStatusLine(state);
            if (stageDone) {
                stage_ = 1;
            }
        }
        else if (stage_ == 1) {
            stageDone = parseHeaders(state);
            if (stageDone) {
                if (!expectBody_
                    || (remainingBody_ == 0 && !useChunkedEncoding_)) {
                    finalizeParsing();
                    stage_ = 0;
                }
                else {
                    stage_ = 2;
                }
            }
        }
        else if (stage_ == 2) {
            stageDone = parseBody(state);
            if (stageDone) {
                finalizeParsing();
                stage_ = 0;
            }
        }
    }

    size_t remaining = state.remainingUncommited();
    if (remaining > 0) {
        if (state.commited > 0 || !state.fromBuffer) {
            buffer_.assign(state.data + state.commited, remaining);
        }
    }
    else if (state.fromBuffer) {
        buffer_.clear();
    }
}

bool
HttpResponseParser::
parseStatusLine(BufferState & state)
{
    /* status line parsing */

    /* sizeof("HTTP/X.X XXX ") */
    if (state.remaining() < 16) {
        return false;
    }

    if (::memcmp(state.currentDataPtr(), "HTTP/", 5) != 0) {
        throw ML::Exception("version must start with 'HTTP/'");
    }
    state.ptr += 5;

    if (!state.skipToChar(' ', true)) {
        /* post-version ' ' not found even though size is sufficient */
        throw ML::Exception("version too long");
    }
    size_t versionEnd = state.ptr;

    state.ptr++;
    size_t codeStart = state.ptr;
    if (!state.skipToChar(' ', true)) {
        /* post-code ' ' not found even though size is sufficient */
        throw ML::Exception("code too long");
    }

    size_t codeEnd = state.ptr;
    int code = ML::antoi(state.data + codeStart, state.data + codeEnd);

    /* we skip the whole "reason" string */
    if (!state.skipToChar('\r', false)) {
        return false;
    }
    state.ptr++;
    if (state.remaining() == 0) {
        return false;
    }
    if (state.data[state.ptr] != '\n') {
        throw ML::Exception("expected \\n");
    }
    state.ptr++;
    state.commit();

    if (onResponseStart) {
        onResponseStart(string(state.data, versionEnd), code);
    }

    return true;
}

bool
HttpResponseParser::
parseHeaders(BufferState & state)
{
    string multiline;
    unsigned int numLines(0);

    /* header line parsing */
    while (state.data[state.ptr] != '\r' || numLines > 0) {
        size_t linePtr = state.ptr;
        if (numLines == 0) {
            if (!state.skipToChar(':', true)) {
                return false;
            }
        }
        if (!state.skipToChar('\r', false)) {
            return false;
        }
        if (state.remaining() < 3) {
            return false;
        }
        state.ptr++;
        if (state.data[state.ptr] != '\n') {
            throw ML::Exception("expected \\n");
        }
        state.ptr++;

        /* does the next line starts with a space or a tab? */
        if (state.data[state.ptr] == ' ' || state.data[state.ptr] == '\t') {
            multiline.append(state.data + linePtr, state.ptr - linePtr - 2);
            numLines++;
            state.ptr++;
        }
        else {
            if (numLines == 0) {
                handleHeader(state.data + linePtr, state.ptr - linePtr);
                state.commit();
            }
            else {
                multiline.append(state.data + linePtr,
                                 state.ptr - linePtr);
                handleHeader(multiline.c_str(), multiline.size());
                multiline.clear();
                state.commit();
                numLines = 0;
            }
        }
    }
    if (state.ptr + 1 == state.dataSize) {
        return false;
    }
    state.ptr++;
    if (state.data[state.ptr] != '\n') {
        throw ML::Exception("expected \\n");
    }
    state.ptr++;
    state.commit();

    if (onHeader) {
        onHeader("\r\n", 2);
    }

    return true;
}

void
HttpResponseParser::
handleHeader(const char * data, size_t dataSize)
{
    size_t ptr(0);

    auto skipToChar = [&] (char c) {
        while (ptr < dataSize) {
            if (data[ptr] == c)
                return true;
            ptr++;
        }

        return false;
    };
    auto skipChar = [&] (char c) {
        while (ptr < dataSize && data[ptr] == c) {
            ptr++;
        }
    };
    auto matchString = [&] (const char * testString, size_t len) {
        bool result;
        if (dataSize >= (ptr + len)
            && ::strncasecmp(data + ptr, testString, len) == 0) {
            ptr += len;
            result = true;
        }
        else {
            result = false;
        }
        return result;
    };

    auto skipToValue = [&] () {
        skipChar(' ');
        skipToChar(':');
        ptr++;
        skipChar(' ');
    };

    if (matchString("Connection", 10)) {
        skipToValue();
        if (matchString("close", 5)) {
            requireClose_ = true;
        }
    }
    else if (matchString("Content-Length", 14)) {
        skipToValue();
        remainingBody_ = ML::antoi(data + ptr, data + dataSize - 2);
    }
    else if (matchString("Transfer-Encoding", 15)) {
        skipToValue();
        if (matchString("chunked", 7)) {
            useChunkedEncoding_ = true;
        }
    }

    if (onHeader) {
        onHeader(data, dataSize);
    }
}

bool
HttpResponseParser::
parseBody(BufferState & state)
{
    return (useChunkedEncoding_
            ? parseChunkedBody(state)
            : parseBlockBody(state));
}

bool
HttpResponseParser::
parseChunkedBody(BufferState & state)
{
    int chunkSize(-1);

    /* we loop as long as there are chunks to process */
    while (chunkSize != 0) {
        const char * sizeStart = state.data + state.ptr;
        if (!state.skipToChar('\r', false)) {
            return false;
        }

        if (state.remaining() < 2) {
            return false;
        }

        const char * sizeEnd = state.data + state.ptr;

        /* look for ';' and adjust sizeEnd in consequence */
        for (const char * ptr = sizeStart; ptr < sizeEnd; ptr++) {
            if (*ptr == ';') {
                sizeEnd = ptr;
                break;
            }
        }

        chunkSize = ML::antoi(sizeStart, sizeEnd, 16);

        state.ptr += 2;
        if (state.remaining() < chunkSize + 2) {
            return false;
        }

        if (onData && chunkSize > 0) {
            onData(state.currentDataPtr(), chunkSize);
        }
        state.ptr += chunkSize + 2;
        state.commit();
    }

    return true;
}

bool
HttpResponseParser::
parseBlockBody(BufferState & state)
{
    uint64_t chunkSize = min(state.remaining(), remainingBody_);
    // cerr << "toSend: " + to_string(chunkSize) + "\n";
    // cerr << "received body: /" + string(data, chunkSize) + "/\n";
    if (onData && chunkSize > 0) {
        onData(state.currentDataPtr(), chunkSize);
    }
    state.ptr += chunkSize;
    remainingBody_ -= chunkSize;
    state.commit();

    return (remainingBody_ == 0);
}

void
HttpResponseParser::
finalizeParsing()
{
    if (onDone) {
        onDone(requireClose_);
    }
    clear();
}
