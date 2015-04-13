/* http_parsers.h                                                  -*- C++ -*-
   Wolfgang Sourdeau, January 2014
   Copyright (c) 2014 Datacratic.  All rights reserved.

*/

#pragma once

#include <functional>


namespace Datacratic {

/****************************************************************************/
/* HTTP RESPONSE PARSER                                                     */
/****************************************************************************/

/* HttpResponseParser offers a very fast and memory efficient HTTP/1.1
 * response parser. It provides a callback-based interface which enables
 * on-the-fly response processing.
 */

struct HttpResponseParser {
    /* Type of callback used when a response is starting, passing the HTTP
     * version in use as well as the HTTP response code as parameters */
    typedef std::function<void (const std::string &, int)> OnResponseStart;

    /* Type of callback used when to report a header-line, including the
     * header key and the value. */
    typedef std::function<void (const char *, size_t)> OnHeader;

    /* Type of callback used when to report a chunk of the response body. Only
       invoked when the body is larger than 0 byte. */
    typedef std::function<void (const char *, size_t)> OnData;

    /* Type of callback used when to report the end of a response */
    typedef std::function<void (bool)> OnDone;

    HttpResponseParser()
        noexcept
    {
        clear();
    }

    /* Indicates whether to expect a body during the parsing of the next
       response. */
    void setExpectBody(bool expBody)
    { expectBody_ = expBody; }

    /* Feed the parsing with a 0-ended data chunk. Slightly slower than the
       explicitly sized version, but useful for testing. Avoid in production
       code. */
    void feed(const char * data);

    /* Feed the parsing with a data chunk of a specied size. */
    void feed(const char * data, size_t size);

    /* Returns the number of bytes remaining to parse from the body response,
     * as specified by the "Content-Length" header. */
    uint64_t remainingBody() const
    {
        return remainingBody_;
    }

    OnResponseStart onResponseStart;
    OnHeader onHeader;
    OnData onData;
    OnDone onDone;

private:
    void clear() noexcept;

    /* structure to hold the temporary state of the parser used when "feed" is
       invoked */
    struct BufferState {
        BufferState()
            : data(nullptr), dataSize(0), fromBuffer(false),
              ptr(0), commited(0)
        {
        }

        /* skip as many characters as possible until character "c" is found */
        bool skipToChar(char c, bool throwOnEol);

        /* number of bytes available for parsing in the buffer */
        size_t remaining() const { return dataSize - ptr; }

        /* number of uncommited bytes available for parsing in the buffer */
        size_t remainingUncommited() const { return dataSize - commited; }

        /* pointer to the current byte ptr */
        const char * currentDataPtr() const { return data + ptr; }

        /* commit the value of ptr so that the next parsing iteration can
         * start from there */
        void commit()
        {
            commited = ptr;
        }

        const char * data;
        size_t dataSize;
        bool fromBuffer;
        size_t ptr;
        size_t commited;
    };

    BufferState prepareParsing(const char * bufferData, size_t bufferSize);
    bool parseStatusLine(BufferState & state);
    bool parseHeaders(BufferState & state);
    bool parseBody(BufferState & state);
    bool parseChunkedBody(BufferState & state);
    bool parseBlockBody(BufferState & state);

    void handleHeader(const char * data, size_t dataSize);
    void finalizeParsing();

    bool expectBody_;

    int stage_;
    std::string buffer_;

    uint64_t remainingBody_;
    bool useChunkedEncoding_;
    bool requireClose_;
};

}
