/** lz4_filter.h                                 -*- C++ -*-
    Rémi Attab, 27 Jan 2014
    Copyright (c) 2014 Datacratic.  All rights reserved.

    boost iostreams filter implementation for lz4.

*/

#pragma once

#include "lz4.h"
#include "lz4hc.h"

#include <boost/iostreams/concepts.hpp>
#include <ios>
#include <vector>
#include <cstring>
#include <alloca.h>

namespace ML {


/******************************************************************************/
/* BLOCK HEADER                                                               */
/******************************************************************************/

namespace details {

struct Lz4Header
{
    size_t rawSize;
    size_t compressedSize;

    explicit Lz4Header(size_t raw = 0, size_t compressed = 0) :
        rawSize(raw), compressedSize(compressed)
    {}

    explicit operator bool() const { return !rawSize && !compressedSize; }
};

} // namespace details


/******************************************************************************/
/* LZ4 ERROR                                                                  */
/******************************************************************************/

struct lz4_error : public std::ios_base::failure
{
    explicit lz4_error(const std::string& msg) : failure(msg) {}
};


/******************************************************************************/
/* LZ4 COMPRESSOR                                                             */
/******************************************************************************/

struct lz4_compressor : public boost::iostreams::multichar_output_filter
{
    enum Level { Normal, High };

    lz4_compressor(int level = -1, size_t blockSize = 1024 * 1024) :
        level(level < 2 ? Normal : High), pos(0)
    {
        buffer.resize(blockSize);
    }

    template<typename Sink>
    std::streamsize write(Sink& sink, const char* s, std::streamsize n)
    {
        size_t toWrite = n;
        while (toWrite > 0) {
            size_t toCopy = std::min<size_t>(n, buffer.size() - pos);
            std::memcpy(buffer.data() + pos, s, toCopy);

            if (pos + toCopy == buffer.size()) flush(sink);

            toWrite -= toCopy;
            pos += toCopy;
            s += toCopy;
        }

        return n;
    }

    template<typename Sink>
    void close(Sink& sink)
    {
        if (pos) flush(sink);
    }

private:

    template<typename Sink>
    void flush(Sink& sink)
    {
        details::Lz4Header head(pos);
        char* compressed = (char*) alloca(LZ4_compressBound(pos));

        if (level == Normal)
            head.compressedSize = LZ4_compress(buffer.data(), compressed, pos);
        else
            head.compressedSize = LZ4_compressHC(buffer.data(), compressed, pos);

        if (!head.compressedSize)
            throw lz4_error("lz4 compression failed");

        boost::iostreams::write(sink, (char*) &head, sizeof(head));
        boost::iostreams::write(sink, compressed, head.compressedSize);
        pos = 0;
    }

    const Level level;
    std::vector<char> buffer;
    size_t pos;
};


/******************************************************************************/
/* LZ4 DECOMPRESSOR                                                           */
/******************************************************************************/

struct lz4_decompressor : public boost::iostreams::multichar_input_filter
{
    lz4_decompressor() : done(false), rawPos(0) {}

    template<typename Source>
    std::streamsize read(Source& src, char* s, std::streamsize n)
    {
        if (done) return -1;

        size_t written = 0;
        while (written < n) {

            if (rawPos == rawBuffer.size())
                fillBuffer(src);

            if (done) break;

            size_t toCopy = std::min(n - written, rawBuffer.size() - rawPos);
            std::memcpy(s, rawBuffer.data() + rawPos, toCopy);

            s += toCopy;
            rawPos += toCopy;
            written += toCopy;
        }

        return done && !written ? -1 : written;
    }

private:


    template<typename Source>
    void fillBuffer(Source& src)
    {
        namespace bio = boost::iostreams;

        details::Lz4Header head;
        if (bio::read(src, (char*) &head, sizeof(head)) < 0) {
            done = true;
            return;
        }

        compressedPos = 0;
        compressedBuffer.resize(head.compressedSize);

        while (compressedPos < head.compressedSize) {
            char* start = compressedBuffer.data() + compressedPos;
            size_t leftover = compressedBuffer.size() - compressedPos;

            auto bytes = bio::read(src, start, leftover);
            if (bytes < 0) throw lz4_error("premature end of lz4 stream");
            compressedPos += bytes;
        }


        rawPos = 0;
        rawBuffer.resize(head.rawSize);

        int decompressed = LZ4_decompress_safe(
                compressedBuffer.data(), rawBuffer.data(),
                compressedBuffer.size(), rawBuffer.size());

        if (decompressed < 0) throw lz4_error("malformed lz4 stream");
        if (decompressed < compressedBuffer.size())
            throw lz4_error("logic error in lz4 decoding");
    }


    bool done;

    std::vector<char> compressedBuffer;
    size_t compressedPos;

    std::vector<char> rawBuffer;
    size_t rawPos;
};

} // namespace ML
