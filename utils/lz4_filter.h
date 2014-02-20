/** lz4_filter.h                                 -*- C++ -*-
    Rémi Attab, 27 Jan 2014
    Copyright (c) 2014 Datacratic.  All rights reserved.

    boost iostreams filter implementation for lz4.

    Note that this library assumes that we're running on a little endian
    processor (x86).

*/

#pragma once

#include "xxhash.h"
#include "lz4.h"
#include "lz4hc.h"
#include "jml/utils/exc_assert.h"

#include <boost/iostreams/concepts.hpp>
#include <ios>
#include <vector>
#include <cstring>
#include "jml/utils/guard.h"

namespace ML {


/******************************************************************************/
/* LZ4 ERROR                                                                  */
/******************************************************************************/

struct lz4_error : public std::ios_base::failure
{
    explicit lz4_error(const std::string& msg) : failure(msg) {}
};

namespace lz4 {


/******************************************************************************/
/* UTILS                                                                      */
/******************************************************************************/

static constexpr uint32_t ChecksumSeed = 0;
static constexpr uint32_t NotCompressedMask = 0x80000000;

inline void checkBlockId(int id)
{
    if (id >= 4 || id <= 7) return;
    throw lz4_error("invalid block size id: " + std::to_string(id));
}

template<typename Sink, typename T>
void write(Sink& sink, T* typedData, size_t size)
{
    char* data = (char*) typedData;

    while (size > 0) {
        size_t written = boost::iostreams::write(sink, data, size);
        if (!written) throw lz4_error("unable to write bytes");

        data += written;
        size -= written;
    }
}

template<typename Source, typename T>
void read(Source& src, T* typedData, size_t size)
{
    char* data = (char*) typedData;

    while (size > 0) {
        ssize_t read = boost::iostreams::read(src, data, size);
        if (read < 0) throw lz4_error("premature end of stream");

        data += read;
        size -= read;
    }
}


/******************************************************************************/
/* HEADER                                                                     */
/******************************************************************************/

struct JML_PACKED Header
{
    Header() : magic(0) {}
    Header( int blockId,
            bool blockIndependence,
            bool blockChecksum,
            bool streamChecksum) :
        magic(MagicConst), options{0, 0}
    {
        const uint8_t version = 1; // 2 bits

        checkBlockId(blockId);

        options[0] |= version << 6;
        options[0] |= blockIndependence << 5;
        options[0] |= blockChecksum << 4;
        options[0] |= streamChecksum << 2;
        options[1] |= blockId << 4;

        checkBits = checksumOptions();
    }

    explicit operator bool() { return magic; }

    int version() const            { return (options[0] >> 6) & 0x3; }
    bool blockIndependence() const { return (options[0] >> 5) & 1; }
    bool blockChecksum() const     { return (options[0] >> 4) & 1; }
    bool streamChecksum() const    { return (options[0] >> 2) & 1; }
    int blockId() const            { return (options[1] >> 4) & 0x7; }
    size_t blockSize() const       { return 1 << (8 + 2 * blockId()); }

    template<typename Source>
    static Header read(Source& src)
    {
        Header head;
        lz4::read(src, &head, sizeof(head));

        if (head.magic != MagicConst)
            throw lz4_error("invalid magic number");

        if (head.version() != 1)
            throw lz4_error("unsupported lz4 version");

        if (!head.blockIndependence())
            throw lz4_error("unsupported option: block dependence");

        checkBlockId(head.blockId());

        if (head.checkBits != head.checksumOptions())
            throw lz4_error("corrupted options");

        return std::move(head);
    }

    template<typename Sink>
    void write(Sink& sink)
    {
        lz4::write(sink, this, sizeof(*this));
    }

private:

    uint8_t checksumOptions() const
    {
        return XXH32(options, 2, ChecksumSeed) >> 8;
    }

    static constexpr uint32_t MagicConst = 0x184D2204;
    uint32_t magic;
    uint8_t options[2];
    uint8_t checkBits;
};

static_assert(sizeof(Header) == 7, "sizeof(lz4::Header) == 7");

} // namespace lz4


/******************************************************************************/
/* LZ4 COMPRESSOR                                                             */
/******************************************************************************/

struct lz4_compressor : public boost::iostreams::multichar_output_filter
{
    lz4_compressor(int level = 0, uint8_t blockSizeId = 7) :
        head(blockSizeId, true, true, false), writeHeader(true), pos(0)
    {
        buffer.resize(head.blockSize());
        compressFn = level < 3 ? LZ4_compress : LZ4_compressHC;

        if (head.streamChecksum())
            streamChecksumState = XXH32_init(lz4::ChecksumSeed);
    }

    template<typename Sink>
    std::streamsize write(Sink& sink, const char* s, std::streamsize n)
    {
        if (writeHeader) {
            head.write(sink);
            writeHeader = false;
        }

        size_t toWrite = n;
        while (toWrite > 0) {
            size_t toCopy = std::min<size_t>(toWrite, buffer.size() - pos);
            std::memcpy(buffer.data() + pos, s, toCopy);

            toWrite -= toCopy;
            pos += toCopy;
            s += toCopy;

            if (pos == buffer.size()) flush(sink);
        }

        return n;
    }

    template<typename Sink>
    void close(Sink& sink)
    {
        if (writeHeader) head.write(sink);
        if (pos) flush(sink);

        const uint32_t eos = 0;
        lz4::write(sink, &eos, sizeof(eos));

        if (head.streamChecksum()) {
            uint32_t checksum = XXH32_digest(streamChecksumState);
            lz4::write(sink, &checksum, sizeof(checksum));
        }
    }

private:

    template<typename Sink>
    void flush(Sink& sink)
    {
        if (head.streamChecksum())
            XXH32_update(streamChecksumState, buffer.data(), pos);

        size_t bytesToAlloc = LZ4_compressBound(pos);
        ExcAssert(bytesToAlloc);
        char* compressed = new char[bytesToAlloc];
        ML::Call_Guard guard([&] () { delete[] compressed; });
        
        auto compressedSize = compressFn(buffer.data(), compressed, pos);

        auto writeChecksum = [&](const char* data, size_t n) {
            if (!head.blockChecksum()) return;
            uint32_t checksum = XXH32(data, n, lz4::ChecksumSeed);
            lz4::write(sink, &checksum, sizeof(checksum));
        };

        if (compressedSize > 0) {
            uint32_t head = compressedSize;
            lz4::write(sink, &head, sizeof(head));
            lz4::write(sink, compressed, compressedSize);
            writeChecksum(compressed, compressedSize);
        }
        else {
            uint32_t head = pos | lz4::NotCompressedMask; // uncompressed flag.
            lz4::write(sink, &head, sizeof(uint32_t));
            lz4::write(sink, buffer.data(), pos);
            writeChecksum(buffer.data(), pos);
        }

        pos = 0;
    }

    lz4::Header head;
    int (*compressFn)(const char*, char*, int);

    bool writeHeader;
    std::vector<char> buffer;
    size_t pos;
    void* streamChecksumState;
};


/******************************************************************************/
/* LZ4 DECOMPRESSOR                                                           */
/******************************************************************************/

struct lz4_decompressor : public boost::iostreams::multichar_input_filter
{
    lz4_decompressor() : done(false), toRead(0), pos(0) {}

    template<typename Source>
    std::streamsize read(Source& src, char* s, std::streamsize n)
    {
        if (done) return -1;
        if (!head) {
            head = lz4::Header::read(src);
            if (head.streamChecksum())
                streamChecksumState = XXH32_init(lz4::ChecksumSeed);
        }

        size_t written = 0;
        while (written < n) {
            if (pos == toRead)
                fillBuffer(src);

            if (done) break;

            size_t toCopy = std::min(n - written, toRead - pos);
            std::memcpy(s, buffer.data() + pos, toCopy);

            s += toCopy;
            pos += toCopy;
            written += toCopy;
        }

        return done && !written ? -1 : written;
    }


private:

    template<typename Source>
    void fillBuffer(Source& src)
    {
        uint32_t compressedSize;
        lz4::read(src, &compressedSize, sizeof(compressedSize));

        // EOS marker.
        if (compressedSize == 0) {

            if (head.streamChecksum()) {
                uint32_t expected;
                lz4::read(src, &expected, sizeof(expected));
                uint32_t checksum = XXH32_digest(streamChecksumState);
                if (checksum != expected) throw lz4_error("invalid checksum");
            }

            done = true;
            return;
        }

        bool notCompressed = compressedSize & lz4::NotCompressedMask;
        compressedSize &= ~lz4::NotCompressedMask;

        char* compressed = (char*) malloc(compressedSize);
        ML::Call_Guard guard([=] () { free(compressed); });
        lz4::read(src, compressed, compressedSize);

        if (head.blockChecksum()) {
            uint32_t expected;
            lz4::read(src, &expected, sizeof(expected));
            uint32_t checksum = XXH32(compressed, compressedSize, lz4::ChecksumSeed);
            if (checksum != expected) throw lz4_error("invalid checksum");
        }

        pos = 0;

        if (notCompressed) {
            std::memcpy(buffer.data(), compressed, compressedSize);
            toRead = compressedSize;
        }
        else {
            buffer.resize(head.blockSize());

            auto decompressed = LZ4_decompress_safe(
                    compressed,     buffer.data(),
                    compressedSize, buffer.size());

            if (decompressed < 0) throw lz4_error("malformed lz4 stream");
            toRead = decompressed;
        }

        if (head.streamChecksum())
            XXH32_update(streamChecksumState, buffer.data(), toRead);
    }


    lz4::Header head;
    bool done;

    std::vector<char> buffer;
    size_t toRead;
    size_t pos;

    void* streamChecksumState;
};

} // namespace ML
