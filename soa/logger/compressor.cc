/* compressor.cc
   Jeremy Barnes, 19 September 2012
   Copyright (c) 2012 Datacratic Inc.  All rights reserved.

   Implementation of compressor abstraction.
*/

#include "compressor.h"
#include "jml/utils/exc_assert.h"
#include <zlib.h>
#include <iostream>

using namespace std;

namespace Datacratic {


/*****************************************************************************/
/* COMPRESSOR                                                                */
/*****************************************************************************/

Compressor::
~Compressor()
{
}

namespace {

bool ends_with(const std::string & str, const std::string & what)
{
    string::size_type result = str.rfind(what);
    return result != string::npos
        && result == str.size() - what.size();
}

} // file scope

std::string
Compressor::
filenameToCompression(const std::string & filename)
{
    if (ends_with(filename, ".gz") || ends_with(filename, ".gz~"))
        return "gzip";
    if (ends_with(filename, ".bz2") || ends_with(filename, ".bz2~"))
        return "bzip2";
    if (ends_with(filename, ".xz") || ends_with(filename, ".xz~"))
        return "lzma";
    return "none";
}

Compressor *
Compressor::
create(const std::string & compression,
       int level)
{
    if (compression == "gzip" || compression == "gz")
        return new GzipCompressor(level);
    else if (compression == "" || compression == "none")
        return new NullCompressor();
    else throw ML::Exception("unknown compression %s:%d", compression.c_str(),
                             level);
}


/*****************************************************************************/
/* NULL COMPRESSOR                                                           */
/*****************************************************************************/

NullCompressor::
NullCompressor()
{
}

NullCompressor::
~NullCompressor()
{
}

size_t
NullCompressor::
compress(const char * data, size_t len, const OnData & onData)
{
    size_t done = 0;

    while (done < len)
        done += onData(data + done, len - done);
    
    ExcAssertEqual(done, len);

    return done;
}
    
size_t
NullCompressor::
flush(FlushLevel flushLevel, const OnData & onData)
{
    return 0;
}

size_t
NullCompressor::
finish(const OnData & onData)
{
    return 0;
}


/*****************************************************************************/
/* GZIP COMPRESSOR                                                           */
/*****************************************************************************/

struct GzipCompressor::Itl : public z_stream {

    Itl(int compressionLevel)
    {
        zalloc = 0;
        zfree = 0;
        opaque = 0;
        int res = deflateInit2(this, compressionLevel, Z_DEFLATED, 15 + 16, 9,
                               Z_DEFAULT_STRATEGY);
        if (res != Z_OK)
            throw ML::Exception("deflateInit2 failed");
    }

    ~Itl()
    {
        deflateEnd(this);
    }

    size_t pump(const char * data, size_t len, const OnData & onData,
                int flushLevel)
    {
        size_t bufSize = 131072;
        char output[bufSize];
        next_in = (Bytef *)data;
        avail_in = len;
        size_t result = 0;

        do {
            next_out = (Bytef *)output;
            avail_out = bufSize;

            int res = deflate(this, flushLevel);

            
            //cerr << "pumping " << len << " bytes through with flushLevel "
            //     << flushLevel << " returned " << res << endl;

            size_t bytesWritten = (const char *)next_out - output;

            switch (res) {
            case Z_OK:
                if (bytesWritten)
                    onData(output, bytesWritten);
                result += bytesWritten;
                break;

            case Z_STREAM_ERROR:
                throw ML::Exception("Stream error on zlib");

            case Z_STREAM_END:
                if (bytesWritten)
                    onData(output, bytesWritten);
                result += bytesWritten;
                return result;

            default:
                throw ML::Exception("unknown output from deflate");
            };
        } while (avail_in != 0);

        if (flushLevel == Z_FINISH)
            throw ML::Exception("finished without getting to Z_STREAM_END");

        return result;
    }


    size_t compress(const char * data, size_t len, const OnData & onData)
    {
        return pump(data, len, onData, Z_NO_FLUSH);
    }

    size_t flush(FlushLevel flushLevel, const OnData & onData)
    {
        int zlibFlushLevel;
        switch (flushLevel) {
        case FLUSH_NONE:       zlibFlushLevel = Z_NO_FLUSH;       break;
        case FLUSH_AVAILABLE:  zlibFlushLevel = Z_PARTIAL_FLUSH;  break;
        case FLUSH_SYNC:       zlibFlushLevel = Z_SYNC_FLUSH;     break;
        case FLUSH_RESTART:    zlibFlushLevel = Z_FULL_FLUSH;     break;
        default:
            throw ML::Exception("bad flush level");
        }

        return pump(0, 0, onData, zlibFlushLevel);
    }

    size_t finish(const OnData & onData)
    {
        return pump(0, 0, onData, Z_FINISH);
    }
};

GzipCompressor::
GzipCompressor(int compressionLevel)
{
    itl.reset(new Itl(compressionLevel));
}

GzipCompressor::
~GzipCompressor()
{
}

void
GzipCompressor::
open(int compressionLevel)
{
    itl.reset(new Itl(compressionLevel));
}

size_t
GzipCompressor::
compress(const char * data, size_t len, const OnData & onData)
{
    return itl->compress(data, len, onData);
}
    
size_t
GzipCompressor::
flush(FlushLevel flushLevel, const OnData & onData)
{
    return itl->flush(flushLevel, onData);
}

size_t
GzipCompressor::
finish(const OnData & onData)
{
    return itl->finish(onData);
}


/*****************************************************************************/
/* LZMA COMPRESSOR                                                           */
/*****************************************************************************/

} // namespace Datacratic
