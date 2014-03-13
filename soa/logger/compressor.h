/* compressor.h                                                    -*- C++ -*-
   Jeremy Barnes, 19 September 2012
   Copyright (c) 2012 Datacratic.  All rights reserved.

   Interface to a compressor object.

   We prefer this to other solutions as we have full control over when a
   stream is flushed and we can use this to minimise the potential for
   data loss.

   It would be nice to use boost::iostreams for this, but their flush() is
   buggy and there is no way to have precise control over flushing.
*/

#ifndef __logger__compressor_h__
#define __logger__compressor_h__


#include <memory>
#include <functional>
#include <string>

namespace Datacratic {


/*****************************************************************************/
/* COMPRESSOR                                                                */
/*****************************************************************************/

struct Compressor {

    virtual ~Compressor();

    typedef std::function<size_t (const char * data, size_t len)> OnData;

    OnData onData;

    /** Flush levels. */
    enum FlushLevel {
        FLUSH_NONE,     ///< No flushing of compressor
        FLUSH_AVAILABLE,///< Flush all data would be available on decompression
        FLUSH_SYNC,     ///< Flush so that we can find our point in the file
        FLUSH_RESTART,  ///< Flush so we could restart the decompression here
    };

    /** Compress the given data block, and write the result into the
        given buffer.  Returns the number of output bytes written to
        consume the entire input buffer.

        This will call onData zero or more times.
    */
    virtual size_t compress(const char * data, size_t len,
                            const OnData & onData) = 0;
    
    /** Flush the stream at the given flush level.  This will call onData
        zero or more times.
    */
    virtual size_t flush(FlushLevel flushLevel, const OnData & onData) = 0;

    /** Finish the stream... no more data can be written to it afterwards,
        and everything will be put into the compression
    */
    virtual size_t finish(const OnData & onData) = 0;

    /** Convert a filename to a compression scheme. */
    static std::string filenameToCompression(const std::string & filename);

    /** Create a compressor with the given scheme. */
    static Compressor * create(const std::string & compression,
                               int level);
};


/*****************************************************************************/
/* NULL COMPRESSOR                                                           */
/*****************************************************************************/

struct NullCompressor : public Compressor {

    NullCompressor();

    virtual ~NullCompressor();

    virtual size_t compress(const char * data, size_t len,
                            const OnData & onData);
    
    virtual size_t flush(FlushLevel flushLevel, const OnData & onData);

    virtual size_t finish(const OnData & onData);
};

/*****************************************************************************/
/* GZIP COMPRESSOR                                                           */
/*****************************************************************************/

struct GzipCompressor : public Compressor {

    GzipCompressor(int level);

    virtual ~GzipCompressor();

    void open(int level);

    virtual size_t compress(const char * data, size_t len,
                            const OnData & onData);
    
    virtual size_t flush(FlushLevel flushLevel, const OnData & onData);

    virtual size_t finish(const OnData & onData);

private:
    struct Itl;
    std::unique_ptr<Itl> itl;
};

/*****************************************************************************/
/* LZMA COMPRESSOR                                                           */
/*****************************************************************************/

struct LzmaCompressor : public Compressor {

    LzmaCompressor();

    ~LzmaCompressor();

    virtual size_t compress(const char * data, size_t len,
                            const OnData & onData);
    
    virtual size_t flush(FlushLevel flushLevel, const OnData & onData);

    virtual size_t finish(const OnData & onData);

private:
    struct Itl;
    std::unique_ptr<Itl> itl;
};

} // namespace Datacratic

#endif /* __logger__compressor_h__ */
