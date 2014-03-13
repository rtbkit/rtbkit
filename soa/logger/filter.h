/* filter.h                                                        -*- C++ -*-
   Jeremy Barnes, 29 May 2011
   Copyright (c) 2011 Datacratic.  All rights reserved.

*/

#ifndef __logger__filter_h__
#define __logger__filter_h__


#include <boost/function.hpp>
#include <boost/shared_ptr.hpp>
#include "soa/sigslot/slot.h"

namespace boost {
namespace iostreams {

struct zlib_params;
struct bzip2_params;
struct gzip_params;

} // namespace iostreams
} // namespace boost


namespace Datacratic {

enum Direction {
    COMPRESS,
    DECOMPRESS
};

std::string print(Direction dir);
std::ostream & operator << (std::ostream & stream, Direction dir);

enum FlushLevel {
    FLUSH_NONE,   ///< Don't flush at all
    FLUSH_SYNC,   ///< Flush so that all current data could be reconstructed
    FLUSH_FULL,   ///< Flush so that anything after is independent of before
    FLUSH_FINISH  ///< Flush so that all footers, etc are written
};

std::string print(FlushLevel lvl);
std::ostream & operator << (std::ostream & stream, FlushLevel lvl);


/*****************************************************************************/
/* FILTER                                                                    */
/*****************************************************************************/

struct Filter {

    virtual ~Filter();

    typedef void (OnOutputFn) (const char *, size_t, FlushLevel, boost::function<void ()>);
    typedef boost::function<OnOutputFn> OnOutput;
    OnOutput onOutput;

    typedef void (OnErrorFn) (const std::string &);
    typedef boost::function<OnErrorFn> OnError;
    OnError onError;

    virtual void flush(FlushLevel level,
                       boost::function<void ()> onFlushDone
                           = boost::function<void ()>());

    virtual void process(const std::string & buf,
                         FlushLevel level = FLUSH_NONE,
                         boost::function<void ()> onFilterDone
                             = boost::function<void ()>());

    virtual void process(const char * first, const char * last,
                         FlushLevel level = FLUSH_NONE,
                         boost::function<void ()> onFilterDone
                             = boost::function<void ()>()) = 0;

    static Filter * create(const std::string & extension,
                           Direction direction);
};


/*****************************************************************************/
/* IDENTITY FILTER                                                           */
/*****************************************************************************/

struct IdentityFilter : public Filter {

    using Filter::process;

    virtual void process(const char * src_begin, const char * src_end,
                         FlushLevel level,
                         boost::function<void ()> onMessageDone);
};


/*****************************************************************************/
/* FILTER STACK                                                              */
/*****************************************************************************/

struct FilterStack : public Filter {

    using Filter::process;

    virtual void process(const char * src_begin, const char * src_end,
                         FlushLevel level,
                         boost::function<void ()> onMessageDone);

    void push(std::shared_ptr<Filter> filter);
    std::shared_ptr<Filter> pop();
    size_t size() const { return filters.size(); }
    bool empty() const { return filters.empty(); }

private:
    std::vector<std::shared_ptr<Filter> > filters;
};


/*****************************************************************************/
/* ZLIB COMPRESSOR                                                           */
/*****************************************************************************/

struct ZlibCompressor
    : public Filter {

    static const boost::iostreams::zlib_params DEFAULT_PARAMS;

    ZlibCompressor(const boost::iostreams::zlib_params& p
                       = DEFAULT_PARAMS);
    
    ~ZlibCompressor();

    using Filter::process;

    virtual void process(const char * src_begin, const char * src_end,
                         FlushLevel level,
                         boost::function<void ()> onMessageDone);

protected:
    ZlibCompressor(const boost::iostreams::zlib_params& p,
                   Direction direction);

private:
    struct Itl;
    std::shared_ptr<Itl> itl;
    Direction direction;
};


/*****************************************************************************/
/* ZLIB DECOMPRESSOR                                                         */
/*****************************************************************************/

struct ZlibDecompressor
    : public ZlibCompressor {

    ZlibDecompressor(const boost::iostreams::zlib_params& p
                     = DEFAULT_PARAMS);
    
    ~ZlibDecompressor();
};


/*****************************************************************************/
/* GZIP COMPRESSOR                                                           */
/*****************************************************************************/

struct GzipCompressorFilter: public Filter {

    static const boost::iostreams::gzip_params DEFAULT_PARAMS;

    GzipCompressorFilter(const boost::iostreams::gzip_params& p
                       = DEFAULT_PARAMS);
    
    ~GzipCompressorFilter();

    using Filter::process;

    virtual void process(const char * src_begin, const char * src_end,
                         FlushLevel level,
                         boost::function<void ()> onMessageDone);

private:
    struct Itl;
    std::shared_ptr<Itl> itl;
};


/*****************************************************************************/
/* GZIP DECOMPRESSOR                                                         */
/*****************************************************************************/

struct GzipDecompressor : public Filter {
    GzipDecompressor();
    
    ~GzipDecompressor();
    
    using Filter::process;

    virtual void process(const char * src_begin, const char * src_end,
                         FlushLevel level,
                         boost::function<void ()> onMessageDone);

private:
    struct Itl;
    std::shared_ptr<Itl> itl;
};


/*****************************************************************************/
/* BZIP2 COMPRESSOR                                                          */
/*****************************************************************************/

struct Bzip2Compressor
    : public Filter {

    static const boost::iostreams::bzip2_params DEFAULT_PARAMS;

    Bzip2Compressor(const boost::iostreams::bzip2_params& p
                       = DEFAULT_PARAMS);
    
    ~Bzip2Compressor();

    using Filter::process;

    virtual void process(const char * src_begin, const char * src_end,
                         FlushLevel level,
                         boost::function<void ()> onMessageDone);

protected:
    Bzip2Compressor(const boost::iostreams::bzip2_params& p,
                   Direction direction);

private:
    struct Itl;
    std::shared_ptr<Itl> itl;
    Direction direction;
};


/*****************************************************************************/
/* BZIP2 DECOMPRESSOR                                                        */
/*****************************************************************************/

struct Bzip2Decompressor
    : public Bzip2Compressor {

    static const boost::iostreams::bzip2_params DEFAULT_PARAMS;

    Bzip2Decompressor(const boost::iostreams::bzip2_params& p
                     = DEFAULT_PARAMS);
    
    ~Bzip2Decompressor();
};


/*****************************************************************************/
/* LZMA COMPRESSOR                                                           */
/*****************************************************************************/

struct LzmaCompressor
    : public Filter {

    LzmaCompressor(int level = 6);
    ~LzmaCompressor();

    using Filter::process;

    virtual void process(const char * src_begin, const char * src_end,
                         FlushLevel level,
                         boost::function<void ()> onMessageDone);

protected:
    LzmaCompressor(Direction direction);

private:
    struct Itl;
    std::shared_ptr<Itl> itl;
    Direction direction;
};


/*****************************************************************************/
/* LZMA DECOMPRESSOR                                                         */
/*****************************************************************************/

struct LzmaDecompressor
    : public LzmaCompressor {

    LzmaDecompressor();
    
    ~LzmaDecompressor();
};


} // namespace Datacratic


#endif /* __logger__filter_h__ */
