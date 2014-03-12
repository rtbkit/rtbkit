/* filter.cc
   Jeremy Barnes, 30 May 2011
   Copyright (c) 2011 Datacratic.  All rights reserved.

   Logging output filters.
*/

#include "filter.h"
#include "zlib.h"
#include "jml/arch/exception.h"
#include <iostream>
#include <boost/iostreams/filter/zlib.hpp>
#include <boost/iostreams/filter/bzip2.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include "jml/utils/hex_dump.h"
#include "jml/utils/string_functions.h"
#include "lzma.h"


using namespace std;
using namespace ML;


namespace Datacratic {

std::string print(Direction dir)
{
    switch (dir) {
    case COMPRESS: return "COMPRESS";
    case DECOMPRESS: return "DECOMPRESS";
    default: return ML::format("Direction(%d)", dir);
    }
}

std::ostream & operator << (std::ostream & stream, Direction dir)
{
    return stream << print(dir);
}

std::string print(FlushLevel lvl)
{
    switch (lvl) {
    case FLUSH_NONE: return "FLUSH_NONE";
    case FLUSH_SYNC: return "FLUSH_SYNC";
    case FLUSH_FULL: return "FLUSH_FULL";
    case FLUSH_FINISH: return "FLUSH_FINISH";
    default: return ML::format("FlushLevel(%d)", lvl);
    }
}

std::ostream & operator << (std::ostream & stream, FlushLevel lvl)
{
    return stream << print(lvl);
}


/*****************************************************************************/
/* FILTER                                                                    */
/*****************************************************************************/

Filter::
~Filter()
{
}

void
Filter::
flush(FlushLevel level, boost::function<void ()> onFlushDone)
{
    process(0, 0, level, onFlushDone);
}

void
Filter::
process(const std::string & buf,
        FlushLevel level,
        boost::function<void ()> onFilterDone)
{
    process(buf.c_str(),
            buf.c_str() + buf.length(),
            level,
            onFilterDone);
}

Filter *
Filter::
create(const std::string & extension,
       Direction direction)
{
    if (extension == "z") {
        if (direction == COMPRESS) return new ZlibCompressor();
        else return new ZlibDecompressor();
    }
    else if (extension == "bz" || extension == "bz2") {
        if (direction == COMPRESS) return new Bzip2Compressor();
        else return new Bzip2Decompressor();
    }
    else if (extension == "xz" || extension == "lzma") {
        if (direction == COMPRESS) return new LzmaCompressor();
        else return new LzmaDecompressor();
    }
    else return new IdentityFilter();
}


/*****************************************************************************/
/* IDENTITY FILTER                                                           */
/*****************************************************************************/

void
IdentityFilter::
process(const char * src_begin, const char * src_end,
        FlushLevel level,
        boost::function<void ()> onMessageDone)
{
    onOutput(src_begin, src_end - src_begin, level, onMessageDone);
}


/*****************************************************************************/
/* FILTER STACK                                                              */
/*****************************************************************************/

void
FilterStack::
process(const char * src_begin, const char * src_end,
        FlushLevel level,
        boost::function<void ()> onMessageDone)
{
    if (filters.empty())
        throw Exception("empty filter stack cannot process anything");

    const std::shared_ptr<Filter> & f
        = filters.front();
    f->process(src_begin, src_end, level, onMessageDone);
}

void
FilterStack::
push(std::shared_ptr<Filter> filter)
{
    if (!filters.empty()) {
        filters.back()->onOutput = [=] (const char * p, size_t n, FlushLevel f,
                                        boost::function<void ()> cb)
            {
                filter->process(p, p + n, f, cb);
            };
    }
    
    filter->onOutput = [=] (const char * p, size_t n, FlushLevel f,
                            boost::function<void ()> cb)
        {
            this->onOutput(p, n, f, cb);
        };

    filters.push_back(filter);
}

std::shared_ptr<Filter>
FilterStack::
pop()
{
    throw Exception("can't pop from a filter stack");
}


/*****************************************************************************/
/* ZLIB COMPRESSOR                                                           */
/*****************************************************************************/

using namespace boost::iostreams;

struct ZlibCompressor::Itl
    : public boost::iostreams::detail::zlib_base {

    typedef boost::iostreams::detail::zlib_base Compressor;

    Itl(const boost::iostreams::zlib_params& p,
        Direction direction)
        : direction(direction)
    {
        detail::zlib_allocator<std::allocator<char> > alloc;
        init(p, direction == COMPRESS, alloc);
    }
    
    ~Itl()
    {
        reset(direction == COMPRESS, false);
    }

    void process(const char * & src_begin,
                 const char * & src_end,
                 char * & dest_begin,
                 char * & dest_end,
                 FlushLevel level)
    {
        int flush;
        
        switch (level) {
        case FLUSH_NONE:   flush = zlib::no_flush;    break;
        case FLUSH_SYNC:   flush = zlib::sync_flush;  break;
        case FLUSH_FULL:   flush = Z_FULL_FLUSH;      break;
        case FLUSH_FINISH: flush = zlib::finish;      break;
        default:
            throw Exception("invalid flush level");
        }

        int result = Z_OK;

        before(src_begin, src_end, dest_begin, dest_end);

        //cerr << "calling " << (direction == COMPRESS ? "deflate" : "inflate")
        //     << endl;

        //ML::hex_dump(src_begin, src_end - src_begin);

        result = (direction == COMPRESS
                  ? xdeflate(flush)
                  : xinflate(zlib::no_flush));

        after(src_begin, dest_begin, direction == COMPRESS);

        switch (result) {
        case Z_OK: 
        case Z_STREAM_END: 
            //case Z_BUF_ERROR: 
            break;
        case Z_MEM_ERROR: 
            boost::throw_exception(std::bad_alloc());
        default:
            throw ML::Exception("zlib error %d on %s at byte %d with flush %d: %s",
                                result,
                                (direction == COMPRESS ? "compression" : "decompression"),
                                total_in(), level,
                                zError(result));
        };
    }

    Direction direction;
};

const zlib_params ZlibCompressor::
DEFAULT_PARAMS(zlib::default_compression,
               zlib::deflated,
               zlib::default_window_bits, 
               zlib::default_mem_level, 
               zlib::default_strategy,
               true /* no header */,
               true /* no CRC */);

ZlibCompressor::
ZlibCompressor(const zlib_params& p)
    : itl(new Itl(p, COMPRESS))
{
}
    
ZlibCompressor::
ZlibCompressor(const zlib_params& p, Direction dir)
    : itl(new Itl(p, dir))
{
}
    
ZlibCompressor::
~ZlibCompressor()
{
}

void
ZlibCompressor::
process(const char * src_begin, const char * src_end,
        FlushLevel level,
        boost::function<void ()> onMessageDone)
{
    size_t buffer_size = 65536;
        
    //cerr << "filter " << src_end - src_begin << " bytes with level "
    //     << level << " direction " << itl->direction
    //     << endl;

    do {
        char dest[buffer_size];
        char * dest_begin = dest;
        char * dest_end = dest_begin + buffer_size;
            
        if (src_end != src_begin || level == FLUSH_FINISH) {
            itl->process(src_begin, src_end, dest_begin, dest_end, level);
        }

        bool done = src_begin == src_end;

        size_t bytes_written = dest_begin - dest;
        //cerr << "calling onOutput with " << bytes_written << " bytes"
        //     << " done = " << done << endl;
        //ML::hex_dump(dest, bytes_written);
        onOutput(dest, bytes_written, done ? level : FLUSH_NONE,
                 done ? onMessageDone : boost::function<void ()>());
            
    } while (src_begin != src_end);
}


/*****************************************************************************/
/* ZLIB DECOMPRESSOR                                                         */
/*****************************************************************************/

ZlibDecompressor::
ZlibDecompressor(const boost::iostreams::zlib_params& p)
    : ZlibCompressor(p, DECOMPRESS)
{
}
    
ZlibDecompressor::
~ZlibDecompressor()
{
}

/*****************************************************************************/
/* GZIP COMPRESSOR                                                           */
/*****************************************************************************/

using namespace boost::iostreams;

struct GzipCompressorFilter::Itl : public boost::iostreams::gzip_compressor {
    
    typedef boost::iostreams::gzip_compressor Compressor;

    Itl(const boost::iostreams::gzip_params& p)
        : Compressor(p)
    {
    }
    
    ~Itl()
    {
    }

    void process(const char * & src_begin,
                 const char * & src_end,
                 char * & dest_begin,
                 char * & dest_end,
                 FlushLevel level)
    {
#if 0
        int flush;
        
        switch (level) {
        case FLUSH_NONE:   flush = gzip::no_flush;    break;
        case FLUSH_SYNC:   flush = gzip::sync_flush;  break;
        case FLUSH_FULL:   flush = Z_FULL_FLUSH;      break;
        case FLUSH_FINISH: flush = gzip::finish;      break;
        default:
            throw Exception("invalid flush level");
        }
#endif

        struct Source {
            Source(const char * & src_begin,
                   const char * & src_end)
                : src_begin(src_begin),
                  src_end(src_end)
            {
            }
            
            typedef char char_type;
            struct category
                : dual_use,
                  filter_tag,
                  multichar_tag,
                  closable_tag {
            };

            const char_type * & src_begin;
            const char_type * & src_end;

            size_t read(char * buf, size_t n)
            {
                size_t left = std::distance(src_begin, src_end);
                size_t todo = std::min(left, n);
                std::copy(src_begin, src_begin + todo, buf);
                src_begin += todo;

                //cerr << "source: presented " << todo << " of "
                //     << left << " bytes (wanted " << n << ")" << endl;

                return todo;
            }
        };

        Source source(src_begin, src_end);

        ssize_t n = read(source, dest_begin, dest_end - dest_begin);
        //cerr << "got " << n << " bytes in output" << endl;

        if (n != -1) {
            dest_begin += n;
        }
    }

    Direction direction;
};

const gzip_params GzipCompressorFilter::
DEFAULT_PARAMS;

GzipCompressorFilter::
GzipCompressorFilter(const gzip_params& p)
    : itl(new Itl(p))
{
}
    
GzipCompressorFilter::
~GzipCompressorFilter()
{
}

void
GzipCompressorFilter::
process(const char * src_begin, const char * src_end,
        FlushLevel level,
        boost::function<void ()> onMessageDone)
{
    size_t buffer_size = 65536;
    
    //cerr << "filter " << src_end - src_begin << " bytes with level "
    //     << level << " direction " << itl->direction
    //     << endl;

    do {
        char dest[buffer_size];
        char * dest_begin = dest;
        char * dest_end = dest_begin + buffer_size;
            
        if (src_end != src_begin || level == FLUSH_FINISH) {
            itl->process(src_begin, src_end, dest_begin, dest_end, level);
        }

        bool done = src_begin == src_end;

        size_t bytes_written = dest_begin - dest;
        //cerr << "calling onOutput with " << bytes_written << " bytes"
        //     << " done = " << done << endl;
        //ML::hex_dump(dest, bytes_written);
        onOutput(dest, bytes_written, done ? level : FLUSH_NONE,
                 done ? onMessageDone : boost::function<void ()>());
        
    } while (src_begin != src_end);
}

/*****************************************************************************/
/* GZIP DECOMPRESSOR                                                         */
/*****************************************************************************/

using namespace boost::iostreams;

struct GzipDecompressor::Itl : public boost::iostreams::gzip_decompressor {
    
    typedef boost::iostreams::gzip_decompressor Decompressor;

    Itl()
        : Decompressor()
    {
    }
    
    ~Itl()
    {
    }

    void process(const char * & src_begin,
                 const char * & src_end,
                 char * & dest_begin,
                 char * & dest_end,
                 FlushLevel level)
    {
        struct Source {
            Source(const char * & src_begin,
                   const char * & src_end)
                : src_begin(src_begin),
                  src_end(src_end)
            {
            }
            
            typedef char char_type;
            struct category
                : dual_use,
                  filter_tag,
                  multichar_tag,
                  closable_tag {
            };

            const char_type * & src_begin;
            const char_type * & src_end;

            ssize_t read(char * buf, size_t n)
            {
                size_t left = std::distance(src_begin, src_end);
                size_t todo = std::min(left, n);
                std::copy(src_begin, src_begin + todo, buf);
                src_begin += todo;
                return todo;
            }
        };

        Source source(src_begin, src_end);

        ssize_t n = read(source, dest_begin, dest_end - dest_begin);
        
        dest_begin += n;
    }

    Direction direction;
};

GzipDecompressor::
GzipDecompressor()
    : itl(new Itl())
{
}
    
GzipDecompressor::
~GzipDecompressor()
{
}

void
GzipDecompressor::
process(const char * src_begin, const char * src_end,
        FlushLevel level,
        boost::function<void ()> onMessageDone)
{
    size_t buffer_size = 65536;
        
    //cerr << "filter " << src_end - src_begin << " bytes with level "
    //     << level << " direction " << itl->direction
    //     << endl;

    do {
        char dest[buffer_size];
        char * dest_begin = dest;
        char * dest_end = dest_begin + buffer_size;
            
        if (src_end != src_begin || level == FLUSH_FINISH) {
            itl->process(src_begin, src_end, dest_begin, dest_end, level);
        }

        bool done = src_begin == src_end;

        size_t bytes_written = dest_begin - dest;
        //cerr << "calling onOutput with " << bytes_written << " bytes"
        //     << " done = " << done << endl;
        //ML::hex_dump(dest, bytes_written);
        onOutput(dest, bytes_written, done ? level : FLUSH_NONE,
                 done ? onMessageDone : boost::function<void ()>());
            
    } while (src_begin != src_end);
}


/*****************************************************************************/
/* BZIP2 COMPRESSOR                                                          */
/*****************************************************************************/

using namespace boost::iostreams;

struct Bzip2Compressor::Itl
    : public boost::iostreams::detail::bzip2_base {

    typedef boost::iostreams::detail::bzip2_base Compressor;

    Itl(const boost::iostreams::bzip2_params& p,
        Direction direction)
        : Compressor(p), direction(direction)
    {
        detail::bzip2_allocator<std::allocator<char> > alloc;
        init(direction == COMPRESS, alloc);
    }
    
    ~Itl()
    {
    }

    void process(const char * & src_begin,
                 const char * & src_end,
                 char * & dest_begin,
                 char * & dest_end,
                 FlushLevel level)
    {
        int flush;
        
        switch (level) {
        case FLUSH_NONE:   flush = bzip2::run;         break;
        case FLUSH_FINISH: flush = bzip2::finish;      break;
        case FLUSH_SYNC:
        case FLUSH_FULL:
            throw ML::Exception("bzip2 doesn't support flushing");
        default:
            throw Exception("invalid flush level");
        }

        int result = Z_OK;

        before(src_begin, src_end, dest_begin, dest_end);

        //cerr << "calling " << (direction == COMPRESS ? "deflate" : "inflate")
        //     << endl;

        //ML::hex_dump(src_begin, src_end - src_begin);

        result = (direction == COMPRESS
                  ? compress(flush)
                  : decompress());
        
        after(src_begin, dest_begin);

        using namespace boost::iostreams::bzip2;

        if (result == ok
            || result == run_ok
            || result == finish_ok
            || result == flush_ok
            || result == stream_end) {
            ; 
        }
        else if (result == sequence_error)
            throw Exception("bzip2 sequence error");
        else if (result == param_error)
            throw Exception("bzip2 param error");
        else if (result == mem_error)
            throw Exception("bzip2 mem error");
        else if (result == data_error)
            throw Exception("bzip2 data error");
        else if (result == data_error_magic)
            throw Exception("bzip2 magic data error");
        else if (result == io_error)
            throw Exception("bzip2 io error");
        else if (result == unexpected_eof)
            throw Exception("bzip2 unexpected eof");
        else if (result == outbuff_full)
            throw Exception("bzip2 output buffer full");
        else if (result == config_error)
            throw Exception("bzip2 config error");
        else {
            throw Exception("unknown bzip2 error %d", result);
        }
#if 0
        throw ML::Exception("bzip2 error %d on %s with flush %d: %s",
                                result,
                                (direction == COMPRESS
                                 ? "compression" : "decompression"),
                                level,
                                zError(result));
#endif
    }

    Direction direction;
};

const bzip2_params Bzip2Compressor::DEFAULT_PARAMS;

Bzip2Compressor::
Bzip2Compressor(const bzip2_params& p)
    : itl(new Itl(p, COMPRESS))
{
}
    
Bzip2Compressor::
Bzip2Compressor(const bzip2_params& p, Direction dir)
    : itl(new Itl(p, dir))
{
}
    
Bzip2Compressor::
~Bzip2Compressor()
{
}

void
Bzip2Compressor::
process(const char * src_begin, const char * src_end,
        FlushLevel level,
        boost::function<void ()> onMessageDone)
{
    size_t buffer_size = 65536;
        
    //cerr << "filter " << src_end - src_begin << " bytes with level "
    //     << level << " direction " << itl->direction
    //     << endl;

    do {
        char dest[buffer_size];
        char * dest_begin = dest;
        char * dest_end = dest_begin + buffer_size;
            
        if (src_end != src_begin || level == FLUSH_FINISH) {
            itl->process(src_begin, src_end, dest_begin, dest_end, level);
        }

        bool done = src_begin == src_end;

        size_t bytes_written = dest_begin - dest;
        //cerr << "calling onOutput with " << bytes_written << " bytes"
        //     << " done = " << done << endl;
        //ML::hex_dump(dest, bytes_written);
        onOutput(dest, bytes_written, done ? level : FLUSH_NONE,
                 done ? onMessageDone : boost::function<void ()>());
            
    } while (src_begin != src_end);
}


/*****************************************************************************/
/* BZIP2 DECOMPRESSOR                                                        */
/*****************************************************************************/

const bzip2_params Bzip2Decompressor::DEFAULT_PARAMS(false /* small */);

Bzip2Decompressor::
Bzip2Decompressor(const boost::iostreams::bzip2_params& p)
    : Bzip2Compressor(p, DECOMPRESS)
{
}
    
Bzip2Decompressor::
~Bzip2Decompressor()
{
}


/*****************************************************************************/
/* LZMA COMPRESSOR                                                           */
/*****************************************************************************/

using namespace boost::iostreams;

struct LzmaCompressor::Itl {
    Itl(Direction direction, int level = -1)
        : direction(direction)
    {
        stream_ = LZMA_STREAM_INIT;

        lzma_ret res;
        if (direction == COMPRESS) {
            res = lzma_easy_encoder(&stream_, level, LZMA_CHECK_CRC32);
        }
        else {
            res = lzma_stream_decoder(&stream_, 100 * 1024 * 1024, 0 /* flags */);
        }

        if (res != LZMA_OK)
            throw ML::Exception("LZMA compressor init for direction %d: %s",
                                direction,
                                lzma_strerror(res).c_str());
    }
    
    ~Itl()
    {
        lzma_end(&stream_);
    }

    Direction direction;
    lzma_stream  stream_;

    void before( const char*& src_begin, const char* src_end,
                             char*& dest_begin, char* dest_end )
    {
        stream_.next_in = reinterpret_cast<const uint8_t*>(src_begin);
        stream_.avail_in = static_cast<size_t>(src_end - src_begin);
        stream_.next_out = reinterpret_cast<uint8_t*>(dest_begin);
        stream_.avail_out= static_cast<size_t>(dest_end - dest_begin);
    }

    void after(const char*& src_begin, char*& dest_begin, bool compress)
    {
        const char* next_in = reinterpret_cast<const char*>(stream_.next_in);
        char* next_out = reinterpret_cast<char*>(stream_.next_out);
        src_begin = next_in;
        dest_begin = next_out;
    }

    std::string lzma_strerror(lzma_ret code)
    {
        switch (code) {
        case LZMA_OK: return "Operation completed successfully";
        case LZMA_STREAM_END: return "End of stream was reached";
        case LZMA_NO_CHECK: return "Input stream has no integrity check";
        case LZMA_UNSUPPORTED_CHECK: return "Cannot calculate the integrity check";
        case LZMA_GET_CHECK: return "Integrity check type is now available";
        case LZMA_MEM_ERROR: return "Cannot allocate memory";
        case LZMA_MEMLIMIT_ERROR: return "Memory usage limit was reached";
        case LZMA_FORMAT_ERROR: return "File format not recognized";
        case LZMA_OPTIONS_ERROR: return "Invalid or unsupported options";
        case LZMA_DATA_ERROR: return "Data is corrupt";
        case LZMA_BUF_ERROR: return "No progress is possible";
        case LZMA_PROG_ERROR: return "Programming error";
        default: return ML::format("lzma_ret(%d)", code);
        }
    }

    void process(const char * & src_begin,
                 const char * & src_end,
                 char * & dest_begin,
                 char * & dest_end,
                 FlushLevel level)
    {
        lzma_action action;
        
        if (direction == COMPRESS) {
            switch (level) {
            case FLUSH_NONE:   action = LZMA_RUN;  break;
            case FLUSH_SYNC:   action = LZMA_SYNC_FLUSH;  break;
            case FLUSH_FULL:   action = LZMA_FULL_FLUSH;  break;
            case FLUSH_FINISH: action = LZMA_FINISH;      break;
            default:
                throw Exception("invalid flush level for LZMA processing");
            }
        } else {
            action = LZMA_RUN;
        }

        for (;;) {
            lzma_ret result = LZMA_OK;

            before(src_begin, src_end, dest_begin, dest_end);

            result = lzma_code(&stream_, action);

            after(src_begin, dest_begin, direction == COMPRESS);

            //cerr << "got result " << lzma_strerror(result) << " for action "
            //     << action << endl;

            if (result == LZMA_OK && action != LZMA_RUN) continue;
            if (result == LZMA_OK && action == LZMA_RUN) break;
            if (result == LZMA_STREAM_END) break;

            throw ML::Exception("lzma error %d on %s at byte %d with action %d: %s",
                                result,
                                (direction == COMPRESS ? "compression" : "decompression"),
                                (int)stream_.total_in, action,
                                lzma_strerror(result).c_str());
        }
    }
};

LzmaCompressor::
LzmaCompressor(int level)
    : itl(new Itl(COMPRESS, level))
{
}
    
LzmaCompressor::
LzmaCompressor(Direction dir)
    : itl(new Itl(dir))
{
}
    
LzmaCompressor::
~LzmaCompressor()
{
}

void
LzmaCompressor::
process(const char * src_begin, const char * src_end,
        FlushLevel level,
        boost::function<void ()> onMessageDone)
{
    size_t buffer_size = 65536;
        
    //cerr << "filter " << src_end - src_begin << " bytes with level "
    //     << level << " direction " << itl->direction
    //     << endl;

    do {
        char dest[buffer_size];
        char * dest_begin = dest;
        char * dest_end = dest_begin + buffer_size;
            
        if (src_end != src_begin || level == FLUSH_FINISH) {
            itl->process(src_begin, src_end, dest_begin, dest_end, level);
        }

        bool done = src_begin == src_end;

        size_t bytes_written = dest_begin - dest;
        //cerr << "calling onOutput with " << bytes_written << " bytes"
        //     << " done = " << done << endl;
        //ML::hex_dump(dest, bytes_written);
        onOutput(dest, bytes_written, done ? level : FLUSH_NONE,
                 done ? onMessageDone : boost::function<void ()>());
            
    } while (src_begin != src_end);
}


/*****************************************************************************/
/* LZMA DECOMPRESSOR                                                         */
/*****************************************************************************/

LzmaDecompressor::
LzmaDecompressor()
    : LzmaCompressor(DECOMPRESS)
{
}
    
LzmaDecompressor::
~LzmaDecompressor()
{
}


} // namespace Datacratic
