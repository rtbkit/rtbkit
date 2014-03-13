// (C) Copyright 2011 Datacratic Inc.
// (C) Copyright 2008 CodeRage, LLC (turkanis at coderage dot com)
// (C) Copyright 2003-2007 Jonathan Turkanis
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt.)

// See http://www.boost.org/libs/iostreams for documentation.

// To configure Boost to work with zlib, see the 
// installation instructions here:
// http://boost.org/libs/iostreams/doc/index.html?path=7

// Define BOOST_IOSTREAMS_SOURCE so that <boost/iostreams/detail/config.hpp> 
// knows that we are building the library (possibly exporting code), rather 
// than using it (possibly importing code).
#define BOOST_IOSTREAMS_SOURCE 

#include "lzma.h"
#include <boost/throw_exception.hpp>
#include <boost/iostreams/detail/config/dyn_link.hpp>
#include <boost/iostreams/filter/zlib.hpp> 
#include <boost/lexical_cast.hpp>
#include <lzma.h>

namespace boost { namespace iostreams {

namespace lzma {

                    // Status codes

const int okay                 = LZMA_OK;
const int stream_end           = LZMA_STREAM_END;
const int data_error           = LZMA_DATA_ERROR;
const int mem_error            = LZMA_MEM_ERROR;
const int buf_error            = LZMA_BUF_ERROR;

                    // Flush codes

const int finish               = LZMA_FINISH;
const int run                  = LZMA_RUN;
const int sync_flush           = LZMA_SYNC_FLUSH;
const int full_flush           = LZMA_FULL_FLUSH;

const int default_crc = LZMA_CHECK_CRC32;
const int default_compression = 6;



} // End namespace lzma. 

//------------------Implementation of lzma_error------------------------------//
                    

lzma_error::lzma_error(int error) 
    : BOOST_IOSTREAMS_FAILURE("lzma error: " + detail::lzma_strerror((lzma_ret)error)),
      error_(error) 
    { }

void lzma_error::check BOOST_PREVENT_MACRO_SUBSTITUTION(int error)
{
    switch (error) {
    case LZMA_OK: 
    case LZMA_STREAM_END: 
    //case LZMA_BUF_ERROR: 
        return;
    case LZMA_MEM_ERROR: 
        boost::throw_exception(std::bad_alloc());
    default:
        boost::throw_exception(lzma_error(error));
        ;
    }
}


namespace detail {

std::string lzma_strerror(int code)
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
    default: return "lzma_ret(" + boost::lexical_cast<std::string>(code) + ")";
    }
}

//------------------Implementation of lzma_base-------------------------------//


lzma_base::lzma_base(bool compress, const lzma_params & params)
    : compress_(compress)
{
    lzma_stream init = LZMA_STREAM_INIT;
    stream_ = init;
    
    lzma_ret res;
    if (compress_)
        res = lzma_easy_encoder(&stream_, params.level,
                                (lzma_check)params.crc);
    else
        res = lzma_stream_decoder(&stream_, 100 * 1024 * 1024, 0 /* flags */);
    
    if (res != LZMA_OK)
        boost::throw_exception(lzma_error(res));
}

lzma_base::~lzma_base()
{
    lzma_end(&stream_);
}

void lzma_base::before( const char*& src_begin, const char* src_end,
                        char*& dest_begin, char* dest_end )
{
    stream_.next_in = reinterpret_cast<const uint8_t*>(src_begin);
    stream_.avail_in = static_cast<size_t>(src_end - src_begin);
    stream_.next_out = reinterpret_cast<uint8_t*>(dest_begin);
    stream_.avail_out= static_cast<size_t>(dest_end - dest_begin);
}

void lzma_base::after(const char*& src_begin, char*& dest_begin)
{
    const char* next_in = reinterpret_cast<const char*>(stream_.next_in);
    char* next_out = reinterpret_cast<char*>(stream_.next_out);
    src_begin = next_in;
    dest_begin = next_out;
}

int lzma_base::process(const char * & src_begin,
                        const char * & src_end,
                        char * & dest_begin,
                        char * & dest_end,
                        int flushLevel)
{
    //cerr << "processing with " << std::distance(src_begin, src_end)
    //     << " bytes in input and " << std::distance(dest_begin, dest_end)
    //     << " bytes in output with flush level " << flushLevel << endl;

    lzma_action action;
        
    if (compress_) {
        switch (flushLevel) {
        case lzma::run:          action = LZMA_RUN;         break;
        case lzma::sync_flush:   action = LZMA_SYNC_FLUSH;  break;
        case lzma::full_flush:   action = LZMA_FULL_FLUSH;  break;
        case lzma::finish:       action = LZMA_FINISH;      break;
        default:
            boost::throw_exception(lzma_error(LZMA_OPTIONS_ERROR));
        }
    } else {
        action = LZMA_RUN;
    }

    lzma_ret result = LZMA_OK;
    
    for (;;) {
        if (dest_begin == dest_end) return result;

        before(src_begin, src_end, dest_begin, dest_end);
        result = lzma_code(&stream_, action);
        after(src_begin, dest_begin);

        //cerr << "    processing with " << std::distance(src_begin, src_end)
        //     << " bytes in input and " << std::distance(dest_begin, dest_end)
        //     << " bytes in output with action " << action
        //     << " returned result " << lzma_strerror(result)
        //     << endl;


        if (result == LZMA_OK && action != LZMA_RUN) continue;
        if (result == LZMA_OK && action == LZMA_RUN) break;
        if (result == LZMA_STREAM_END) break;

        boost::throw_exception(lzma_error(result));
    }

    return result;
}

} // End namespace detail.

//----------------------------------------------------------------------------//

} } // End namespaces iostreams, boost.
