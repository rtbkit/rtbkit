// -*- C++ -*-
// (C) Copyright 2011 Datacratic Inc.
// (C) Copyright 2008 CodeRage, LLC (turkanis at coderage dot com)
// (C) Copyright 2003-2007 Jonathan Turkanis
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt.)

// See http://www.boost.org/libs/iostreams for documentation.

// Note: custom allocators are not supported on VC6, since that compiler
// had trouble finding the function zlib_base::do_init.

#ifndef __utils__lzma_h__
#define __utils__lzma_h__

#include <iostream>

#if defined(_MSC_VER) && (_MSC_VER >= 1020)
# pragma once
#endif              

#include <cassert>                            
#include <iosfwd>            // streamsize.                 
#include <memory>            // allocator, bad_alloc.
#include <new>          
#include <boost/config.hpp>  // MSVC, STATIC_CONSTANT, DEDUCED_TYPENAME, DINKUM.
#include <boost/cstdint.hpp> // uint*_t
#include <boost/detail/workaround.hpp>
#include <boost/iostreams/constants.hpp>   // buffer size.
#include <boost/iostreams/detail/config/auto_link.hpp>
#include <boost/iostreams/detail/config/dyn_link.hpp>
#include <boost/iostreams/detail/config/wide_streams.hpp>
#include <boost/iostreams/detail/config/zlib.hpp>
#include <boost/iostreams/detail/ios.hpp>  // failure, streamsize.
#include <boost/iostreams/filter/symmetric.hpp>                
#include <boost/iostreams/pipeline.hpp>                
#include <boost/type_traits/is_same.hpp>
#include <boost/iostreams/filter/zlib.hpp>
#include <lzma.h>

// Must come last.
#ifdef BOOST_MSVC
# pragma warning(push)
# pragma warning(disable:4251 4231 4660)         // Dependencies not exported.
#endif
#include <boost/config/abi_prefix.hpp>           

namespace boost { namespace iostreams {

namespace lzma {
                    // Typedefs

typedef uint32_t uint;
typedef uint8_t byte;
typedef uint32_t ulong;

                    // crc codes

BOOST_IOSTREAMS_DECL extern const int crc32;

                    // Defaults

BOOST_IOSTREAMS_DECL extern const int default_crc;
BOOST_IOSTREAMS_DECL extern const int default_compression;

                    // Compression levels

BOOST_IOSTREAMS_DECL extern const int okay;
BOOST_IOSTREAMS_DECL extern const int stream_end;
BOOST_IOSTREAMS_DECL extern const int stream_error;
BOOST_IOSTREAMS_DECL extern const int version_error;
BOOST_IOSTREAMS_DECL extern const int data_error;
BOOST_IOSTREAMS_DECL extern const int mem_error;
BOOST_IOSTREAMS_DECL extern const int buf_error;

                    // Flush codes

BOOST_IOSTREAMS_DECL extern const int finish;
BOOST_IOSTREAMS_DECL extern const int run;
BOOST_IOSTREAMS_DECL extern const int sync_flush;
BOOST_IOSTREAMS_DECL extern const int full_flush;

const int null                               = 0;

} // End namespace lzma. 

//
// Class name: lzma_params.
// Description: Encapsulates the parameters passed to deflateInit2
//      and inflateInit2 to customize compression and decompression.
//
struct lzma_params {

    // Non-explicit constructor.
    lzma_params( int level           = lzma::default_compression,
                 int crc             = lzma::default_crc)
        : level(level), crc(crc)
        { }
    int level;
    int crc;
};

//
// Class name: lzma_error.
// Description: Subclass of std::ios::failure thrown to indicate
//     lzma errors other than out-of-memory conditions.
//
class BOOST_IOSTREAMS_DECL lzma_error : public BOOST_IOSTREAMS_FAILURE {
public:
    explicit lzma_error(int error);
    int error() const { return error_; }
    static void check BOOST_PREVENT_MACRO_SUBSTITUTION(int error);
private:
    int error_;
};

namespace detail {

std::string lzma_strerror(int code);


class BOOST_IOSTREAMS_DECL lzma_base { 
public:
    typedef char char_type;
protected:
    lzma_base(bool compress, const lzma_params & params = lzma_params());

    ~lzma_base();

    void before( const char*& src_begin, const char* src_end,
                 char*& dest_begin, char* dest_end );

    void after( const char*& src_begin, char*& dest_begin );

    int process(const char * & src_begin,
                const char * & src_end,
                char * & dest_begin,
                char * & dest_end,
                int flushLevel);

public:
    int total_in();
    int total_out();
private:
    bool compress_;
    lzma_stream stream_;
};

//
// Template name: lzma_compressor_impl
// Description: Model of C-Style Filter implementing compression by
//      delegating to the lzma function run.
//
template<typename Alloc = std::allocator<char> >
class lzma_compressor_impl : public lzma_base { 
public: 
    lzma_compressor_impl(const lzma_params& = lzma::default_compression);
    ~lzma_compressor_impl();
    bool filter( const char*& src_begin, const char* src_end,
                 char*& dest_begin, char* dest_end, bool flush );
    void close();
};

//
// Template name: lzma_compressor
// Description: Model of C-Style Filte implementing decompression by
//      delegating to the lzma function inflate.
//
template<typename Alloc = std::allocator<char> >
class lzma_decompressor_impl : public lzma_base {
public:
    lzma_decompressor_impl();
    ~lzma_decompressor_impl();
    bool filter( const char*& begin_in, const char* end_in,
                 char*& begin_out, char* end_out, bool flush );
    void close();
    bool eof() const
    {
        return eof_;
    }
private:
    bool eof_;
};

} // End namespace detail.

//
// Template name: lzma_compressor
// Description: Model of InputFilter and OutputFilter implementing
//      compression using lzma.
//
template<typename Alloc = std::allocator<char> >
struct basic_lzma_compressor 
    : symmetric_filter<detail::lzma_compressor_impl<Alloc>, Alloc> 
{
private:
    typedef detail::lzma_compressor_impl<Alloc>         impl_type;
    typedef symmetric_filter<impl_type, Alloc>  base_type;
public:
    typedef typename base_type::char_type               char_type;
    typedef typename base_type::category                category;
    basic_lzma_compressor( const lzma_params& = lzma::default_compression,
                           int buffer_size = default_device_buffer_size);
    int total_in() {  return this->filter().total_in(); }
};
BOOST_IOSTREAMS_PIPABLE(basic_lzma_compressor, 1)

typedef basic_lzma_compressor<> lzma_compressor;

//
// Template name: lzma_decompressor
// Description: Model of InputFilter and OutputFilter implementing
//      decompression using lzma.
//
template<typename Alloc = std::allocator<char> >
struct basic_lzma_decompressor 
    : symmetric_filter<detail::lzma_decompressor_impl<Alloc>, Alloc> 
{
private:
    typedef detail::lzma_decompressor_impl<Alloc>       impl_type;
    typedef symmetric_filter<impl_type, Alloc>  base_type;
public:
    typedef typename base_type::char_type               char_type;
    typedef typename base_type::category                category;
    basic_lzma_decompressor(int buffer_size = default_device_buffer_size);
    int total_out() {  return this->filter().total_out(); }
    bool eof() { return this->filter().eof(); }
};
BOOST_IOSTREAMS_PIPABLE(basic_lzma_decompressor, 1)

typedef basic_lzma_decompressor<> lzma_decompressor;

//----------------------------------------------------------------------------//

namespace detail {

//------------------Implementation of lzma_compressor_impl--------------------//

template<typename Alloc>
lzma_compressor_impl<Alloc>::lzma_compressor_impl(const lzma_params& p)
    : lzma_base(true, p)
{ }

template<typename Alloc>
lzma_compressor_impl<Alloc>::~lzma_compressor_impl()
{ }

template<typename Alloc>
bool lzma_compressor_impl<Alloc>::filter
    ( const char*& src_begin, const char* src_end,
      char*& dest_begin, char* dest_end, bool flush )
{
    int result = process(src_begin, src_end, dest_begin, dest_end,
                         flush ? lzma::finish : lzma::run);
    return result != lzma::stream_end; 
}

template<typename Alloc>
void lzma_compressor_impl<Alloc>::close()
{
}

//------------------Implementation of lzma_decompressor_impl------------------//

template<typename Alloc>
lzma_decompressor_impl<Alloc>::~lzma_decompressor_impl()
{
}

template<typename Alloc>
lzma_decompressor_impl<Alloc>::lzma_decompressor_impl()
    : lzma_base(false), eof_(false)
{ 
}

template<typename Alloc>
bool lzma_decompressor_impl<Alloc>::filter
    ( const char*& src_begin, const char* src_end,
      char*& dest_begin, char* dest_end, bool /* flush */ )
{
    int result = process(src_begin, src_end, dest_begin, dest_end,
                         lzma::run);
    return !(eof_ = result == lzma::stream_end);
}

template<typename Alloc>
void lzma_decompressor_impl<Alloc>::close() {
    // no real way to close or reopen this...
    //eof_ = false;
    //reset(false, true);
}

} // End namespace detail.

//------------------Implementation of lzma_decompressor-----------------------//

template<typename Alloc>
basic_lzma_compressor<Alloc>::basic_lzma_compressor
(const lzma_params& p, int buffer_size ) 
    : base_type(buffer_size, p) { }

//------------------Implementation of lzma_decompressor-----------------------//

template<typename Alloc>
basic_lzma_decompressor<Alloc>::basic_lzma_decompressor(int buffer_size)
    : base_type(buffer_size) { }

//----------------------------------------------------------------------------//

} } // End namespaces iostreams, boost.

#include <boost/config/abi_suffix.hpp> // Pops abi_suffix.hpp pragmas.
#ifdef BOOST_MSVC
# pragma warning(pop)
#endif

#endif
