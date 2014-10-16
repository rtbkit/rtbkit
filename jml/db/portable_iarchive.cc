/* portable_iarchive.cc                                            -*- C++ -*-
   Jeremy Barnes, 18 March 2005
   Copyright (c) 2005 Jeremy Barnes.  All rights reserved.

   This file is part of "Jeremy's Machine Learning Library", copyright (c)
   1999-2005 Jeremy Barnes.
   
   This program is available under the GNU General Public License, the terms
   of which are given by the file "license.txt" in the top level directory of
   the source code distribution.  If this file is missing, you have no right
   to use the program; please contact the author.

   This program is distributed in the hope that it will be useful, but
   WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
   or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
   for more details.

   ---

   Implementation of the portable iarchive class.
*/

#include "portable_iarchive.h"
#include "jml/utils/file_functions.h"
#include "jml/utils/filter_streams.h"
#include <boost/scoped_ptr.hpp>
#include <iostream>
#include <algorithm>


using namespace std;


namespace ML {
namespace DB {


/*****************************************************************************/
/* BINARY_INPUT                                                              */
/*****************************************************************************/

struct Binary_Input::Source {
    virtual ~Source()
    {
    }

    virtual size_t more(Binary_Input & input, size_t amount) = 0;
};

struct Binary_Input::Buffer_Source
    : public Binary_Input::Source {
    Buffer_Source(const File_Read_Buffer & buf)
        : region(buf.region)
    {
    }

    Buffer_Source(const std::string & filename)
    {
        File_Read_Buffer buf(filename);
        region = buf.region;
    }
    
    virtual ~Buffer_Source()
    {
    }

    virtual size_t more(Binary_Input & input, size_t amount)
    {
        // First try?  Initialize
        if (input.end_ != region->start + region->size) {
            input.pos_ = region->start;
            input.end_ = input.pos_ + region->size;
            input.offset_ = 0;
        }

        return input.avail();  // we can never get more after this
    }

    std::shared_ptr<File_Read_Buffer::Region> region;
};

struct Binary_Input::Stream_Source
    : public Binary_Input::Source {

    enum { DEFAULT_BUF_SIZE = 4096 };

    Stream_Source(std::istream & stream)
        : stream(stream), buf_start(0), buf_size(0)
    {
    }

    Stream_Source(std::istream * stream)
        : stream(*stream), buf_start(0), buf_size(0), owned_stream(stream)
    {
    }

    virtual ~Stream_Source()
    {
        delete[] buf_start;
    }

    size_t get_more(char * buffer, size_t amount)
    {
        //cerr << "get_more: amount = " << amount << endl;
        stream.read(buffer, amount);
        //cerr << "get_more got " << stream.gcount() << endl;
        return stream.gcount();
    }

    virtual size_t more(Binary_Input & input, size_t amount)
    {
        //cerr << "calling more(): input.avail() = "
        //     << input.avail() << " amount = "
        //     << amount << " buf_size = " << buf_size << endl;

        //size_t avail_at_start = input.avail();

        /* Check if we need to enlarge the buffer. */
        if (input.avail() + amount > buf_size) {
            //cerr << "enlarging" << endl;
            /* We need to enlarge the buffer to hold the current data plus the
               new data. */
            /* At least double it */
            size_t new_size = std::max(buf_size * 2, input.avail() + amount);

            //cerr << "new_size = " << new_size << endl;

            /* Make it at least one page big */
            new_size = std::max<size_t>(DEFAULT_BUF_SIZE, new_size);
            char * old_buf_start = buf_start;

            buf_start = new char[new_size];

            /* Copy what's left in the old buffer into the start of the new
               one. */
            std::copy(input.pos_, input.end_, buf_start);

            delete[] old_buf_start;
            
            size_t leftover = input.end_ - input.pos_;

            //cerr << "leftover = " << leftover << endl;

            //input.offset_ += input.pos_ - old_buf_start;
            input.pos_     = buf_start;

            /* Read some more */
            input.end_ 
                = buf_start
                + get_more(buf_start + leftover, new_size - leftover)
                + leftover;
            
            //input.end_ = buf_start + leftover + new_size;
            buf_size = new_size;

            //cerr << "buf_size = " << buf_size << " avail = "
            //     << input.avail() << endl;
        }
        else {
            //cerr << "reading without enlarging" << endl;
            /* We can make do with the same buffer.  Reshuffle to put it back
               at the start. */
            size_t leftover = input.end_ - input.pos_;

            if (input.pos_ != buf_start)
                std::copy(input.pos_, input.end_, buf_start);
            //input.offset_ += (input.pos_ - buf_start);
            input.pos_ = buf_start;
            input.end_ = buf_start + leftover;
            
            /* Now read it in. */
            input.end_ += get_more(buf_start + leftover, buf_size - leftover);
        }

        //cerr << "*** more returned " << (input.end_ - buf_start)
        //     << endl;

        //cerr << "avail(): start " << avail_at_start << " end: "
        //     << input.avail() << endl;
        
        return input.end_ - buf_start;
    }

    std::istream & stream;

    char * buf_start;
    size_t buf_size;

    boost::scoped_ptr<std::istream> owned_stream;
};

struct Binary_Input::No_Source
    : public Binary_Input::Source {
    virtual size_t more(Binary_Input & input, size_t amount)
    {
        return 0;
    }
};

Binary_Input::Binary_Input()
    : offset_(0), pos_(0), end_(0)
{
}

Binary_Input::Binary_Input(const File_Read_Buffer & buf)
    : offset_(0), pos_(0), end_(0)
{
    open(buf);
}

Binary_Input::Binary_Input(const std::string & filename)
    : offset_(0), pos_(0), end_(0)
{
    open(filename);
}

Binary_Input::Binary_Input(std::istream & stream)
    : offset_(0), pos_(0), end_(0)
{
    open(stream);
}

Binary_Input::Binary_Input(const char * c, size_t sz)
    : offset_(0), pos_(0), end_(0)
{
    open(c, sz);
}

void Binary_Input::open(const File_Read_Buffer & buf)
{
    offset_ = 0;
    pos_ = end_ = 0;
    source.reset(new Buffer_Source(buf));
    source->more(*this, 0);
}

bool endsWith(const std::string & str, const std::string & val)
{
    return val.size() <= str.size()
        && str.rfind(val) == str.size() - val.size();
}

void Binary_Input::open(const std::string & filename)
{
    if (endsWith(filename, ".gz")
        || endsWith(filename, ".bz2")
        || endsWith(filename, ".xz")) {
        source.reset(new Stream_Source(new filter_istream(filename)));
        offset_ = 0;
        pos_ = end_ = 0;
        source->more(*this, 0);
        return;
    }

    offset_ = 0;
    pos_ = end_ = 0;
    source.reset(new Buffer_Source(filename));
    source->more(*this, 0);
}

void Binary_Input::open(std::istream & stream)
{
    offset_ = 0;
    pos_ = end_ = 0;
    source.reset(new Stream_Source(stream));
    source->more(*this, 0);
}

void Binary_Input::open(const char * c, size_t sz)
{
    offset_ = 0;
    pos_ = c;
    end_ = c + sz;
    source.reset(new No_Source());
}

void Binary_Input::make_avail(size_t min_avail)
{
    size_t avail = source->more(*this, min_avail);

    //cerr << "avail = " << avail << " min_avail = " << min_avail << endl;
    //cerr << "source: " << typeid(*source).name() << endl;

    if (avail < min_avail)
        throw Exception("Binary_Input: read past end of data");
}

size_t
Binary_Input::
try_make_avail(size_t min_avail)
{
    return source->more(*this, min_avail);
}


/*****************************************************************************/
/* PORTABLE_BIN_IARCHIVE                                                     */
/*****************************************************************************/

portable_bin_iarchive::portable_bin_iarchive()
{
}

portable_bin_iarchive::portable_bin_iarchive(const File_Read_Buffer & buf)
    : Binary_Input(buf)
{
}

portable_bin_iarchive::portable_bin_iarchive(const std::string & filename)
    : Binary_Input(filename)
{
}

portable_bin_iarchive::portable_bin_iarchive(std::istream & stream)
    : Binary_Input(stream)
{
}

portable_bin_iarchive::portable_bin_iarchive(const char * c, size_t sz)
    : Binary_Input(c, sz)
{
}

} // namespace DB
} // namespace ML
