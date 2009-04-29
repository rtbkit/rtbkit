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
#include "utils/file_functions.h"
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
        return 0;  // we can never get more
    }

    boost::shared_ptr<File_Read_Buffer::Region> region;
};

struct Binary_Input::Stream_Source
    : public Binary_Input::Source {

    enum { DEFAULT_BUF_SIZE = 4096 };

    Stream_Source(std::istream & stream)
        : stream(stream), buf_start(0), buf_size(0)
    {
    }

    virtual ~Stream_Source()
    {
        delete[] buf_start;
    }

    size_t get_more(char * buffer, size_t amount)
    {
        stream.read(buf_start, amount);
        return stream.gcount();
    }

    virtual size_t more(Binary_Input & input, size_t amount)
    {
        /* Check if we need to enlarge the buffer. */
        if (input.avail() + amount > buf_size) {
            /* We need to enlarge the buffer to hold the current data plus the
               new data. */
            /* At least double it */
            size_t new_size = std::max(buf_size * 2, input.avail() + buf_size);
            /* Make it at least one page big */
            new_size = std::max<size_t>(DEFAULT_BUF_SIZE, new_size);
            char * old_buf_start = buf_start;

            buf_start = new char[new_size];

            /* Copy the start of the old buffer into the new one. */
            std::copy(input.pos_, input.end_, buf_start);

            delete[] old_buf_start;
            
            size_t leftover = input.end_ - input.pos_;

            input.offset_ += input.pos_ - old_buf_start;
            input.pos_     = buf_start;

            /* Read some more */
            input.end_ 
                = buf_start + get_more(buf_start + leftover, new_size - leftover)
                + leftover;
            
            input.end_ = buf_start + leftover + new_size;
            buf_size = new_size;
        }
        else {
            /* We can make do with the same buffer.  Reshuffle to put it back
               at the start. */
            std::copy(input.pos_, input.end_, buf_start);
            input.offset_ += (input.pos_ - buf_start);
            input.pos_ = buf_start;
            
            /* Now read it in. */
            input.end_ = buf_start + get_more(buf_start, buf_size);
        }
        
        return input.end_ - buf_start;
    }

    std::istream & stream;

    char * buf_start;
    size_t buf_size;
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

void Binary_Input::open(const File_Read_Buffer & buf)
{
    offset_ = 0;
    pos_ = end_ = 0;
    source.reset(new Buffer_Source(buf));
    source->more(*this, 0);
}

void Binary_Input::open(const std::string & filename)
{
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

void Binary_Input::make_avail(size_t min_avail)
{
    size_t avail = source->more(*this, min_avail);
    if (avail < min_avail)
        throw Exception("Binary_Input: read past end of data");
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

} // namespace DB
} // namespace ML
