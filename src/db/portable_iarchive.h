/* portable_iarchive.h                                             -*- C++ -*-
   Jeremy Barnes, 13 March 2005
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

   Portable format for iarchives.
*/

#ifndef __db__portable_iarchive_h__
#define __db__portable_iarchive_h__

#include <algorithm>
#include <boost/shared_ptr.hpp>
#include "serialization_order.h"
#include "utils/floating_point.h"
#include "arch/exception.h"
#include "compact_size_types.h"
#include <vector>
#include <boost/array.hpp>

namespace boost {

template<typename T, std::size_t NumDims, class Allocator>
class multi_array;

} // namespace boost

namespace ML {

class File_Read_Buffer;

namespace DB {

/*****************************************************************************/
/* BINARY_INPUT                                                              */
/*****************************************************************************/

/* This is a class that can get its input from an istream or a buffer. */

struct Binary_Input {
public:
    Binary_Input();
    Binary_Input(const File_Read_Buffer & buf);
    Binary_Input(const std::string & filename);
    Binary_Input(std::istream & stream);

    void open(const File_Read_Buffer & buf);
    void open(const std::string & filename);
    void open(std::istream & stream);

    size_t avail() const { return end_ - pos_; }

    size_t must_have(size_t amount)
    {
        if (avail() < amount) make_avail(amount);
        return avail();
    }

    /** Trys to make it available, but if it's not then it just returns
        what is available. */
    size_t try_to_have(size_t amount)
    {
        if (avail() < amount) return try_make_avail(amount);
        return avail();
    }

    void skip(size_t amount)
    {
        if (amount >= avail())
            make_avail(amount);

        if (avail() < amount)
            throw Exception("skipped past end of store");

        offset_ += amount;
        pos_ += amount;
    }

    //const char * start() const { return start_; }
    const char * pos() const { return pos_; }
    const char * end() const { return end_; }

    char operator * () const { return *pos_; }

    char operator [] (int idx) const { return pos_[idx]; }

    size_t offset() const { return offset_; }

private:
    size_t offset_;       ///< Offset of start from archive start
    const char * pos_;    ///< Position in memory region
    const char * end_;    ///< End of memory region

    void make_avail(size_t min_avail);
    size_t try_make_avail(size_t amount);    

    struct Source;
    struct Buffer_Source;
    struct Stream_Source;
    boost::shared_ptr<Source> source;
};


/*****************************************************************************/
/* PORTABLE_BIN_IARCHIVE                                                     */
/*****************************************************************************/

class portable_bin_iarchive
    : public Binary_Input {
public:
    portable_bin_iarchive();
    portable_bin_iarchive(const File_Read_Buffer & buf);
    portable_bin_iarchive(const std::string & filename);
    portable_bin_iarchive(std::istream & stream);

    void load(unsigned char & x)
    {
        load_binary(&x, 1);
    }

    void load(signed char & x)
    {
        load_binary(&x, 1);
    }

    void load(char & x)
    {
        load_binary(&x, 1);
    }

    void load(unsigned short & x)
    {
        uint16_t xx;
        load_binary(&xx, 2);
        x = native_order(xx);
    }

    void load(signed short & x)
    {
        int16_t xx;
        load_binary(&xx, 2);
        x = native_order(xx);
    }

    void load(unsigned int & x)
    {
        uint32_t xx;
        load_binary(&xx, 4);
        x = native_order(xx);
    }

    void load(signed int & x)
    {
        int32_t xx;
        load_binary(&xx, 4);
        x = native_order(xx);
    }

    void load(unsigned long & x)
    {
        compact_size_t sz(*this);
        x = sz;
    }

    void load(signed long & x)
    {
        compact_int_t sz(*this);
        x = sz;
    }

    void load(unsigned long long & x)
    {
        compact_size_t sz(*this);
        x = sz;
    }

    void load(signed long long & x)
    {
        compact_int_t sz(*this);
        x = sz;
    }
    
    void load(bool & x)
    {
        unsigned char xx;
        load_binary(&xx, 1);
        x = xx;
    }
    
    void load(float & x)
    {
        // Saved as the 4 bytes in network order
        uint32_t xx;
        load_binary(&xx, 4);
        xx = native_order(xx);
        x = reinterpret_as_float(xx);
    }
    
    void load(double & x)
    {
        // Saved as the 8 bytes in network order
        uint64_t xx;
        load_binary(&xx, 8);
        xx = native_order(xx);
        x = reinterpret_as_double(xx);
    }
    
    void load(long double & x)
    {
        throw Exception("long double not supported yet");
    }

    void load(std::string & str)
    {
        compact_size_t size(*this);
        str.resize(size);
        load_binary(&str[0], size);
    }

    void load(const char * & str)
    {
        compact_size_t size(*this);
        char * res = new char[size];  // keep track of this?
        load_binary(res, size);
        str = res;
    }

    template<class T, class A>
    void load(std::vector<T, A> & vec)
    {
        compact_size_t sz(*this);

        std::vector<T, A> v;
        v.reserve(sz);
        for (unsigned i = 0;  i < sz;  ++i) {
            T t;
            *this >> t;
            v.push_back(t);
        }
        vec.swap(v);
    }

    template<typename T, std::size_t NumDims, class Allocator>
    void load(boost::multi_array<T, NumDims, Allocator> & arr)
    {
        using namespace std;
        char version;
        load(version);
        if (version != 1)
            throw Exception("unknown multi array version");

        char nd;
        load(nd);
        if (nd != NumDims)
            throw Exception("NumDims wrong");

        boost::array<size_t, NumDims> sizes;
        for (unsigned i = 0;  i < NumDims;  ++i) {
            compact_size_t sz(*this);
            sizes[i] = sz;
        }

        arr.resize(sizes);

        size_t ne = arr.num_elements();
        T * el = arr.data();
        for (unsigned i = 0;  i < ne;  ++i, ++el)
            *this >> *el;
    }

    void load_binary(void * address, size_t size)
    {
        must_have(size);
        std::copy(pos(), pos() + size,
                  reinterpret_cast<char *>(address));
        skip(size);
    }
};

} // namespace DB
} // namespace ML

#endif /* __db__portable_iarchive_h__ */


