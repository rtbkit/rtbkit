/* portable_oarchive.h                                             -*- C++ -*-
   Jeremy Barnes, 17 March 2005
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

   Portable output archive.
*/

#ifndef __db__portable_oarchive_h__
#define __db__portable_oarchive_h__


#include <algorithm>
#include "arch/exception.h"
#include "serialization_order.h"
#include "utils/floating_point.h"
#include "compact_size_types.h"
#include <boost/shared_ptr.hpp>
#include <boost/type_traits.hpp>
#include <vector>
#include <string.h>

namespace ML {
namespace DB {


class Nested_Writer;


/*****************************************************************************/
/* PORTABLE_BIN_OARCHIVE                                                     */
/*****************************************************************************/

class portable_bin_oarchive {
public:
    portable_bin_oarchive();
    portable_bin_oarchive(const std::string & filename);
    portable_bin_oarchive(std::ostream & stream);

    void open(const std::string & filename);
    void open(std::ostream & stream);

    void save(unsigned char x)
    {
        save_binary(&x, 1);
    }

    void save(signed char x)
    {
        save_binary(&x, 1);
    }

    void save(unsigned short x)
    {
        uint16_t xx = x;
        if (xx != x) throw Exception("truncated");
        xx = serialization_order(xx);
        save_binary(&xx, 2);
    }

    void save(signed short x)
    {
        int16_t xx = x;
        if (xx != x) throw Exception("truncated");
        xx = serialization_order(xx);
        save_binary(&xx, 2);
    }

    void save(unsigned int x)
    {
        uint32_t xx = x;
        if (xx != x) throw Exception("truncated");
        xx = serialization_order(xx);
        save_binary(&xx, 4);
    }

    void save(signed int x)
    {
        int32_t xx = x;
        if (xx != x) throw Exception("truncated");
        xx = serialization_order(xx);
        save_binary(&xx, 4);
    }
    
    void save(unsigned long x)
    {
        if (!stream)
            throw Exception("Writing to unopened portable_bin_oarchive");
        compact_size_t sz(x);
        sz.serialize(*stream);
    }

    void save(signed long x)
    {
        if (!stream)
            throw Exception("Writing to unopened portable_bin_oarchive");
        compact_int_t sz(x);
        sz.serialize(*stream);
    }

    void save(unsigned long long x)
    {
        if (!stream)
            throw Exception("Writing to unopened portable_bin_oarchive");
        compact_size_t sz(x);
        sz.serialize(*stream);
    }

    void save(signed long long x)
    {
        if (!stream)
            throw Exception("Writing to unopened portable_bin_oarchive");
        compact_int_t sz(x);
        sz.serialize(*stream);
    }
    
    void save(bool x)
    {
        unsigned char xx = x;
        save_binary(&xx, 1);
    }
    
    void save(float x)
    {
        // Saved as the 4 bytes in network order
        uint32_t xx = reinterpret_as_int(x);
        xx = serialization_order(xx);
        save_binary(&xx, 4);
    }

    void save(double x)
    {
        // Saved as the 8 bytes in network order
        uint64_t xx = reinterpret_as_int(x);
        xx = serialization_order(xx);
        save_binary(&xx, 8);
    }
    
    void save(long double x)
    {
        throw Exception("long double not supported yet");
    }

    void save(const std::string & str)
    {
        if (!stream)
            throw Exception("Writing to unopened portable_bin_oarchive");
        compact_size_t size(str.length());
        size.serialize(*stream);
        save_binary(&str[0], size);
    }

    void save(const char * str)
    {
        if (!stream)
            throw Exception("Writing to unopened portable_bin_oarchive");
        compact_size_t size(strlen(str));
        size.serialize(*stream);
        save_binary(str, size);
    }

    template<class T, class A>
    void save(const std::vector<T, A> & vec)
    {
        compact_size_t size(vec.size());
        size.serialize(*stream);
        for (unsigned i = 0;  i < vec.size();  ++i)
            *this << vec[i];
    }

    void save(const Nested_Writer & writer);

    void save_binary(const void * address, size_t size)
    {
        if (!stream)
            throw Exception("Writing to unopened portable_bin_oarchive");
        stream->write((char *)address, size);
        if (!(*stream))
            throw Exception("Error writing to stream");
    }

private:
    std::ostream * stream;
    boost::shared_ptr<std::ostream> owned_stream;
};


} // namespace DB
} // namespace ML

#endif /* __db__portable_oarchive_h__ */
