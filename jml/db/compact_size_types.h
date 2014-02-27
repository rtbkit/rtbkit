/* compact_size_types.h                                            -*- C++ -*-
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

   Compact serialization of integers.
*/

#ifndef __db__compact_size_types_h__
#define __db__compact_size_types_h__


#include "persistent_fwd.h"
#include <iostream>
#include <stdint.h>

namespace ML {
namespace DB {


/** Return the number of characters necessary to save the given
    size value, given its actual value. */
int compact_encode_length(unsigned long long val);

void encode_compact(Store_Writer & store, unsigned long long val);
void encode_compact(char * & first, char * last, unsigned long long val);

/** Return the number of characters that need to be available in order to
    read the entire compact_size_t, given the first character.
*/
int compact_decode_length(char firstChar);

unsigned long long decode_compact(Store_Reader & store);
unsigned long long decode_compact(const char * & first, const char * last);

void encode_signed_compact(Store_Reader & store, signed long long val);

signed long long decode_signed_compact(Store_Reader & store);

void serialize_compact_size(Store_Writer & store, unsigned long long size);

unsigned long long
reconstitute_compact_size(Store_Reader & store);


/*****************************************************************************/
/* COMPACT_SIZE_T                                                            */
/*****************************************************************************/

struct compact_size_t {
    compact_size_t() : size_(0) {}
    compact_size_t(unsigned long long size) : size_(size) {}
    compact_size_t(Store_Reader & archive);

    operator unsigned long long () const { return size_; }

    void serialize(Store_Writer & store) const;
    void reconstitute(Store_Reader & store);
    void serialize(std::ostream & stream) const;

    uint64_t size_;
};

std::ostream & operator << (std::ostream & stream, const compact_size_t & s);
IMPL_SERIALIZE_RECONSTITUTE(compact_size_t);
const compact_size_t compact_const(unsigned val);


/*****************************************************************************/
/* COMPACT_INT_T                                                             */
/*****************************************************************************/

struct compact_int_t {
    compact_int_t() : size_(0) {}
    compact_int_t(signed long long size) : size_(size) {}
    compact_int_t(Store_Reader & archive);
    
    operator signed long long () const { return size_; }

    void serialize(Store_Writer & store) const;
    void reconstitute(Store_Reader & store);
    void serialize(std::ostream & stream) const;
    
    int64_t size_;
};

std::ostream & operator << (std::ostream & stream, const compact_int_t & s);
IMPL_SERIALIZE_RECONSTITUTE(compact_int_t);


} // namespace DB
} // namespace ML

#endif /* __db__compact_size_types_h__ */
