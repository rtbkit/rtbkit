/* compact_size_types.cc
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

   Implementation of persistence for compact size types.
*/

#include "compact_size_types.h"
#include "jml/compiler/compiler.h"
#include "jml/arch/bitops.h"
#include "persistent.h"
#include <stdint.h>
#include <iomanip>


using namespace std;


namespace ML {
namespace DB {

const compact_size_t compact_const(unsigned val)
{
    return compact_size_t(val);
}

/*****************************************************************************/
/* UNSIGNED VERSIONS                                                         */
/*****************************************************************************/

/* byte1     extra     min     max  max32bit   
   0 xxxxxxx     0    0     2^7-1
   10 xxxxxx     1    2^7   2^14-1
   110 xxxxx     2    2^14  2^21-1
   1110 xxxx     3    2^21  2^28-1
   11110 xxx     4    2^28  2^35-1  (2^32-1)
   111110 xx     5    2^35  2^42-1
   1111110 x     6    2^42  2^49-1
   11111110      7    2^49  2^56-1
   11111111      8    2^56  2^64-1
*/

int compact_encode_length(unsigned long long val)
{
    int highest = highest_bit(val);
    int idx = highest / 7;
    int len = idx + 1;
    return len;
}

void encode_compact(Store_Writer & store, unsigned long long val)
{
    char buf[9];

    /* Length depends upon highest bit / 7 */
    int highest = highest_bit(val);
    int idx = highest / 7;
    int len = idx + 1;

    //cerr << "val = " << val << " highest = " << highest << " len = "
    //     << len << endl;

    /* Pack it into the back bytes. */
    for (int i = len-1;  i >= 0;  --i) {
        //cerr << "i = " << i << endl;
        buf[i] = val & 0xff;
        val >>= 8;
    }

    /* Add the indicator to the first byte. */
    uint32_t indicator = ~((1 << (8-idx)) - 1);
    buf[0] |= indicator;
    
    //size_t offset = store.offset();
    store.save_binary(buf, len);
    //size_t offset_after = store.offset();

    //if (offset_after - offset != len)
    //    throw Exception("offsets are wrong");
}

void encode_compact(char * & first, char * last, unsigned long long val)
{
    /* Length depends upon highest bit / 7 */
    int highest = highest_bit(val);
    int idx = highest / 7;
    int len = idx + 1;

    if (first + len > last)
        throw ML::Exception("not enough space to encode compact_size_t");

    /* Pack it into the back bytes. */
    for (int i = len-1;  i >= 0;  --i) {
        //cerr << "i = " << i << endl;
        first[i] = val & 0xff;
        val >>= 8;
    }

    /* Add the indicator to the first byte. */
    uint32_t indicator = ~((1 << (8-idx)) - 1);
    first[0] |= indicator;

    first += len;
}

int compact_decode_length(char firstChar)
{
    uint8_t marker = firstChar;
    // no bits set=-1, so len=9 as reqd
    int len = 8 - highest_bit((char)~marker);
    return len;
}

unsigned long long decode_compact(Store_Reader & store)
{
    /* Find the first zero bit in the marker.  We do this by bit flipping
       and finding the first 1 bit in the result. */
    store.must_have(1);

    int len = compact_decode_length(*store);

    //cerr << "marker = " << int(marker) << endl;
    //cerr << "len = " << len << endl;

    /* Make sure this data is available. */
    store.must_have(len);

    /* Construct our value from the bytes. */
    unsigned long long result = 0;
    for (int i = 0;  i < len;  ++i) {
        int val = store[i];
        if (val < 0) val += 256;

        result <<= 8;
        result |= val; 
        //cerr << "i " << i << " result " << result << endl;
    }

    /* Filter off the top bits, which told us the length. */
    if (len == 9) ;
    else {
        int bits = len * 7;
        //cerr << "bits = " << bits << endl;
        result &= ((1ULL << bits)-1);
        //cerr << "result = " << result << endl;
    }

    /* Skip the data.  Makes sure we are in sync even if we throw. */
    store.skip(len);

    return result;
}

unsigned long long decode_compact(const char * & first, const char * last)
{
    /* Find the first zero bit in the marker.  We do this by bit flipping
       and finding the first 1 bit in the result. */
    if (first >= last)
        throw Exception("not enough bytes to decode compact_size_t");
        
    int len = compact_decode_length(*first);
    if (first + len > last)
        throw Exception("not enough bytes to decode compact_size_t");

    /* Construct our value from the bytes. */
    unsigned long long result = 0;
    for (int i = 0;  i < len;  ++i) {
        int val = first[i];
        if (val < 0) val += 256;

        result <<= 8;
        result |= val; 
        //cerr << "i " << i << " result " << result << endl;
    }

    /* Filter off the top bits, which told us the length. */
    if (len == 9) ;
    else {
        int bits = len * 7;
        //cerr << "bits = " << bits << endl;
        result &= ((1ULL << bits)-1);
        //cerr << "result = " << result << endl;
    }

    first += len;

    return result;
}


/*****************************************************************************/
/* COMPACT_SIZE_T                                                            */
/*****************************************************************************/

compact_size_t::compact_size_t(Store_Reader & store)
{
    size_ = decode_compact(store);
}
    
void compact_size_t::serialize(Store_Writer & store) const
{
    encode_compact(store, size_);
}

void compact_size_t::reconstitute(Store_Reader & store)
{
    size_ = decode_compact(store);
}

void compact_size_t::serialize(std::ostream & stream) const
{
    Store_Writer writer(stream);
    serialize(writer);
}

std::ostream & operator << (std::ostream & stream, const compact_size_t & s)
{
    stream << s.size_;
    return stream;
}


/*****************************************************************************/
/* SIGNED VERSIONS                                                           */
/*****************************************************************************/

/* byte1      byte2    others  range
   0 s xxxxxx          0       2^6
   10 s xxxxx xxxxxxxx 0       2^13
   110 s xxxx xxxxxxxx 1       2^20
   1110 s xxx xxxxxxxx 2       2^27
   11110 s xx xxxxxxxx 3       2^34 (2^31)
   111110 s x xxxxxxxx 4       2^41 
   1111110 s  xxxxxxxx 5       2^48
   11111110   sxxxxxxx 6       2^55
   11111111   sxxxxxxx 7       2^63
*/

void encode_signed_compact(Store_Reader & store, signed long long val)
{
    throw Exception("not implemented");
}

signed long long decode_signed_compact(Store_Reader & store)
{
    throw Exception("not implemented");
}



} // namespace DB
} // namespace ML

