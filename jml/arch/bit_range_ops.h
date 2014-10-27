/* bit_range_ops.h                                                 -*- C++ -*-
   Jeremy Barnes, 23 March 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

   Operations for operating over a range of bits.
*/

#ifndef __arch__bit_range_ops_h__
#define __arch__bit_range_ops_h__

#include "jml/arch/arch.h"
#include "jml/compiler/compiler.h"
#include "jml/arch/format.h"
#include "jml/arch/atomic_ops.h"
#include <cstddef>
#include <stdint.h>
#include <algorithm>
#include <boost/iterator/iterator_facade.hpp>


namespace ML {

typedef uint32_t shift_t;

/** Performs the same as the shrd instruction in x86 land: shifts a low and
    a high value together and returns a middle set of bits.
    
    2n                n                    0
    +-----------------+--------------------+
    |      high       |       low          |
    +--------+--------+-----------+--------+
             |     result         |<-bits---
             +--------------------+
                                     
    NOTE: bits MUST be LESS than the number of bits in the size.  The result
    is undefined if it's equal or greater.

*/
template<typename T>
JML_ALWAYS_INLINE JML_COMPUTE_METHOD
T shrd_emulated(T low, T high, shift_t bits)
{
    enum { TBITS = sizeof(T) * 8 };
    //
    //if (JML_UNLIKELY(bits >= TBITS)) return high >> (bits - TBITS);
    //if (JML_UNLIKELY(bits == TBITS)) return high;
    low >>= bits;
    high <<= (TBITS - bits);
    return low | high;
}

#if defined( JML_INTEL_ISA ) && ! defined(JML_COMPILER_NVCC)

template<typename T>
JML_ALWAYS_INLINE JML_PURE_FN JML_COMPUTE_METHOD
T shrd(T low, T high, shift_t bits)
{
    enum { TBITS = sizeof(T) * 8 };
    if (JML_UNLIKELY(bits > TBITS)) return 0;

    __asm__("shrd   %[bits], %[high], %[low] \n\t"
            : [low] "+r,r" (low)
            : [bits] "J,c" ((uint8_t)bits), [high] "r,r" (high)
            : "cc"
            );

    return low;

}

// There's no 8 bit shrd instruction available
JML_ALWAYS_INLINE JML_COMPUTE_METHOD
unsigned char shrd(unsigned char low, unsigned char high, shift_t bits)
{
    return shrd_emulated(low, high, bits);
}

// There's no 8 bit shrd instruction available
JML_ALWAYS_INLINE JML_COMPUTE_METHOD
signed char shrd(signed char low, signed char high, shift_t bits)
{
    return shrd_emulated(low, high, bits);
}

#if JML_BITS == 32

// There's no 8 byte shrd instruction available
JML_ALWAYS_INLINE JML_COMPUTE_METHOD
unsigned long long shrd(unsigned long long low, unsigned long long high, shift_t bits)
{
    return shrd_emulated(low, high, bits);
}

// There's no 8 byte shrd instruction available
JML_ALWAYS_INLINE JML_COMPUTE_METHOD
signed long long shrd(signed long long low, signed long long high, shift_t bits)
{
    return shrd_emulated(low, high, bits);
}
#endif


#else // no SHRD instruction available

template<typename T>
JML_ALWAYS_INLINE JML_COMPUTE_METHOD
T shrd(T low, T high, shift_t bits)
{
    return shrd_emulated(low, high, bits);
}

#endif // JML_INTEL_ISA


#if defined( JML_INTEL_ISA ) && ! defined(JML_COMPILER_NVCC) && false

/** Mask off the highest bits of the results, leaving n bits. */

template<typename T>
JML_ALWAYS_INLINE JML_COMPUTE_METHOD
T maskLower(T val, shift_t bits)
{
    // use SHL instruction
}

#else

template<typename T>
JML_ALWAYS_INLINE JML_COMPUTE_METHOD
T maskLower(T val, shift_t bits)
{
    T full = bits < sizeof(T) * 8;
    T mask = ((full << bits) - 1);
    return val & mask;
}

#endif

/** Extract the bits from bit to bit+numBits starting at the pointed to
    address.  There can be a maximum of one word extracted in this way.

    NO ADJUSTMENT IS DONE ON THE ADDRESS; bit is expected to already be
    there.

    Algorithm:
    1.  If it fits in a single word: copy it, shift it right
    2.  Otherwise, do a double shift right across the two memory locations
*/
template<typename Data>
JML_ALWAYS_INLINE JML_COMPUTE_METHOD
Data extract_bit_range(const Data * p, size_t bit, shift_t bits)
{
    if (bits == 0) return 0;

    Data result = p[0];

    enum { DBITS = sizeof(Data) * 8 };
    
    if (bit + bits > DBITS) {
        // We need to extract from the two values
        result = shrd(result, p[1], bit);
    }
    else result >>= bit;

    if (bits == DBITS) return result;

    result *= Data(bits != 0);  // bits == 0: should return 0; mask doesn't work
    return result & ((Data(1) << bits) - 1);
}

/** Same, but the low and high values are passed it making it pure. */
template<typename Data>
JML_ALWAYS_INLINE JML_PURE_FN JML_COMPUTE_METHOD
Data extract_bit_range(Data p0, Data p1, size_t bit, shift_t bits)
{
    return maskLower(shrd(p0, p1, bit), bits); // extract and mask
}

/** Set the given range of bits in out to the given value.  Note that val
    mustn't have bits set outside bits, and it must entirely fit within
    the value. */
template<typename Data>
JML_ALWAYS_INLINE JML_PURE_FN
Data set_bits(Data in, Data val, shift_t bit, shift_t bits)
{
    // Create a mask with the bits to modify
    Data mask = bits >= 64 ? -1 : (Data(1) << bits) - 1;
    mask <<= bit;
    val  <<= bit;

#if 0
    using namespace std;
    cerr << "set_bits: in = " << in << " val = " << val
         << " bit = " << (int)bit << " mask = " << mask
         << endl;
#endif

    return (in & ~mask) | (val & mask);
}

template<typename Data>
JML_ALWAYS_INLINE
void set_bit_range(Data& p0, Data& p1, Data val, shift_t bit, shift_t bits)
{
    if (JML_UNLIKELY(bits == 0)) return;

    /* There's some part of the first and some part of the second
       value (both zero or more bits) that need to be replaced by the
       low and high bits respectively. */

    enum { DBITS = sizeof(Data) * 8 };

    shift_t bits0 = std::min<shift_t>(bits, DBITS - bit);
    shift_t bits1 = bits - bits0;

#if 0
    using namespace std;
    cerr << "bits0 = " << bits0 << endl;
    cerr << "bits1 = " << bits1 << endl;
    cerr << "val = " << val << " bit = " << (int)bit << " bits = " << (int)bits
         << endl;
#endif

    p0 = set_bits<Data>(p0, val, bit, bits0);
    if (bits1) p1 = set_bits<Data>(p1, val >> bits0, 0, bits1);
}



/*****************************************************************************/
/* UTILITY FUNCTIONS                                                         */
/*****************************************************************************/

/** Sign extend the sign from the given bit into the rest of the sign bits
    of the given type. */
template<typename T>
T sign_extend(T raw, int sign_bit)
{
    using namespace std;
    T sign = (raw & ((T)1 << sign_bit)) != 0;
    T new_bits = T(-sign) << sign_bit;
    return raw | new_bits;
}

// TODO: ix86 can be optimized by doing a shl and then a sar

template<typename T>
T fixup_extract(T extracted, shift_t bits)
{
    return extracted;
}

JML_ALWAYS_INLINE signed char fixup_extract(signed char e, shift_t bits)
{
    return sign_extend(e, bits);
}

JML_ALWAYS_INLINE signed short fixup_extract(signed short e, shift_t bits)
{
    return sign_extend(e, bits);
} 

JML_ALWAYS_INLINE signed int fixup_extract(signed int e, shift_t bits)
{
    return sign_extend(e, bits);
} 

JML_ALWAYS_INLINE signed long fixup_extract(signed long e, shift_t bits)
{
    return sign_extend(e, bits);
} 

JML_ALWAYS_INLINE signed long long fixup_extract(signed long long e, shift_t bits)
{
    return sign_extend(e, bits);
}

/*****************************************************************************/
/* MEM_BUFFER                                                                */
/*****************************************************************************/

/** This is a buffer that handles sequential access to memory.  It ensures
    that there are always N words available, and can skip.  Memory is read
    in such a way that memory is always read as few times as possible and
    with a whole buffer at once.
*/

template<typename Data>
struct Simple_Mem_Buffer {
    Simple_Mem_Buffer(const Data * data)
        : data(data)
    {
    }

    JML_ALWAYS_INLINE Data curr() const { return data[0]; }
    JML_ALWAYS_INLINE Data next() const { return data[1]; }
    
    void operator += (int offset) { data += offset; }

    //private:
    const Data * data;  // always aligned to 2 * alignof(Data)
};

template<typename Data>
struct Buffered_Mem_Buffer {
    Buffered_Mem_Buffer()
        : data(0), b0(0), b1(0)
    {
    }
    
    Buffered_Mem_Buffer(const Data * data)
        : data(data)
    {
        b0 = data[0];
        b1 = data[1];
    }

    JML_ALWAYS_INLINE Data curr() const { return b0; }
    JML_ALWAYS_INLINE Data next() const { return b1; }
    
    void operator += (int offset)
    {
        data += offset;
        b0 = (offset == 0 ? b0 : (offset == 1 ? b1 : data[0]));
        b1 = (offset == 0 ? b1 : data[1]);
    }

    void operator ++ (int)
    {
        b0 = b1;
        b1 = data[1];
    }

    //private:
    const Data * data;  // always aligned to 2 * alignof(Data)
    Data b0, b1;
};


/*****************************************************************************/
/* BIT_BUFFER                                                                */
/*****************************************************************************/

template<typename Data, class MemBuf = Simple_Mem_Buffer<Data> >
struct Bit_Buffer {
    Bit_Buffer()
        : bit_ofs(0)
    {
    }

    Bit_Buffer(const Data * data)
        : data(data), bit_ofs(0)
    {
    }

    /// Extracts bits starting from the least-significant bits of the buffer.
    Data extract(shift_t bits)
    {
        if (JML_UNLIKELY(bits <= 0)) return Data(0);

        Data result;
        if (bit_ofs + bits <= 8 * sizeof(Data))
            // TODO: simplify
            result = extract_bit_range(data.curr(), Data(0), bit_ofs, bits);
        else
            result
                = extract_bit_range(data.curr(), data.next(), bit_ofs, bits);
        advance(bits);
        return result;
    }

    /** Extracts bits starting from the least-significant bits of the buffer.
        The caller guarantees that:
        1.  The word after the last word in the buffer can be safely read;
        2.  Bits is not zero.

        This allows further optimizations to be made.
    */
    Data extractFast(shift_t bits)
    {
        Data result = extract_bit_range(data.curr(), data.next(), bit_ofs, bits);
        advance(bits);
        return result;
    }

    /// Extracts bits starting from the most-significant bits of the buffer.
    Data rextract(shift_t bits)
    {
        if (JML_UNLIKELY(bits == 0)) return Data(0);

        enum { DBITS = 8 * sizeof(Data) };

        Data result;
        if (bit_ofs + bits <= DBITS) {
            shift_t shift = DBITS - (bit_ofs + bits);
            result = extract_bit_range(data.curr(), Data(0), shift, bits);
        }
        else {
            shift_t shift = (DBITS * 2) - (bit_ofs + bits);
            result = extract_bit_range(
                    data.next(), data.curr(), shift, bits);
        }

        advance(bits);

        return result;
    }

    /// Increase the cursor in the bit buffer. For performance reasons, no
    /// bound checking is performed and it is up to the caller to ensure that
    /// the sum of the current offset and the "bits" parameter is within
    /// [0, sizeof(buffer)[.
    void advance(ssize_t bits)
    {
        bit_ofs += bits;
        data += (bit_ofs / (sizeof(Data) * 8));
        bit_ofs %= sizeof(Data) * 8;
    }

    size_t current_offset(const Data * start)
    {
        return (data.data - start) * sizeof(Data) * 8 + bit_ofs;
    }

private:
    MemBuf data;
    size_t bit_ofs;     // number of bits from start
};


/*****************************************************************************/
/* BIT_EXTRACTOR                                                             */
/*****************************************************************************/

/** A class to perform extraction of bits from a range of memory.  The goal
    is to support the streaming case efficiently (where we stream across a
    lot of memory, extracting different numbers of bits all the time).

    This class will:
    - Read memory in as large a chunks as possible, with each chunk aligned
      properly (in order to minimise the memory bandwidth used);
    - Perform the minimum number of memory reads necessary;
    - Keep enough state information around to allow multiple extractions to
      be performed efficiently.

    Note that it is not designed to be used where there is lots of random
    access to the fields.  For that, use set_bit_range and extract_bit_range
    instead.

    Extraction is done as integers, respecting the endianness of the current
    machine.  Signed integers will be automatically sign extended, but other
    manipulations will need to be done manually (or fixup_extract specialized).
*/

template<typename Data, typename Buffer = Bit_Buffer<Data> >
struct Bit_Extractor {

    JML_COMPUTE_METHOD
    Bit_Extractor()
        : bit_ofs(0)
    {
    }

    JML_COMPUTE_METHOD
    Bit_Extractor(const Data * data)
        : buf(data), bit_ofs(0)
    {
    }

    template<typename T>
    JML_COMPUTE_METHOD
    T extract(int num_bits)
    {
        return buf.extract(num_bits);
    }

    template<typename T>
    JML_COMPUTE_METHOD
    T extractFast(int num_bits)
    {
        return buf.extractFast(num_bits);
    }

    template<typename T>
    JML_COMPUTE_METHOD
    void extract(T & where, int num_bits)
    {
        where = buf.extract(num_bits);
    }

    template<typename T>
    JML_COMPUTE_METHOD
    void extractFast(T & where, int num_bits)
    {
        where = buf.extractFast(num_bits);
    }

    template<typename T1, typename T2>
    JML_COMPUTE_METHOD
    void extract(T1 & where1, int num_bits1,
                 T2 & where2, int num_bits2)
    {
        where1 = buf.extract(num_bits1);
        where2 = buf.extract(num_bits2);
    }

    template<typename T1, typename T2, typename T3>
    JML_COMPUTE_METHOD
    void extract(T1 & where1, int num_bits1,
                 T2 & where2, int num_bits2,
                 T3 & where3, int num_bits3)
    {
        where1 = buf.extract(num_bits1);
        where2 = buf.extract(num_bits2);
        where3 = buf.extract(num_bits3);
    }

    template<typename T1, typename T2, typename T3, typename T4>
    JML_COMPUTE_METHOD
    void extract(T1 & where1, shift_t num_bits1,
                 T2 & where2, shift_t num_bits2,
                 T3 & where3, shift_t num_bits3,
                 T4 & where4, shift_t num_bits4)
    {
        where1 = buf.extract(num_bits1);
        where2 = buf.extract(num_bits2);
        where3 = buf.extract(num_bits3);
        where4 = buf.extract(num_bits4);
    }

    JML_COMPUTE_METHOD
    void advance(ssize_t bits)
    {
        buf.advance(bits);
    }

    template<typename T, class OutputIterator>
    OutputIterator extract(shift_t num_bits, size_t num_objects,
                           OutputIterator where);

    size_t current_offset(const Data * start)
    {
        return buf.current_offset(start);
    }

private:
    Buffer buf;
    size_t bit_ofs;
};


/*****************************************************************************/
/* BIT_WRITER                                                                */
/*****************************************************************************/

template<class Data>
struct Bit_Writer {
    Bit_Writer(Data * data)
        : data(data), bit_ofs(0)
    {
    }

    /// Writes bits starting from the least-significant bits of the buffer.
    void write(Data val, shift_t bits)
    {
        if (JML_UNLIKELY(bits == 0)) return;

        //using namespace std;
        //cerr << "write: val = " << val << " bits = " << bits << endl;
        //cerr << "data[0] = " << data[0] << " data[1] = "
        //     << data[1] << endl;
        set_bit_range(data[0], data[1], val, bit_ofs, bits);
        //cerr << "after: data[0] = " << data[0] << " data[1] = "
        //     << data[1] << endl;

        //Data readBack = extract_bit_range(data, bit_ofs, bits);
        //if (readBack != val) {
        //    cerr << "val = " << val << " readBack = "
        //         << readBack << endl;
        //    cerr << "bit_ofs = " << bit_ofs << endl;
        //    cerr << "bits = " << bits << endl;
        //    throw Exception("didn't read back what was written");
        //}

        skip(bits);
    }

    /// Writes bits starting from the most-significant bits of the buffer.
    void rwrite(Data val, shift_t bits)
    {
        if (JML_UNLIKELY(bits == 0)) return;

        enum { DBITS = sizeof(Data) * 8 };

        if (bits + bit_ofs <= DBITS) {
            shift_t shift = DBITS - (bit_ofs + bits);
            data[0] = set_bits(data[0], val, shift, bits);
        }
        else {
            shift_t shift = (DBITS * 2) - (bit_ofs + bits);
            set_bit_range(data[1], data[0], val, shift, bits);
        }

        skip(bits);
    }

    void skip(shift_t bits) {
        bit_ofs += bits;
        data += (bit_ofs / (sizeof(Data) * 8));
        bit_ofs %= sizeof(Data) * 8;
    }

    size_t current_offset(Data * start)
    {
        return (data - start) * sizeof(Data) * 8 + bit_ofs;
    }

private:
    Data * data;
    size_t bit_ofs;
};

template<typename Value, typename Array = Value>
struct BitArrayIterator
    : public boost::iterator_facade<BitArrayIterator<Value, Array>,
                                    Value,
                                    boost::random_access_traversal_tag,
                                    Value> {
    BitArrayIterator(const Array * data = 0, int numBits = -1, int index = -1)
        : data(data), numBits(numBits), index(index)
    {
    }

    const Array * data;
    int numBits;
    int index;

    Value dereference() const
    {
        ML::Bit_Extractor<Array> extractor(data);
        extractor.advance(index * numBits);
        return extractor.template extract<Value>(numBits);
    }

    bool equal(const BitArrayIterator & other) const
    {
        return data == other.data && index == other.index;
    }

    void increment()
    {
        ++index;
    }

    void decrement()
    {
        --index;
    }

    void advance(int n)
    {
        index += n;
    }

    ssize_t distance_to(const BitArrayIterator & other) const
    {
        return other.index - index;
    }
};


} // namespace ML


#endif /* __arch__bit_range_ops_h__ */
