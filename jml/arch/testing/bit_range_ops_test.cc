/* bit_range_ops_test.cc
   Jeremy Barnes, 2 April February 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

   Test of the bit range operations.
*/

#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <iostream>

#include "jml/arch/bit_range_ops.h"
#include "jml/arch/demangle.h"
#include "jml/arch/tick_counter.h"

#include <boost/test/unit_test.hpp>
#include <boost/test/auto_unit_test.hpp>
#include <vector>


using namespace ML;
using namespace std;

using boost::unit_test::test_suite;

/** Check that the sign extension works properly */

BOOST_AUTO_TEST_CASE( test_sign_extend )
{
    BOOST_CHECK_EQUAL(sign_extend<int>(1, 0), -1);
    BOOST_CHECK_EQUAL(sign_extend<int>(1, 1),  1);
    BOOST_CHECK_EQUAL(sign_extend<int>(2, 1), -2);
    BOOST_CHECK_EQUAL(sign_extend<int>(3, 1), -1);

    BOOST_CHECK_EQUAL((int)sign_extend<signed char>(1, 0), -1);
    BOOST_CHECK_EQUAL((int)sign_extend<signed char>(1, 1),  1);
    BOOST_CHECK_EQUAL((int)sign_extend<signed char>(2, 1), -2);
    BOOST_CHECK_EQUAL((int)sign_extend<signed char>(3, 1), -1);
    BOOST_CHECK_EQUAL((int)sign_extend<signed char>(-1, 7), -1);
    BOOST_CHECK_EQUAL((int)sign_extend<signed char>(127, 7), 127);
    BOOST_CHECK_EQUAL((int)sign_extend<signed char>(127, 6), -1);
    BOOST_CHECK_EQUAL((int)sign_extend<signed char>(64, 6), -64);

    BOOST_CHECK_EQUAL(sign_extend<int64_t>(1, 0), -1);
    BOOST_CHECK_EQUAL(sign_extend<int64_t>(1, 1),  1);
    BOOST_CHECK_EQUAL(sign_extend<int64_t>(2, 1), -2);
    BOOST_CHECK_EQUAL(sign_extend<int64_t>(3, 1), -1);
    BOOST_CHECK_EQUAL(sign_extend<int64_t>(-1, 63), -1);
    BOOST_CHECK_EQUAL(sign_extend<int64_t>(0, 63), 0);
    BOOST_CHECK_EQUAL(sign_extend<int64_t>(27, 63), 27);
    BOOST_CHECK_EQUAL(sign_extend<int64_t>((1ULL << 63) - 1, 63),
                      (1ULL << 63) - 1);
}

template<class T>
JML_ALWAYS_INLINE
T do_shrd(T low, T high, int bits)
{
    T val1 = shrd_emulated(low, high, bits);
    T val2 = shrd(low, high, bits);

    if (val1 != val2) {
        cerr << "size: " << sizeof(T) << endl;
	cerr << "type: " << demangle(typeid(T).name()) << endl;
        cerr << "val1 = " << val1 << endl;
        cerr << "val2 = " << val2 << endl;
        cerr << "low  = " << low << endl;
        cerr << "high = " << high << endl;
        cerr << "bits = " << bits << endl;
    }

    BOOST_CHECK_EQUAL(val1, val2);

    return val1;
}

template<class T>
void do_test_shrd()
{
    cerr << "testing shrd for " << demangle(typeid(T).name()) << endl;

    enum { TBITS = sizeof(T) * 8 };

    const T TBITSALL = (T)-1;
    const T TBITSTOP = (T)1 << (TBITS - 1);

    BOOST_CHECK_EQUAL(do_shrd<T>(0, 0, 0), 0);
    BOOST_CHECK_EQUAL(do_shrd<T>(0, 0, 1), 0);
    BOOST_CHECK_EQUAL(do_shrd<T>(0, 0, TBITS - 1), 0);

    // Result is undefined; don't check
    //BOOST_CHECK_EQUAL(do_shrd<T>(0, 0, TBITS), 0);
    //BOOST_CHECK_EQUAL(do_shrd<T>(0, 0, TBITS * 2), 0);

    BOOST_CHECK_EQUAL(do_shrd<T>(1, 0, 0), 1);
    BOOST_CHECK_EQUAL(do_shrd<T>(1, 0, 1), 0);
    BOOST_CHECK_EQUAL(do_shrd<T>(1, 0, TBITS - 1), 0);


    // Result is undefined; don't check
    //BOOST_CHECK_EQUAL(do_shrd<T>(1, 0, TBITS), 0);
    //BOOST_CHECK_EQUAL(do_shrd<T>(1, 0, TBITS * 2), 0);

    BOOST_CHECK_EQUAL(do_shrd<T>(1, 1, 0), 1);
    BOOST_CHECK_EQUAL(do_shrd<T>(1, 1, 1), TBITSTOP);
    BOOST_CHECK_EQUAL(do_shrd<T>(1, 1, TBITS - 1), 2);

    BOOST_CHECK_EQUAL(do_shrd<T>(TBITSALL, TBITSALL, 0), TBITSALL);
    BOOST_CHECK_EQUAL(do_shrd<T>(TBITSALL, TBITSALL, 1), TBITSALL);
    BOOST_CHECK_EQUAL(do_shrd<T>(TBITSALL, TBITSALL, TBITS - 1), TBITSALL);

    BOOST_CHECK_EQUAL(do_shrd<T>(TBITSALL, 0, 0), TBITSALL);
    BOOST_CHECK_EQUAL(do_shrd<T>(TBITSALL, 0, 1), TBITSALL - TBITSTOP);
    BOOST_CHECK_EQUAL(do_shrd<T>(TBITSALL, 0, TBITS - 1), 1);

    BOOST_CHECK_EQUAL(do_shrd<T>(0, TBITSALL, 0), 0);
    BOOST_CHECK_EQUAL(do_shrd<T>(0, TBITSALL, 1), TBITSTOP);
    BOOST_CHECK_EQUAL(do_shrd<T>(0, TBITSALL, TBITS - 1), TBITSALL - 1);

    // Result is undefined; don't check
    //BOOST_CHECK_EQUAL(do_shrd<T>(1, 1, TBITS), 1);
    //BOOST_CHECK_EQUAL(do_shrd<T>(1, 1, TBITS * 2), 0);
}

BOOST_AUTO_TEST_CASE( test_shrd )
{
    do_test_shrd<uint8_t>();
    do_test_shrd<uint16_t>();
    do_test_shrd<uint32_t>();
    do_test_shrd<uint64_t>();
}

template<class T>
void do_test_ebr()
{
    cerr << "testing extract_bit_range for" << demangle(typeid(T).name())
         << endl;

    enum { TBITS = sizeof(T) * 8 };
    
    const T TBITSALL = (T)-1;
    //const T TBITSTOP = (T)1 << (TBITS - 1);

    T values[2];

    values[0] = values[1] = 0;

    BOOST_CHECK_EQUAL(extract_bit_range(values, 0, 0), 0);
    BOOST_CHECK_EQUAL(extract_bit_range(values, 0, TBITS), 0);
    BOOST_CHECK_EQUAL(extract_bit_range(values, TBITS-1, TBITS), 0);

    values[0] = 1;
    BOOST_CHECK_EQUAL(extract_bit_range(values, 0, 0), 0);
    BOOST_CHECK_EQUAL(extract_bit_range(values, 0, TBITS), 1);
    BOOST_CHECK_EQUAL(extract_bit_range(values, TBITS-1, TBITS), 0);

    values[1] = 1;
    BOOST_CHECK_EQUAL(extract_bit_range(values, 0, 0), 0);
    BOOST_CHECK_EQUAL(extract_bit_range(values, 0, TBITS), 1);
    BOOST_CHECK_EQUAL(extract_bit_range(values, TBITS-1, TBITS), 2);
    
    values[0] = values[1] = TBITSALL;
    BOOST_CHECK_EQUAL(extract_bit_range(values, 0, 0), 0);
    for (unsigned i = 0;  i < TBITS;  ++i) {
        for (unsigned j = 0;  j < TBITS;  ++j) {
            T expected = (1ULL << j) - 1;
            T result = extract_bit_range(values, i, j);
            if (result != expected)
                cerr << "result = " << result << " expected = " << expected
                     << " i = " << i << " j = " << j << endl;

            BOOST_CHECK_EQUAL(result, expected);
        }
    }
}

BOOST_AUTO_TEST_CASE( test_extract_bit_range )
{
    do_test_ebr<uint8_t>();
    do_test_ebr<uint16_t>();
    do_test_ebr<uint32_t>();
    do_test_ebr<uint64_t>();
}


template<class T>
void test_set_extract(T * data, T value, int bit, int bits)
{
    //enum { TBITS = sizeof(T) * 8 };
    
    //const T TBITSALL = (T)-1;

    T old_data[2] = { data[0], data[1] };
    
    T old_value = extract_bit_range(data, bit, bits);

    set_bit_range(data[0], data[1], value, bit, bits);

    T new_value = extract_bit_range(data, bit, bits);

    BOOST_CHECK_EQUAL(value, new_value);

    set_bit_range(data[0], data[1], old_value, bit, bits);
    
    BOOST_CHECK_EQUAL(data[0], old_data[0]);
    BOOST_CHECK_EQUAL(data[1], old_data[1]);
    
    set_bit_range(data[0], data[1], value, bit, bits);

    T new_value2 = extract_bit_range(data, bit, bits);

    BOOST_CHECK_EQUAL(value, new_value2);
}


template<class T>
void do_test_sbr()
{
    cerr << "testing set_bit_range for " << demangle(typeid(T).name()) << endl;

    enum { TBITS = sizeof(T) * 8 };
    
    const T TBITSALL = (T)-1;
    //const T TBITSTOP = (T)1 << (TBITS - 1);

    T values[2];

    values[0] = values[1] = 0;

    test_set_extract<T>(values, 0, 0, 0);
    test_set_extract<T>(values, 1, 0, 1);
    test_set_extract<T>(values, 1, 0, 1);

    for (unsigned i = 0;  i < TBITS;  ++i) {
        for (unsigned j = 0;  j < TBITS;  ++j) {
            values[0] = values[1] = TBITSALL;
            test_set_extract<T>(values, rand() % (1ULL << j), i, j);
        }
    }

    for (unsigned i = 0;  i < TBITS;  ++i) {
        for (unsigned j = 0;  j < TBITS;  ++j) {
            values[0] = values[1] = 0;
            test_set_extract<T>(values, rand() % (1ULL << j), i, j);
        }
    }

    for (unsigned i = 0;  i < TBITS;  ++i) {
        for (unsigned j = 0;  j < TBITS;  ++j) {
            test_set_extract<T>(values, rand() % (1ULL << j), i, j);
        }
    }
}

BOOST_AUTO_TEST_CASE( test_set_bit_range )
{
    do_test_sbr<uint8_t>();
    do_test_sbr<uint16_t>();
    do_test_sbr<uint32_t>();
    do_test_sbr<uint64_t>();
}

BOOST_AUTO_TEST_CASE( test_64_bit_set_extract )
{
    {
        uint64_t data[2] = { 0, 0 };
        
        Bit_Writer<uint64_t> writer(data);
        writer.write(1ULL << 63, 64);
        writer.write(-1ULL, 64);
        
        BOOST_CHECK_EQUAL(data[0], 1ULL << 63);
        BOOST_CHECK_EQUAL(data[1], -1);

        Bit_Extractor<uint64_t> extractor(data);
        BOOST_CHECK_EQUAL(extractor.extract<uint64_t>(64), 1ULL << 63);
        BOOST_CHECK_EQUAL(extractor.extract<uint64_t>(64), -1);
    }

    {
        uint64_t data[3] = { 0, 0, 0 };
        
        Bit_Writer<uint64_t> writer(data);
        writer.write(1, 3);
        writer.write(1ULL << 63, 64);
        writer.write(-1ULL, 64);
        
        Bit_Extractor<uint64_t> extractor(data);
        BOOST_CHECK_EQUAL(extractor.extract<uint64_t>(3), 1);
        BOOST_CHECK_EQUAL(extractor.extract<uint64_t>(64), 1ULL << 63);
        BOOST_CHECK_EQUAL(extractor.extract<uint64_t>(64), -1);
    }
}

BOOST_AUTO_TEST_CASE( test_64_bit_set_rextract )
{
    {
        uint64_t data[2] = { 0, 0 };

        Bit_Writer<uint64_t> writer(data);
        writer.rwrite(-1ULL, 64);
        writer.rwrite(1ULL << 63, 64);

        BOOST_CHECK_EQUAL(data[0], -1);
        BOOST_CHECK_EQUAL(data[1], 1ULL << 63);

        Bit_Buffer<uint64_t> extractor(data);
        BOOST_CHECK_EQUAL(extractor.rextract(64), -1);
        BOOST_CHECK_EQUAL(extractor.rextract(64), 1ULL << 63);
    }

    {
        uint64_t data[3] = { 0, 0, 0 };

        Bit_Writer<uint64_t> writer(data);
        writer.rwrite(1, 3);
        writer.rwrite(1ULL << 63, 64);
        writer.rwrite(-1ULL, 64);

        Bit_Buffer<uint64_t> extractor(data);
        BOOST_CHECK_EQUAL(extractor.rextract(3), 1);
        BOOST_CHECK_EQUAL(extractor.rextract(64), 1ULL << 63);
        BOOST_CHECK_EQUAL(extractor.rextract(64), -1);
    }
}

BOOST_AUTO_TEST_CASE( testMaskLower )
{
    {
        BOOST_CHECK_EQUAL(maskLower<uint64_t>(-1, 0), 0);
        BOOST_CHECK_EQUAL(maskLower<uint64_t>(-1, 1), 1);
        BOOST_CHECK_EQUAL(maskLower<uint64_t>(-1, 64), -1);

        BOOST_CHECK_EQUAL(maskLower<int64_t>(-1, 0), 0);
        BOOST_CHECK_EQUAL(maskLower<int64_t>(-1, 1), 1);
        BOOST_CHECK_EQUAL(maskLower<int64_t>(-1, 64), -1);

        BOOST_CHECK_EQUAL(maskLower<uint64_t>(0, 0), 0);
        BOOST_CHECK_EQUAL(maskLower<uint64_t>(0, 1), 0);
        BOOST_CHECK_EQUAL(maskLower<uint64_t>(0, 64), 0);

        BOOST_CHECK_EQUAL(maskLower<int64_t>(0, 0), 0);
        BOOST_CHECK_EQUAL(maskLower<int64_t>(0, 1), 0);
        BOOST_CHECK_EQUAL(maskLower<int64_t>(0, 64), 0);
    }
}

// Test skip, by ensuring the value returned by current_offset matches the
// offset passed as param.
BOOST_AUTO_TEST_CASE( test_skip )
{
    /* "buffer" does not need to be a huge buffer for this test, as no
       write/read is actually performed from it */
    char buffer[] = { '\0' };

    ML::Bit_Writer<char> writer(buffer);

    /* we ensure that values close to UINT_MAX are not converted to negative
       64 ints */
    writer.skip(4294967295);
    BOOST_CHECK_EQUAL(writer.current_offset(buffer), 4294967295);
}

// Test Bit_Buffer_advance.
BOOST_AUTO_TEST_CASE( test_Bit_Buffer_advance )
{
    /* "buffer" does not need to be a huge buffer for this test, as no
       write/read is actually performed to/from it */
    uint64_t data[] = {0};

    ML::Bit_Buffer<uint64_t> buffer(data);
    BOOST_CHECK_EQUAL(buffer.current_offset(data), 0);

    // simple forward
    buffer.advance(1);
    BOOST_CHECK_EQUAL(buffer.current_offset(data), 1);

    // simple backward
    buffer.advance(-1);
    BOOST_CHECK_EQUAL(buffer.current_offset(data), 0);

    // 0xffffffff is considered as a positive int
    buffer.advance(UINT_MAX);
    BOOST_CHECK_EQUAL(buffer.current_offset(data), UINT_MAX);

    // a positive int64_t is considered as a positive int
    buffer.advance(int64_t(UINT_MAX) + 1);
    BOOST_CHECK_EQUAL(buffer.current_offset(data), (1LL << 33) - 1);
}
