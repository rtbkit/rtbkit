/* compact_size_type_test.cc
   Jeremy Barnes, 12 August 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

   Testing for the compact size type.
*/

#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include "jml/utils/parse_context.h"
#include "jml/utils/file_functions.h"
#include "jml/utils/guard.h"
#include "jml/db/persistent.h"
#include "jml/db/compact_size_types.h"
#include "jml/utils/hex_dump.h"
#include <boost/test/unit_test.hpp>
#include <boost/bind.hpp>
#include <sstream>

using namespace ML;
using namespace ML::DB;
using namespace std;

using boost::unit_test::test_suite;

void test_compact_size_type(uint64_t value)
{
    std::ostringstream os;

    {
        DB::Store_Writer store(os);
        compact_size_t cs(value);
        store << cs;
    }

    //cerr << "os.str().size() = " << os.str().size() << endl;

    std::istringstream is(os.str());
    DB::Store_Reader store(is);

    compact_size_t cs(store);

    if (cs.size_ != value) {
        cerr << "value:      ";
        hex_dump(&value, 8);
        cerr << "serialized: ";
        string s = os.str();
        hex_dump(s.c_str(), s.length());
    }

    BOOST_CHECK_EQUAL(cs.size_, value);
}

//void test_signed_compact_size_type(uint64_t value)
//{
//}

BOOST_AUTO_TEST_CASE( test1 )
{
    for (unsigned i = 0;  i < 64;  ++i) {
        unsigned long long val = 1ULL << i;

        //cerr << "testing i = " << i << endl;

        test_compact_size_type(val - 1);
        test_compact_size_type(val);
        test_compact_size_type(val + 1);
    }

    //cerr << "testing 64 bit values" << endl;
    test_compact_size_type(0xf000000000000000ULL);
    test_compact_size_type(0xf000000000000001ULL);
    test_compact_size_type(0xff00000000000000ULL);
    test_compact_size_type(0xfff0000000000000ULL);
    test_compact_size_type(0xffff000000000000ULL);
    test_compact_size_type(0xfffffffffffffffeULL);
    test_compact_size_type(0xffffffffffffffffULL);
}

void test_compact_int_type(int64_t value)
{
    std::ostringstream os;

    {
        DB::Store_Writer store(os);
        compact_int_t cs(value);
        store << cs;
    }

    //cerr << "os.str().int() = " << os.str().int() << endl;

    std::istringstream is(os.str());
    DB::Store_Reader store(is);

    compact_int_t cs(store);

    if (cs.size_ != value) {
        cerr << "value:      ";
        hex_dump(&value, 8);
        cerr << "serialized: ";
        string s = os.str();
        hex_dump(s.c_str(), s.length());
    }

    BOOST_CHECK_EQUAL(cs.size_, value);
}

BOOST_AUTO_TEST_CASE( test2 )
{
    test_compact_int_type(0);
    test_compact_int_type(1);
    test_compact_int_type(-1);

    for (unsigned i = 0;  i < 63;  ++i) {
        signed long long val = 1ULL << i;

        cerr << "testing i = " << i << endl;

        test_compact_int_type(val - 1);
        test_compact_int_type(val);
        test_compact_int_type(val + 1);

        test_compact_int_type(-val - 2);
        test_compact_int_type(-val - 1);
        test_compact_int_type(-val);
        test_compact_int_type(-val + 1);
        test_compact_int_type(-val + 2);
    }

    cerr << "testing 64 bit values" << endl;
    test_compact_int_type(0xf000000000000000ULL);
    test_compact_int_type(0xf000000000000001ULL);
    test_compact_int_type(0xff00000000000000ULL);
    test_compact_int_type(0xfff0000000000000ULL);
    test_compact_int_type(0xffff000000000000ULL);
    test_compact_int_type(0xfffffffffffffffeULL);
    test_compact_int_type(0xffffffffffffffffULL);
}

