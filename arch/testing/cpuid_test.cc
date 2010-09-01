/* cpuid_test.cc
   Jeremy Barnes, 21 February 2007
   Copyright (c) 2007 Jeremy Barnes.  All rights reserved.

   Test of the CPUID detection code.
*/

#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include "jml/arch/cpuid.h"

#include <boost/test/unit_test.hpp>
#include <vector>
#include <set>
#include <iostream>


using namespace ML;
using namespace std;

using boost::unit_test::test_suite;

BOOST_AUTO_TEST_CASE( test1 )
{
    BOOST_CHECK(cpuid_flags() != 0);
}

BOOST_AUTO_TEST_CASE( test2 )
{
    set<string> known;
    known.insert("GenuineIntel");
    known.insert("AuthenticAMD");

    string id = vendor_id();
    
    cerr << "vendor ID = " << id << endl;

    BOOST_CHECK(known.count(id));

    if (!known.count(id))
        cerr << "unknown vendor ID was \"" << id << "\"" << endl;
}

BOOST_AUTO_TEST_CASE( test3 )
{
    string id = model_id();
    cerr << "model ID = " << id << endl;
}

BOOST_AUTO_TEST_CASE( test4 )
{
    const CPU_Info & info = cpu_info();
    cerr << "cpuid level = " << info.cpuid_level << endl;
}
