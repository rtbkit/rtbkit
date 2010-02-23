/* info_test.cc
   Jeremy Barnes, 21 February 2007
   Copyright (c) 2007 Jeremy Barnes.  All rights reserved.

   Test for the info functions.
*/

#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include "jml/arch/vm.h"
#include "jml/arch/exception.h"
#include "jml/utils/vector_utils.h"

#include <boost/test/unit_test.hpp>
#include <iostream>


using namespace ML;
using namespace std;

using boost::unit_test::test_suite;

void test_function()
{
    cerr << "hello" << endl;
}

BOOST_AUTO_TEST_CASE( test_page_info )
{
    vector<Page_Info> null_pi = page_info(0, 1);

    BOOST_CHECK_EQUAL(null_pi.size(), 1);
    BOOST_CHECK_EQUAL(null_pi[0].pfn, 0);

    BOOST_CHECK_EQUAL(null_pi, page_info((void *)1, 1));
    
    vector<Page_Info> stack_pi = page_info(&null_pi, 1);

    BOOST_CHECK_EQUAL(stack_pi.size(), 1);

    cerr << "null_pi  = " << null_pi[0] << endl;
    cerr << "stack_pi = " << stack_pi[0] << endl;

    vector<Page_Info> code_pi = page_info((void *)&test_function, 1);
    
    cerr << "code_pi  = " << code_pi[0] << endl;
}
