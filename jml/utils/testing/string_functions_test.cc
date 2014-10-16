#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#define BOOST_AUTO_TEST_MAIN
#include <boost/test/auto_unit_test.hpp>

#include "jml/utils/string_functions.h"
#include "jml/arch/exception.h"

using namespace ML;

BOOST_AUTO_TEST_CASE( test_antoi )
{
    const char * c = "12";
    BOOST_CHECK(antoi(c, c + 2) == 12);

    const char * d  = "-1";
    BOOST_CHECK(antoi(d, d + 2) == -1);

    const char * p = "patate";
    BOOST_CHECK_THROW(antoi(p, p + 6), ML::Exception);
}
