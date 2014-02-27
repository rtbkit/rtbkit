/* fixed_array_test.cc
   Jeremy Barnes, 8 February 2005
   Copyright (c) 2005 Jeremy Barnes.  All rights reserved.
   
   Test of the fixed array class.
*/

#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include "jml/utils/csv.h"
#include "jml/utils/vector_utils.h"
#include "jml/utils/parse_context.h"

#include <sstream>
#include <fstream>

using namespace std;
using namespace ML;

using boost::unit_test::test_suite;

vector<string> parseCsvLine(const std::string & line)
{
    ML::Parse_Context context(line, line.c_str(), line.c_str() + line.size());
    return expect_csv_row(context);
}

void testCsvLine(const std::string & line,
                 const std::vector<std::string> & values)
{
    vector<string> parsed = parseCsvLine(line);
    BOOST_CHECK_EQUAL(parsed, values);
}

BOOST_AUTO_TEST_CASE (test1)
{
    testCsvLine("", {});
    testCsvLine(",", {"",""});
    testCsvLine("\"\"", {""});
    testCsvLine("\"\",", {"",""});
    testCsvLine("\"\",\"\"", {"",""});
}
