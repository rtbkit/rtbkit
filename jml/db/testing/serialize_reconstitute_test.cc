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
#include <boost/test/unit_test.hpp>
#include <boost/bind.hpp>
#include <sstream>
#include <boost/multi_array.hpp>
#include "jml/algebra/matrix_ops.h"
#include "jml/stats/distribution.h"


using namespace ML;
using namespace ML::DB;
using namespace std;

using boost::unit_test::test_suite;

#if 0
namespace boost {
namespace test_tools {
namespace tt_detail {

predicate_result
equal_impl(const ML::Stats::distribution<float> & d1,
           const ML::Stats::distribution<float> & d2)
{
    return (d1.size() == d2.size()
            && (d1 == d2).all());
}

} // namespace tt_detail
} // namespace test_tools
} // namespace boost

#endif

template<typename X>
void test_serialize_reconstitute(const X & x)
{
    ostringstream stream_out;

    {
        DB::Store_Writer writer(stream_out);
        writer << x;
        writer << std::string("END");
    }

    istringstream stream_in(stream_out.str());
    
    DB::Store_Reader reader(stream_in);
    X y;
    std::string s;

    try {
        reader >> y;
        reader >> s;
    } catch (const std::exception & exc) {
        cerr << "serialized representation:" << endl;

        string s = stream_out.str();
        for (unsigned i = 0;  i < s.size();  i += 16) {
            cerr << format("%04x | ", i);
            for (unsigned j = i;  j < i + 16;  ++j) {
                if (j < s.size())
                    cerr << format("%02x ", (int)*(unsigned char *)(&s[j]));
                else cerr << "   ";
            }

            cerr << "| ";

            for (unsigned j = i;  j < i + 16;  ++j) {
                if (j < s.size()) {
                    if (s[j] >= ' ' && s[j] <= 127)
                        cerr << s[i];
                    else cerr << '.';
                }
                else cerr << " ";
            }
            cerr << endl;
        }

        throw;
    }

    BOOST_CHECK_EQUAL(x, y);
    BOOST_CHECK_EQUAL(s, "END");
}

BOOST_AUTO_TEST_CASE( test_char )
{
    test_serialize_reconstitute('a');
}

BOOST_AUTO_TEST_CASE( test_multi_array )
{
    boost::multi_array<float, 2> A(boost::extents[3][3]);
    for (unsigned i = 0;  i < 3;  ++i)
        for (unsigned j = 0;  j < 3;  ++j)
            A[i][j] = j * j - i;

    test_serialize_reconstitute(A);
}

BOOST_AUTO_TEST_CASE( test_bool )
{
    ostringstream stream_out;
    {
        DB::Store_Writer writer(stream_out);
        writer << true;
    }

    BOOST_CHECK_EQUAL(stream_out.str().size(), 1);
    BOOST_CHECK_EQUAL(stream_out.str().at(0), 1);

    test_serialize_reconstitute(true);

    test_serialize_reconstitute(false);
}

BOOST_AUTO_TEST_CASE( test_distribution )
{
    distribution<float> dist;
    test_serialize_reconstitute(dist);

    dist.push_back(1.0);
    test_serialize_reconstitute(dist);

    dist.push_back(2.0);
    test_serialize_reconstitute(dist);
}
