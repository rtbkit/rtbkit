/* dense_layer_test.cc
   Jeremy Barnes, 28 October 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

   Unit tests for the dense layer class.
*/


#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include "utils/parse_context.h"
#include "utils/file_functions.h"
#include "utils/guard.h"
#include "db/persistent.h"
#include "db/compact_size_types.h"
#include <boost/test/unit_test.hpp>
#include <boost/bind.hpp>
#include <sstream>
#include <boost/multi_array.hpp>
#include "algebra/matrix_ops.h"
#include "stats/distribution.h"
#include "boosting/thread_context.h"
#include "neural/layer.h"


using namespace ML;
using namespace ML::DB;
using namespace ML::Stats;
using namespace std;

using boost::unit_test::test_suite;

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
        for (unsigned i = 0;  i < s.size() && i < 1024;  i += 16) {
            cerr << format("%04x | ", i);
            for (unsigned j = i;  j < i + 16;  ++j) {
                if (j < s.size())
                    cerr << format("%02x ", (int)*(unsigned char *)(&s[j]));
                else cerr << "   ";
            }

            cerr << "| ";

            for (unsigned j = i;  j < i + 16;  ++j) {
                if (j < s.size()) {
                    if (s[j] >= ' ' && s[j] < 127)
                        cerr << s[j];
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

BOOST_AUTO_TEST_CASE( test_serialize_reconstitute_dense_layer )
{
    Thread_Context context;
    Dense_Layer<float> layer(200, 400, TF_TANH, context);
    test_serialize_reconstitute(layer);
}

BOOST_AUTO_TEST_CASE( test_serialize_reconstitute_dense_layer_double )
{
    Thread_Context context;
    Dense_Layer<double> layer(200, 400, TF_TANH, context);
    test_serialize_reconstitute(layer);
}

