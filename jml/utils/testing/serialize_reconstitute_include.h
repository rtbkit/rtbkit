/* serialize_reconstitute_include.h                                -*- C++ -*-
   Jeremy Barnes, 5 November 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

   Include to test serialization and reconstitution.
*/

#ifndef __jml__utils__testing__serialize_reconstitute_include_h__
#define __jml__utils__testing__serialize_reconstitute_include_h__

#include "jml/db/persistent.h"
#include "jml/db/compact_size_types.h"
#include <boost/test/unit_test.hpp>
#include "jml/arch/demangle.h"

using namespace ML;
using namespace ML::DB;
using namespace std;

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

template<typename Base, typename X>
void test_poly_serialize_reconstitute(const X & x)
{
    ostringstream stream_out;

    {
        DB::Store_Writer writer(stream_out);
        x.poly_serialize(writer);
        writer << std::string("END");
    }

    istringstream stream_in(stream_out.str());
    
    DB::Store_Reader reader(stream_in);
    std::shared_ptr<Base> y;
    std::string s;

    try {
        y = Base::poly_reconstitute(reader);
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

    BOOST_REQUIRE(y != 0);
    BOOST_REQUIRE_EQUAL(demangle(typeid(*y).name()),
                        demangle(typeid(X).name()));

    const X & y2 = dynamic_cast<const X &>(*y);

    BOOST_CHECK_EQUAL(x, y2);
    BOOST_CHECK_EQUAL(s, "END");
}




#endif /* __jml__utils__testing__serialize_reconstitute_include_h__ */
