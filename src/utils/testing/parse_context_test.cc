/* parse_context_test.cc                                           -*- C++ -*-
   Jeremy Barnes, 16 February 2007
   Copyright (c) 2007 Jeremy Barnes.  All rights reserved.

   Test of tick counter functionality.
*/

#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include "utils/parse_context.h"
#include "utils/file_functions.h"
#include "utils/guard.h"
#include <boost/test/unit_test.hpp>
#include <boost/bind.hpp>
#include <sstream>
#include <fstream>

using namespace ML;
using namespace std;

using boost::unit_test::test_suite;

static const char * test1_str = "Here \t is a\tparse context\nwith two\ni mean 3 lines";

void run_test1(Parse_Context & context)
{
    BOOST_CHECK(!context.eof());
    BOOST_CHECK_EQUAL(context.get_offset(), 0);
    BOOST_REQUIRE(!context.eof());
    BOOST_CHECK(context.match_literal('H'));
    BOOST_REQUIRE(!context.eof());
    BOOST_CHECK(context.match_literal("ere"));
    BOOST_CHECK(context.match_whitespace());
    string word;
    BOOST_CHECK(context.match_text(word, ' '));
    BOOST_CHECK_EQUAL(word, "is");
    BOOST_CHECK(context.match_whitespace());
    BOOST_CHECK(!context.match_literal('A'));
    BOOST_CHECK(context.match_text(word, " \t"));
    BOOST_CHECK_EQUAL(word, "a");
    BOOST_CHECK_EQUAL(std::string(test1_str).find('p') - 1, context.get_offset());
    BOOST_CHECK(context.match_text(word, " \t"));
    BOOST_CHECK_EQUAL(word, "");
    BOOST_CHECK(context.match_text(word, " "));
    BOOST_CHECK_EQUAL(word, "\tparse");
    int col_before = context.get_col(), line_before = context.get_line(),
        ofs_before = context.get_offset();
    BOOST_CHECK(!context.match_literal("context"));
    BOOST_CHECK_EQUAL(*context, ' ');
    BOOST_CHECK_EQUAL(context.get_col(), col_before);
    BOOST_CHECK_EQUAL(context.get_line(), line_before);
    BOOST_CHECK_EQUAL(context.get_offset(), ofs_before);
    BOOST_CHECK(context.match_whitespace());
    BOOST_CHECK(context.match_literal("context"));
    BOOST_CHECK_EQUAL(context.get_line(), 1);
    BOOST_CHECK_EQUAL(context.get_col(), 26);
    BOOST_CHECK(context.match_eol());
    BOOST_CHECK_EQUAL(context.get_line(), 2);
    BOOST_CHECK_EQUAL(context.get_col(), 1);
    BOOST_CHECK(context.match_line(word));
    BOOST_CHECK(context.match_literal("i mean "));
    BOOST_CHECK_EQUAL(context.get_line(), 3);
    BOOST_CHECK_EQUAL(context.get_col(), 8);
    int i = -1;
    BOOST_CHECK(context.match_int(i));
    BOOST_CHECK_EQUAL(i, 3);
    BOOST_CHECK(!context.match_int(i));
    BOOST_CHECK(context.match_whitespace());
    BOOST_CHECK(context.match_text(word, "abcdfghjkmopqrstuvwxyz"));
    BOOST_CHECK_EQUAL(word, "line");
    BOOST_CHECK(context.match_literal('s'));
    BOOST_CHECK(context.match_eol());
    BOOST_CHECK(context.eof());
    BOOST_CHECK_THROW(*context, ML::Exception);
    BOOST_CHECK_EQUAL(context.get_offset(), strlen(test1_str));
}

BOOST_AUTO_TEST_CASE( test1 )
{
    {
        Parse_Context context("test",
                              test1_str, test1_str + strlen(test1_str));
        run_test1(context);
    }

    size_t sizes[] = { 1, 65530, 1, 2, 3, 5, 128, 0 };
    
    for (unsigned i = 0;  sizes[i];  ++i) {
        istringstream stream(test1_str);
        Parse_Context context("test", stream, 1, 1, sizes[i]);
        run_test1(context);
    }
}

/* Test opening a file */
BOOST_AUTO_TEST_CASE( test2 )
{
    string tmp_filename = "parse_context_test_file";
    Call_Guard guard;
    {
        ofstream stream(tmp_filename.c_str());
        guard.set(boost::bind(&delete_file, "parse_context_test_file"));
        stream << test1_str;
    }

    Parse_Context context(tmp_filename);

    run_test1(context);
}

std::string expect_feature_name(Parse_Context & c)
{
    std::string result;
    /* We have a backslash escaped name. */
    bool after_backslash = false;
    
    Parse_Context::Revert_Token tok(c);
    
    int len = 0;
    while (c && *c != '\n') {
        if (!after_backslash && (isspace(*c) || *c == ':' || *c == ','))
            break;
        if (*c == '\\') after_backslash = true;
        else { ++len;  after_backslash = false; }
        ++c;
    }
    
    result.reserve(len);
    tok.apply();
    
    if (after_backslash) c.exception("Invalid backslash escaping");
    after_backslash = false;
    while (c && *c != '\n') {
        if (!after_backslash && (isspace(*c) || *c == ':'
                                 || *c == ',')) break;
        if (*c == '\\') after_backslash = true;
        else { result += *c;  after_backslash = false; }
        ++c;
    }
    
    if (result.empty())
        c.exception("expect_feature_name(): no feature name found");
    
    return result;
}

/* Test parse bug */
BOOST_AUTO_TEST_CASE( test3 )
{
    string header = "LABEL X Y";
    Parse_Context context("test file", header.c_str(),
                          header.c_str() + header.size());

    BOOST_CHECK(context);
    context.skip_whitespace();
    BOOST_CHECK_EQUAL(expect_feature_name(context), "LABEL");
    context.skip_whitespace();
    BOOST_CHECK_EQUAL(expect_feature_name(context), "X");
    context.skip_whitespace();
    BOOST_CHECK(context);
    BOOST_CHECK_EQUAL(expect_feature_name(context), "Y");
    BOOST_CHECK(!context);
}

BOOST_AUTO_TEST_CASE( test4 )
{
    string file = "LABEL X Y\n1 0 0\n0 1 0\n0 0 1\n1 1 1\n";
    Parse_Context context("test file", file.c_str(),
                          file.c_str() + file.size());
    
    BOOST_CHECK(context);
    context.skip_line();
    BOOST_CHECK_EQUAL(context.get_offset(), 10);
    BOOST_CHECK_EQUAL(context.expect_text(" \n\t"), "1");
    BOOST_CHECK_EQUAL(context.get_offset(), 11);
    BOOST_CHECK(context.match_whitespace());
    BOOST_CHECK_EQUAL(context.get_offset(), 12);
    BOOST_CHECK_EQUAL(context.expect_text(" \n\t"), "0");
    BOOST_CHECK_EQUAL(context.get_offset(), 13);
    BOOST_CHECK(context.match_whitespace());
    BOOST_CHECK_EQUAL(context.get_offset(), 14);
    BOOST_CHECK_EQUAL(context.expect_text(" \n\t"), "0");
    BOOST_CHECK_EQUAL(context.get_offset(), 15);
    BOOST_CHECK_EQUAL((int)*context, (int)'\n');
    BOOST_CHECK(!context.match_whitespace());
    BOOST_CHECK(context.match_eol());
    BOOST_CHECK_EQUAL(context.get_offset(), 16);
    BOOST_CHECK_EQUAL(context.expect_text(" \n\t"), "0");
}

BOOST_AUTO_TEST_CASE( test5 )
{
    string file = "1.234e-05";
    Parse_Context context("test file", file.c_str(),
                          file.c_str() + file.size());

    float f = context.expect_float();
    float f2 = 1.234e-05;

    BOOST_CHECK_EQUAL(f, f2);
}

BOOST_AUTO_TEST_CASE( test6 )
{
    string file = "33 nan nan 1 nan nan 2 -nan +nan";
    Parse_Context context("test file", file.c_str(),
                          file.c_str() + file.size());

    float f = context.expect_float();
    float f2 = 33;
    context.expect_whitespace();
    
    BOOST_CHECK_EQUAL(f, f2);

    f = context.expect_float();
    cerr << "f = " << f << endl;
    BOOST_CHECK(isnanf(f));
    context.expect_whitespace();

    f = context.expect_float();
    BOOST_CHECK(isnanf(f));
    context.expect_whitespace();

    f = context.expect_float();
    BOOST_CHECK_EQUAL(f, 1.0f);
    context.expect_whitespace();

    f = context.expect_float();
    BOOST_CHECK(isnanf(f));
    context.expect_whitespace();

    f = context.expect_float();
    BOOST_CHECK(isnanf(f));
    context.expect_whitespace();

    f = context.expect_float();
    BOOST_CHECK_EQUAL(f, 2.0f);
}
