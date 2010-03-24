/* parse_context_test.cc                                           -*- C++ -*-
   Jeremy Barnes, 16 February 2007
   Copyright (c) 2007 Jeremy Barnes.  All rights reserved.

   Test of tick counter functionality.
*/

#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include "jml/utils/parse_context.h"
#include "jml/utils/file_functions.h"
#include "jml/utils/guard.h"
#include "jml/utils/filter_streams.h"
#include "jml/utils/vector_utils.h"
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

BOOST_AUTO_TEST_CASE( test_big_chunk_size )
{
    size_t NCHARS = 1024 * 1024 * 16;

    string s(NCHARS, 0);

    BOOST_CHECK_EQUAL(s.size(), NCHARS);

    for (unsigned i = 0;  i < NCHARS;  ++i)
        s[i] = i % 256;
    
    istringstream stream(s);

    Parse_Context context("test file", stream);
    
    // 1G, can't possibly fit on stack
    context.set_chunk_size(1024 * 1024 * 1024);

    int n = 0;
    while (context) {
        if (s[n] != *context++) {
            cerr << "error at position " << n << endl;
            break;
        }
        ++n;
    }

    BOOST_CHECK_EQUAL(n, NCHARS);

    BOOST_CHECK_EQUAL(context.get_offset(), s.size());
}

BOOST_AUTO_TEST_CASE( test_chunking_stream1 )
{
    // Make some random records in a string
    string s = "33 nan nan 1 nan nan 2 -nan +nan";

    int chunk_sizes[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                          31, 61, 63, 1024 * 1024 * 16 };
    
    int nchunk_sizes = sizeof(chunk_sizes) / sizeof(chunk_sizes[0]);

    for (unsigned i = 0;  i < nchunk_sizes;  ++i) {
        istringstream stream(s);

        Parse_Context context("test file", stream, 1, 1, 1);
        
        context.set_chunk_size(chunk_sizes[i]);

        float f = context.expect_float();
        float f2 = 33;
        context.expect_whitespace();
        
        BOOST_CHECK_EQUAL(f, f2);
        
        f = context.expect_float();
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
        context.expect_whitespace();

        f = context.expect_float();
        BOOST_CHECK(isnanf(f));
        context.expect_whitespace();
        
        f = context.expect_float();
        BOOST_CHECK(isnanf(f));
        
        BOOST_CHECK(context.eof());
    }
}

std::string expect_csv_field(Parse_Context & context)
{
    bool quoted = false;
    std::string result;
    while (context) {
        //if (context.get_line() == 9723)
        //    cerr << "*context = " << *context << " quoted = " << quoted
        //         << " result = " << result << endl;
        
        if (quoted) {
            if (context.match_literal("\"\"")) {
                result += "\"";
                continue;
            }
            if (context.match_literal('\"')) {
                if (!context || context.match_literal(',')
                    || *context == '\n' || *context == '\r')
                    return result;

#if 0
                cerr << "(bool)context = " << (bool)context << endl;
                cerr << "*context = " << *context << endl;
                cerr << "result = " << result << endl;

                for (unsigned i = 0; i < 20;  ++i)
                    cerr << *context++;
#endif

                context.exception_fmt("invalid end of line: %d %c", (int)*context, *context);
            }
        }
        else {
            if (context.match_literal('\"')) {
                if (result == "") {
                    quoted = true;
                    continue;
                }
                else context.exception("non-quoted string with embedded quote");
            }
            else if (context.match_literal(','))
                return result;
            else if (*context == '\n')
                return result;
            
        }
        result += *context++;
    }

    if (quoted)
        context.exception("file finished inside quote");

    return result;
}

std::vector<std::string> expect_csv_row(Parse_Context & context)
{
    context.skip_whitespace();

    vector<string> result;

    while (context && !context.match_eol()) {
        result.push_back(expect_csv_field(context));
        //cerr << "read " << result.back() << endl;
    }

    return result;
}

void test_csv_data_size(int chunk_size, vector<vector<string> > & reference)
{
    string input_file = "utils/testing/parse_context_test_data.csv.gz";

    filter_istream stream(input_file);
    Parse_Context context(input_file, stream);

    context.set_chunk_size(chunk_size);

    bool is_reference = reference.empty();

    try {
        for (int i = 0;  context;  ++i) {
            vector<string> row = expect_csv_row(context);
            if (is_reference)
                reference.push_back(row);
            else {
                if (reference.at(i) != row) {
                    cerr << "error on row " << i << endl;
                    BOOST_CHECK_EQUAL(reference.at(i), row);
                }
            }
        }
    } catch (...) {
        if (is_reference) reference.clear();
        throw;
    }

    cerr << "succeeded in reading " << reference.size()
         << " rows with chunk_size " << chunk_size << endl;

    BOOST_CHECK_EQUAL(reference.size(), 10000);
}

BOOST_AUTO_TEST_CASE( test_csv_data )
{
    int chunk_sizes[] = { 1024 * 1024 * 16,
                          1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                          31, 61, 63, 65531 };
    
    int nchunk_sizes = sizeof(chunk_sizes) / sizeof(chunk_sizes[0]);
    
    vector<vector<string> > reference;

    for (unsigned i = 0;  i < nchunk_sizes;  ++i) {
        BOOST_CHECK_NO_THROW(test_csv_data_size(chunk_sizes[i], reference));
    }
}
