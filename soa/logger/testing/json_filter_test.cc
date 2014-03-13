/* json_filter_test.cc
   Jeremy Barnes, 5 June 2011
   Copyright (c) 2011 Datacratic.  All rights reserved.

   Test for JSON filters.
*/

#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include "soa/logger/json_filter.h"
#include "jml/utils/guard.h"
#include "jml/arch/exception_handler.h"
#include "jml/arch/timers.h"
#include "jml/utils/filter_streams.h"
#include "jml/utils/smart_ptr_utils.h"

using namespace std;
using namespace ML;
using namespace Datacratic;

string test_data;

size_t max_bytes = 2000000;

size_t default_buffer_size = 16384;//1024 * 1024;

BOOST_AUTO_TEST_CASE( get_stream )
{
    cerr << "reading in test data" << endl;
    filter_istream input("/mnt/shared/testdata/rtb_router-auctions-2011-Jun-29-20:46:43.log.gz");

    ostringstream output;

    ML::Timer timer;

    char buffer[default_buffer_size];

    size_t bytes = 0;

    while (input && bytes < max_bytes) {
        input.read(buffer, default_buffer_size);
        int nread = input.gcount();
        output.write(buffer, nread);
        bytes += nread;
    };

    test_data = output.str();

    cerr << format("read in %.2fMB in %.2f s, rate = %.2fMB/sec",
                   test_data.length() / 1000000.0, timer.elapsed_wall(),
                   test_data.length() / 1000000.0 / timer.elapsed_wall())
         << endl;
}

void test_filters(Filter & compressor, Filter & decompressor,
                  const std::string & description,
                  size_t buffer_size)
{
    istringstream input(test_data);

    size_t in_bytes = 0, out_bytes = 0;

    ostringstream output;

    ML::Timer timer;

    compressor.onOutput = [&] (const char * buf, size_t n, FlushLevel flush,
                               boost::function<void ()> cb)
        {
            out_bytes += n;
            //decompressor.process(buf, buf + n, flush, cb);
            output.write(buf, n);
        };

    char buffer[buffer_size];

    while (input) {
        input.read(buffer, buffer_size);
        int nread = input.gcount();
        compressor.process(buffer, buffer + nread, FLUSH_NONE, []{});
    };

    compressor.process("", FLUSH_FINISH, [] {});

    double cmp_elapsed = timer.elapsed_wall();

    //cerr << "cmp_elapsed = " << cmp_elapsed << " out_bytes = "
    //     << out_bytes << endl;

    //cerr << "wrote " << output.str().size() << " bytes" << endl;

    size_t out_offset = 0;
    decompressor.onOutput = [&] (const char * buf, size_t n, FlushLevel flush,
                                 boost::function<void ()> cb)
        {
            in_bytes += n;
            
            if (!std::equal(buf, buf + n,
                            test_data.c_str() + out_offset)) {
                cerr << "in_bytes = " << in_bytes << endl;
                BOOST_REQUIRE(false);
            }

            out_offset += n;

            if (cb) cb();
        };

    timer.restart();

    istringstream input2(output.str());

    while (input2) {
        input2.read(buffer, buffer_size);
        int nread = input2.gcount();
        decompressor.process(buffer, buffer + nread, FLUSH_NONE, []{});
    };

    //decompressor.process("", FLUSH_FINISH, [] {});

    double dec_elapsed = timer.elapsed_wall();
    
    cerr << format("%-20s: %6.2fMB in, %6.2fMB out, ratio %6.2f%%, ctime %6.2fs, crate %8.2fMB/s, dtime %6.2fs, drate %8.2fMB/s",
                   description.c_str(),
                   in_bytes / 1000000.0, out_bytes / 1000000.0,
                   100.0 * out_bytes / in_bytes,
                   cmp_elapsed, in_bytes / 1000000.0 / cmp_elapsed,
                   dec_elapsed, in_bytes / 1000000.0 / dec_elapsed)
         << endl;
}

#if 1
BOOST_AUTO_TEST_CASE( test_gzip_filter )
{
    GzipCompressorFilter compressor;
    GzipDecompressor decompressor;

    size_t buffer_size = default_buffer_size;

    test_filters(compressor, decompressor, "gzip", buffer_size);
}

#endif


BOOST_AUTO_TEST_CASE( test_json_filter1 )
{
    JsonCompressor compressor;
    JsonDecompressor decompressor;

    size_t buffer_size = default_buffer_size;

    test_filters(compressor, decompressor, "Json", buffer_size);
}

#if 1

#if 1
BOOST_AUTO_TEST_CASE( test_bzip2_filter )
{
    Bzip2Compressor compressor;
    Bzip2Decompressor decompressor;

    size_t buffer_size = default_buffer_size;

    test_filters(compressor, decompressor, "bzip2", buffer_size);
}

BOOST_AUTO_TEST_CASE( test_json_plus_bzip2_filter )
{
    FilterStack compressor, decompressor;
    compressor.push(ML::make_std_sp(new JsonCompressor()));
    compressor.push(ML::make_std_sp(new Bzip2Compressor()));
    decompressor.push(ML::make_std_sp(new Bzip2Decompressor()));
    decompressor.push(ML::make_std_sp(new JsonDecompressor()));

    size_t buffer_size = default_buffer_size;

    test_filters(compressor, decompressor, "json+bzip2", buffer_size);
}
#endif

BOOST_AUTO_TEST_CASE( test_zlib_filter )
{
    ZlibCompressor compressor;
    ZlibDecompressor decompressor;

    size_t buffer_size = default_buffer_size;

    test_filters(compressor, decompressor, "zlib", buffer_size);
}

BOOST_AUTO_TEST_CASE( test_json_plus_zlib_filter )
{
    FilterStack compressor, decompressor;
    compressor.push(ML::make_std_sp(new JsonCompressor()));
    compressor.push(ML::make_std_sp(new ZlibCompressor()));
    decompressor.push(ML::make_std_sp(new ZlibDecompressor()));
    decompressor.push(ML::make_std_sp(new JsonDecompressor()));

    size_t buffer_size = default_buffer_size;

    test_filters(compressor, decompressor, "json+zlib", buffer_size);
}

BOOST_AUTO_TEST_CASE( test_lzma1_filter )
{
    LzmaCompressor compressor(1);
    LzmaDecompressor decompressor;

    size_t buffer_size = default_buffer_size;

    test_filters(compressor, decompressor, "lzma1", buffer_size);
}

BOOST_AUTO_TEST_CASE( test_json_plus_lzma1_filter )
{
    FilterStack compressor, decompressor;
    compressor.push(ML::make_std_sp(new JsonCompressor()));
    compressor.push(ML::make_std_sp(new LzmaCompressor(1)));
    decompressor.push(ML::make_std_sp(new LzmaDecompressor()));
    decompressor.push(ML::make_std_sp(new JsonDecompressor()));

    size_t buffer_size = default_buffer_size;

    test_filters(compressor, decompressor, "json+lzma1", buffer_size);
}

BOOST_AUTO_TEST_CASE( test_lzma2_filter )
{
    LzmaCompressor compressor(2);
    LzmaDecompressor decompressor;

    size_t buffer_size = default_buffer_size;

    test_filters(compressor, decompressor, "lzma2", buffer_size);
}

BOOST_AUTO_TEST_CASE( test_json_plus_lzma2_filter )
{
    FilterStack compressor, decompressor;
    compressor.push(ML::make_std_sp(new JsonCompressor()));
    compressor.push(ML::make_std_sp(new LzmaCompressor(2)));
    decompressor.push(ML::make_std_sp(new LzmaDecompressor()));
    decompressor.push(ML::make_std_sp(new JsonDecompressor()));

    size_t buffer_size = default_buffer_size;

    test_filters(compressor, decompressor, "json+lzma2", buffer_size);
}

BOOST_AUTO_TEST_CASE( test_lzma3_filter )
{
    LzmaCompressor compressor(3);
    LzmaDecompressor decompressor;

    size_t buffer_size = default_buffer_size;

    test_filters(compressor, decompressor, "lzma3", buffer_size);
}

BOOST_AUTO_TEST_CASE( test_json_plus_lzma3_filter )
{
    FilterStack compressor, decompressor;
    compressor.push(ML::make_std_sp(new JsonCompressor()));
    compressor.push(ML::make_std_sp(new LzmaCompressor(3)));
    decompressor.push(ML::make_std_sp(new LzmaDecompressor()));
    decompressor.push(ML::make_std_sp(new JsonDecompressor()));

    size_t buffer_size = default_buffer_size;

    test_filters(compressor, decompressor, "json+lzma3", buffer_size);
}

BOOST_AUTO_TEST_CASE( test_lzma4_filter )
{
    LzmaCompressor compressor(4);
    LzmaDecompressor decompressor;

    size_t buffer_size = default_buffer_size;

    test_filters(compressor, decompressor, "lzma4", buffer_size);
}

BOOST_AUTO_TEST_CASE( test_json_plus_lzma4_filter )
{
    FilterStack compressor, decompressor;
    compressor.push(ML::make_std_sp(new JsonCompressor()));
    compressor.push(ML::make_std_sp(new LzmaCompressor(4)));
    decompressor.push(ML::make_std_sp(new LzmaDecompressor()));
    decompressor.push(ML::make_std_sp(new JsonDecompressor()));

    size_t buffer_size = default_buffer_size;

    test_filters(compressor, decompressor, "json+lzma4", buffer_size);
}

BOOST_AUTO_TEST_CASE( test_lzma5_filter )
{
    LzmaCompressor compressor(5);
    LzmaDecompressor decompressor;

    size_t buffer_size = default_buffer_size;

    test_filters(compressor, decompressor, "lzma5", buffer_size);
}

BOOST_AUTO_TEST_CASE( test_json_plus_lzma5_filter )
{
    FilterStack compressor, decompressor;
    compressor.push(ML::make_std_sp(new JsonCompressor()));
    compressor.push(ML::make_std_sp(new LzmaCompressor(5)));
    decompressor.push(ML::make_std_sp(new LzmaDecompressor()));
    decompressor.push(ML::make_std_sp(new JsonDecompressor()));

    size_t buffer_size = default_buffer_size;

    test_filters(compressor, decompressor, "json+lzma5", buffer_size);
}

BOOST_AUTO_TEST_CASE( test_lzma6_filter )
{
    LzmaCompressor compressor(6);
    LzmaDecompressor decompressor;

    size_t buffer_size = default_buffer_size;

    test_filters(compressor, decompressor, "lzma6", buffer_size);
}

BOOST_AUTO_TEST_CASE( test_json_plus_lzma6_filter )
{
    FilterStack compressor, decompressor;
    compressor.push(ML::make_std_sp(new JsonCompressor()));
    compressor.push(ML::make_std_sp(new LzmaCompressor(6)));
    decompressor.push(ML::make_std_sp(new LzmaDecompressor()));
    decompressor.push(ML::make_std_sp(new JsonDecompressor()));

    size_t buffer_size = default_buffer_size;

    test_filters(compressor, decompressor, "json+lzma6", buffer_size);
}

BOOST_AUTO_TEST_CASE( test_lzma9_filter )
{
    LzmaCompressor compressor(9);
    LzmaDecompressor decompressor;

    size_t buffer_size = default_buffer_size;

    test_filters(compressor, decompressor, "lzma9", buffer_size);
}

BOOST_AUTO_TEST_CASE( test_json_plus_lzma9_filter )
{
    FilterStack compressor, decompressor;
    compressor.push(ML::make_std_sp(new JsonCompressor()));
    compressor.push(ML::make_std_sp(new LzmaCompressor(9)));
    decompressor.push(ML::make_std_sp(new LzmaDecompressor()));
    decompressor.push(ML::make_std_sp(new JsonDecompressor()));

    size_t buffer_size = default_buffer_size;

    test_filters(compressor, decompressor, "json+lzma9", buffer_size);
}
#endif
