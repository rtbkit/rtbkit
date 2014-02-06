/* filter_streams_test.cc
   Jeremy Barnes, 29 June 2011
   Copyright (c) 2011 Datacratic.
   Copyright (c) 2011 Jeremy Barnes.

*/

#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include "jml/utils/file_functions.h"
#include "jml/utils/filter_streams.h"
#include "jml/arch/exception.h"
#include "jml/arch/exception_handler.h"

#include <boost/filesystem.hpp>
#include <boost/test/unit_test.hpp>
#include <boost/thread.hpp>
#include <boost/thread/barrier.hpp>
#include <boost/bind.hpp>
#include <vector>
#include <stdint.h>
#include <iostream>
#include <fcntl.h>

#include "jml/utils/guard.h"
#include "jml/arch/exception_handler.h"
#include "jml/arch/demangle.h"

using namespace std;
namespace fs = boost::filesystem;
using namespace ML;

using boost::unit_test::test_suite;

struct FileCleanup {
    FileCleanup(const string & filename)
        : filename_(filename)
    {
    }

    ~FileCleanup()
    {
        ::unlink(filename_.c_str());
    }

    string filename_;
};


void system(const std::string & command)
{
    int res = ::system(command.c_str());
    if (res == -1)
        throw ML::Exception(errno, "system(): system");
    if (res != 0)
        throw ML::Exception("command %s returned code %d",
                            command.c_str(), res);
}

void compress_using_tool(const std::string & input_file,
                         const std::string & output_file,
                         const std::string & command)
{
    system("cat " + input_file + " | " + command + " > " + output_file);
}

void decompress_using_tool(const std::string & input_file,
                           const std::string & output_file,
                           const std::string & command)
{
    system("cat " + input_file + " | " + command + " > " + output_file);
}

void compress_using_stream(const std::string & input_file,
                           const std::string & output_file)
{
    ifstream in(input_file.c_str());

    filter_ostream out(output_file);

    char buf[16386];

    while (in) {
        in.read(buf, 16384);
        int n = in.gcount();
        
        out.write(buf, n);
    }
}

void decompress_using_stream(const std::string & input_file,
                             const std::string & output_file)
{
    filter_istream in(input_file);
    ofstream out(output_file.c_str());

    char buf[16386];

    while (in) {
        in.read(buf, 16384);
        int n = in.gcount();
        
        out.write(buf, n);
    }
}

void assert_files_identical(const std::string & input_file,
                            const std::string & output_file)
{
    system("diff " + input_file + " " + output_file);
}

void test_compress_decompress(const std::string & input_file,
                              const std::string & extension,
                              const std::string & zip_command,
                              const std::string & unzip_command)
{
    string base = "filter_streams_test-" + extension;
    string cmp1 = base + ".1." + extension;
    string cmp2 = base + ".2." + extension;
    string dec1 = base + ".1";
    string dec2 = base + ".2";
    string dec3 = base + ".3";
    string dec4 = base + ".4";


    // Test 1: compress using filter stream
    Call_Guard guard1(boost::bind(&::unlink, cmp1.c_str()));
    compress_using_stream(input_file, cmp1);

    // Test 2: compress using tool
    Call_Guard guard2(boost::bind(&::unlink, cmp2.c_str()));
    compress_using_tool(input_file, cmp2, zip_command);

    // Test 3: decompress stream file using tool (sanity check)
    Call_Guard guard3(boost::bind(&::unlink, dec1.c_str()));
    decompress_using_tool(cmp1, dec1, unzip_command);
    assert_files_identical(input_file, dec1);

    // Test 4: decompress tool file using stream
    Call_Guard guard4(boost::bind(&::unlink, dec2.c_str()));
    decompress_using_stream(cmp2, dec2);
    assert_files_identical(input_file, dec2);
    
    // Test 5: decompress stream file using stream
    Call_Guard guard5(boost::bind(&::unlink, dec3.c_str()));
    decompress_using_stream(cmp1, dec3);
    assert_files_identical(input_file, dec3);
    
    // Test 6: decompress tool file using tool (sanity)
    Call_Guard guard6(boost::bind(&::unlink, dec4.c_str()));
    decompress_using_tool(cmp2, dec4, unzip_command);
    assert_files_identical(input_file, dec4);
}

BOOST_AUTO_TEST_CASE( test_compress_decompress_gz )
{
    string input_file = "jml/utils/testing/filter_streams_test.cc";
    test_compress_decompress(input_file, "gz", "gzip", "gzip -d");
}

BOOST_AUTO_TEST_CASE( test_compress_decompress_bzip2 )
{
    string input_file = "jml/utils/testing/filter_streams_test.cc";
    test_compress_decompress(input_file, "bz2", "bzip2", "bzip2 -d");
}

BOOST_AUTO_TEST_CASE( test_compress_decompress_xz )
{
    string input_file = "jml/utils/testing/filter_streams_test.cc";
    test_compress_decompress(input_file, "xz", "xz", "xz -d");
}

BOOST_AUTO_TEST_CASE( test_compress_decompress_lz4 )
{
    string input_file = "jml/utils/testing/filter_streams_test.cc";
    string lz4_cmd = "./build/x86_64/bin/lz4cli";
    test_compress_decompress(input_file, "lz4", lz4_cmd, lz4_cmd + " -d");
}

BOOST_AUTO_TEST_CASE( test_open_failure )
{
    filter_ostream stream;
    {
        JML_TRACE_EXCEPTIONS(false);
        BOOST_CHECK_THROW(stream.open("/no/file/is/here"), std::exception);
        BOOST_CHECK_THROW(stream.open("/no/file/is/here.gz"), std::exception);
        BOOST_CHECK_THROW(stream.open("/no/file/is/here.gz"), std::exception);
    }
}

BOOST_AUTO_TEST_CASE( test_write_failure )
{
    int fd = open("/dev/null", O_RDWR, 0);

    cerr << "fd = " << fd << endl;

    filter_ostream stream(fd);

    stream << "hello" << std::endl;

    close(fd);

    cerr <<" done close" << endl;

    {
        JML_TRACE_EXCEPTIONS(false);
        BOOST_CHECK_THROW(stream << "hello again" << std::endl, std::exception);
    }
}

/* ensures that empty gz/bzip2/xz streams have a valid header */
BOOST_AUTO_TEST_CASE( test_empty_gzip )
{
    fs::create_directories("build/x86_64/tmp");

    string fileprefix("build/x86_64/tmp/empty.");
    vector<string> exts = { "gz", "bz2", "xz", "lz4" };

    for (const auto & ext: exts) {
        string filename = fileprefix + ext;

        /* stream from filename */
        {
            FileCleanup cleanup(filename);

            {
                ML::filter_ostream filestream(filename);
            }

            BOOST_CHECK(get_file_size(filename) > 2);
            {
                ML::filter_istream filestream(filename);
                string line;
                filestream >> line;
                BOOST_CHECK_EQUAL(line.size(), 0);
            }
        }

        /* stream from file descriptor */
        {
            FileCleanup cleanup(filename);

            int fd = open(filename.c_str(), O_RDWR| O_CREAT, S_IRWXU);
            {
                ML::filter_ostream filestream;
                filestream.open(fd, ios::out, ext);
            }
            close(fd);

            BOOST_CHECK(get_file_size(filename) > 2);
            {
                ML::filter_istream filestream(filename);
                string line;
                filestream >> line;
                BOOST_CHECK_EQUAL(line.size(), 0);
            }
        }
    }
}
