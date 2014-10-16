/* filter_streams_test.cc
   Jeremy Barnes, 29 June 2011
   Copyright (c) 2011 Datacratic.
   Copyright (c) 2011 Jeremy Barnes.

*/

#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <string.h>

#include "jml/utils/file_functions.h"
#include "jml/utils/filter_streams.h"
#include "jml/arch/exception.h"
#include "jml/arch/exception_handler.h"

#include <boost/filesystem.hpp>
#include <boost/iostreams/stream_buffer.hpp>
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

#if 0
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

BOOST_AUTO_TEST_CASE( test_large_blocks )
{
    size_t blockSize = 64 * 1024 * 1024;
    char * block = new char [blockSize];
    for (unsigned i = 0;  i < blockSize;  ++i) {
        block[i] = i ^ (i << 8) ^ (i << 16) ^ (1 << 24);
    }
    
    string filename = "build/x86_64/tmp/badfile.xz4";

    FileCleanup cleanup(filename);
        
    {
        ML::filter_ostream stream(filename);
        stream.write(block, blockSize);
    }

    {
        ML::filter_istream stream(filename);
        char * block2 = new char[blockSize];
        stream.read(block2, blockSize);
        BOOST_CHECK_EQUAL(stream.gcount(), blockSize);

        //BOOST_CHECK(stream.eof());

        BOOST_CHECK_EQUAL_COLLECTIONS(block, block + blockSize,
                                      block2, block2 + blockSize);
    }
}
#endif

#if 1
/* ensures that writing a 8M bytes text works */
BOOST_AUTO_TEST_CASE( test_mem_scheme_out )
{
    Call_Guard fn([&]() {deleteAllMemStreamStrings();});

    string text("");
    {
        string pattern("AbCdEfGh");
        text.reserve(pattern.size() * 1000000);
        for (int i = 0; i < 1000000; i++) {
            text += pattern;
        }
    }

    {
        ML::filter_ostream outS("mem://out_file.txt");
        outS << text;
    }

    string result = getMemStreamString("out_file.txt");
    BOOST_CHECK_EQUAL(text, result);
}
#endif

#if 1
/* ensures that writing a 8M bytes text to a gz target invokes the gz
 * filter */
BOOST_AUTO_TEST_CASE( test_mem_scheme_out_gz )
{
    Call_Guard fn([&]() {deleteAllMemStreamStrings();});

    string text("");
    {
        string pattern("AbCdEfGh");
        text.reserve(pattern.size() * 1000000);
        for (int i = 0; i < 1000000; i++) {
            text += pattern;
        }
    }

    {
        ML::filter_ostream outS("mem://out_file.gz");
        outS << text;
    }

    /* A very poor check, indeed... Ultimatly, a real decompression should
    occur, but since we know that the gz filter works, and that unfiltered
    output works, logically we can deduce that the gz filter was invoked
    correctly if the strings are not equal. */
    string result = getMemStreamString("out_file.gz");
    BOOST_CHECK_NE(text, result);
}
#endif

#if 1
/* ensures that reading a 8M bytes text works well too */
BOOST_AUTO_TEST_CASE( test_mem_scheme_in )
{
    Call_Guard fn([&]() {deleteAllMemStreamStrings();});

    string text("");
    {
        string pattern("AbCdEfGh");
        text.reserve(pattern.size() * 1000000);
        for (int i = 0; i < 1000000; i++) {
            text += pattern;
        }
        setMemStreamString("in_file.txt", text);
    }

    string result;
    ML::filter_istream inS("mem://in_file.txt");
    while (inS) {
        char buf[16384];
        inS.read(buf, 16384);
        result.append(buf, inS.gcount());
    }

    BOOST_CHECK_EQUAL(text, result);
}
#endif

#if 1
/* Testing the behaviour of filter_stream when exceptions occur during read,
 * write, close or destruction */
struct ExceptionSource {
    enum ThrowType {
        ThrowOnWrite,
        ThrowOnRead,
        ThrowOnClose
    };

    ExceptionSource(const OnUriHandlerException & onException,
                    ThrowType throwType)
        : onException_(onException), throwType_(throwType)
    {
    }

    typedef char char_type;
    struct category
        : public boost::iostreams::bidirectional_device_tag,
          public boost::iostreams::closable_tag
    {
    };

    bool is_open() const
    {
        return true;
    }

    std::streamsize write(const char_type* s, std::streamsize n)
    {
        if (throwType_ == ThrowType::ThrowOnWrite) {
            throw ML::Exception("throwing when writing");
        }
        return n;
    }

    std::streamsize read(char_type* s, std::streamsize n)
    {
        if (throwType_ == ThrowType::ThrowOnRead) {
            throw ML::Exception("throwing when reading");
        }
        char randomdata[n];
        ::memcpy(s, randomdata, n);
        return n;
    }

    void close(ios::openmode which)
    {
        if (throwType_ == ThrowType::ThrowOnClose) {
            onException_();
        }
    }

    OnUriHandlerException onException_;
    ThrowType throwType_;
};

struct RegisterExcHandlers {
    static std::pair<std::streambuf *, bool>
    getExceptionSource(const OnUriHandlerException & onException,
                       ExceptionSource::ThrowType throwType)
    {
        unique_ptr<streambuf> handler;

        handler.reset(new boost::iostreams::stream_buffer<ExceptionSource>
                      (ExceptionSource(onException, throwType),
                       1));

        return make_pair(handler.release(), true);
    }

    static std::pair<std::streambuf *, bool>
    getExcOnReadHandler(const std::string & scheme,
                        const std::string & resource,
                        std::ios_base::open_mode mode,
                        const std::map<std::string, std::string> & options,
                        const OnUriHandlerException & onException)
    {
        return getExceptionSource(onException, ExceptionSource::ThrowOnRead);
    }

    static std::pair<std::streambuf *, bool>
    getExcOnWriteHandler(const std::string & scheme,
                         const std::string & resource,
                         std::ios_base::open_mode mode,
                         const std::map<std::string, std::string> & options,
                         const OnUriHandlerException & onException)
    {
        return getExceptionSource(onException, ExceptionSource::ThrowOnWrite);
    }

    static std::pair<std::streambuf *, bool>
    getExcOnCloseHandler(const std::string & scheme,
                         const std::string & resource,
                         std::ios_base::open_mode mode,
                         const std::map<std::string, std::string> & options,
                         const OnUriHandlerException & onException)
    {
        return getExceptionSource(onException, ExceptionSource::ThrowOnClose);
    }

    void registerBuckets()
    {
    }

    RegisterExcHandlers()
    {
        ML::registerUriHandler("throw-on-read", getExcOnReadHandler);
        ML::registerUriHandler("throw-on-write", getExcOnWriteHandler);
        ML::registerUriHandler("throw-on-close", getExcOnCloseHandler);
    }
} registerExcHandlers;

BOOST_AUTO_TEST_CASE(test_filter_stream_exceptions_read)
{
    ML::filter_istream stream("throw-on-read://exception-zone");

    string data;
    auto action = [&]() {
        JML_TRACE_EXCEPTIONS(false);
        stream >> data;
    };

    BOOST_CHECK_THROW(action(), ML::Exception);
}

BOOST_AUTO_TEST_CASE(test_filter_stream_exceptions_write)
{
    JML_TRACE_EXCEPTIONS(false);
    ML::filter_ostream stream("throw-on-write://exception-zone");

    auto action = [&]() {
        string sample("abcdef0123456789");
        /* we loop enough times to saturate the stream internal buffer
         * and cause "write" to be called */
        for (int i = 0; i < 1000000; i++) {
            stream << sample;
        }
    };

    BOOST_CHECK_THROW(action(), ML::Exception);
}

BOOST_AUTO_TEST_CASE(test_filter_stream_exceptions_close)
{
    ML::filter_ostream stream("throw-on-close://exception-zone");

    auto action = [&]() {
        JML_TRACE_EXCEPTIONS(false);
        stream.close();
    };

    BOOST_CHECK_THROW(action(), std::ios_base::failure);
}

BOOST_AUTO_TEST_CASE(test_filter_stream_exceptions_destruction_ostream)
{
    unique_ptr<ML::filter_ostream> stream;
    stream.reset(new ML::filter_ostream("throw-on-close://exception-zone"));

    auto action = [&]() {
        JML_TRACE_EXCEPTIONS(false);
        stream.reset();
    };

    action();
}

BOOST_AUTO_TEST_CASE(test_filter_stream_exceptions_destruction_istream)
{
    unique_ptr<ML::filter_istream> stream;
    stream.reset(new ML::filter_istream("throw-on-close://exception-zone"));

    auto action = [&]() {
        JML_TRACE_EXCEPTIONS(false);
        stream.reset();
    };

    action();
}

#endif
