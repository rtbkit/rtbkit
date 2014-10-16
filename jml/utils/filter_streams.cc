/* filter_streams.cc
   Jeremy Barnes, 17 March 2005
   Copyright (c) 2005 Jeremy Barnes.  All rights reserved.
   
   This file is part of "Jeremy's Machine Learning Library", copyright (c)
   1999-2005 Jeremy Barnes.
   
   This program is available under the GNU General Public License, the terms
   of which are given by the file "license.txt" in the top level directory of
   the source code distribution.  If this file is missing, you have no right
   to use the program; please contact the author.

   This program is distributed in the hope that it will be useful, but
   WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
   or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
   for more details.

   ---
   
   Implementation of filter streams.
*/

#include "filter_streams.h"
#include <fstream>
#include <mutex>
#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/filter/bzip2.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include <boost/iostreams/device/file.hpp>
#include <boost/iostreams/device/file_descriptor.hpp>
#include <boost/version.hpp>
#include <boost/lexical_cast.hpp>
#include "jml/arch/exception.h"
#include "jml/arch/format.h"
#include "string_functions.h"
#include <errno.h>
#include <sstream>
#include <thread>
#include <unordered_map>
#include "lzma.h"
#include "lz4_filter.h"


using namespace std;


namespace ML {

const UriHandlerFunction &
getUriHandler(const std::string & scheme);

std::pair<std::string, std::string>
getScheme(const std::string & uri)
{
    string::size_type pos = uri.find("://");
    if (pos == string::npos) {
        return make_pair("file", uri);
    }

    string scheme(uri, 0, pos);
    string resource(uri, pos + 3);

    return make_pair(scheme, resource);
}

/*****************************************************************************/
/* FILTER_OSTREAM                                                            */
/*****************************************************************************/

filter_ostream::filter_ostream()
    : ostream(std::cout.rdbuf()), deferredFailure(false)
{
}

filter_ostream::
filter_ostream(filter_ostream && other) noexcept
    : ostream(other.rdbuf()),
    stream(std::move(other.stream)),
    sink(std::move(other.sink)),
    deferredFailure(false)
{
}

filter_ostream::
filter_ostream(const std::string & file, std::ios_base::openmode mode,
               const std::string & compression, int level)
    : ostream(std::cout.rdbuf()), deferredFailure(false)
{
    open(file, mode, compression, level);
}

filter_ostream::
filter_ostream(int fd, std::ios_base::openmode mode,
               const std::string & compression, int level)
    : ostream(std::cout.rdbuf()), deferredFailure(false)
{
    open(fd, mode, compression, level);
}

filter_ostream::
filter_ostream(int fd,
               const std::map<std::string, std::string> & options)
    : ostream(std::cout.rdbuf()), deferredFailure(false)
{
    open(fd, options);
}

filter_ostream::
filter_ostream(const std::string & uri,
               const std::map<std::string, std::string> & options)
    : ostream(std::cout.rdbuf()), deferredFailure(false)
{
    open(uri, options);
}

filter_ostream::
~filter_ostream()
{
    try {
        close();
    }
    catch (...) {
        cerr << "~filter_ostream: ignored exception\n";
    }
}

filter_ostream &
filter_ostream::
operator = (filter_ostream && other)
{
    exceptions(ios::goodbit);
    stream = std::move(other.stream);
    sink = std::move(other.sink);
    rdbuf(other.rdbuf());
    exceptions(other.exceptions());
    other.exceptions(ios::goodbit);
    other.rdbuf(0);

    return *this;
}

namespace {

bool ends_with(const std::string & str, const std::string & what)
{
    string::size_type result = str.rfind(what);
    return result != string::npos
        && result == str.size() - what.size();
}

void addCompression(streambuf & buf,
                    boost::iostreams::filtering_ostream & stream,
                    const std::string & resource,
                    const std::string & compression,
                    int compressionLevel)
{
    using namespace boost::iostreams;

    if (compression == "gz" || compression == "gzip"
        || (compression == ""
            && (ends_with(resource, ".gz") || ends_with(resource, ".gz~")))) {
        gzip_compressor compressor;
        if (compressionLevel != -1) {
            compressor = gzip_compressor(compressionLevel);
        }
        compressor.write(buf, "", 0);
        stream.push(compressor);
    }
    else if (compression == "bz2" || compression == "bzip2"
        || (compression == ""
            && (ends_with(resource, ".bz2") || ends_with(resource, ".bz2~")))) {
        if (compressionLevel == -1)
            stream.push(bzip2_compressor());
        else stream.push(bzip2_compressor(compressionLevel));
    }
    else if (compression == "lzma" || compression == "xz"
        || (compression == ""
            && (ends_with(resource, ".xz") || ends_with(resource, ".xz~")))) {
        if (compressionLevel == -1)
            stream.push(lzma_compressor());
        else stream.push(lzma_compressor(compressionLevel));
    }
    else if (compression == "lz4"
        || (compression == ""
            && (ends_with(resource, ".lz4") || ends_with(resource, ".lz4~")))) {
        stream.push(lz4_compressor(compressionLevel));
    }
    else if (compression != "" && compression != "none")
        throw ML::Exception("unknown filter compression " + compression);
    
}

void addCompression(streambuf & buf,
                    boost::iostreams::filtering_ostream & stream,
                    const std::string & resource,
                    const std::map<std::string, std::string> & options)
{
    string compression;
    auto it = options.find("compression");
    if (it != options.end())
        compression = it->second;

    int compressionLevel = -1;
    it = options.find("compressionLevel");
    if (it != options.end())
        compressionLevel = boost::lexical_cast<int>(it->second);
    
    addCompression(buf, stream, resource, compression, compressionLevel);
}


/** Create an options map from a set of legacy options passed by value. */
std::map<std::string, std::string>
createOptions(std::ios_base::openmode mode,
              const std::string & compression,
              int compressionLevel)
{
    /* 
app	(append) Set the stream's position indicator to the end of the stream before each output operation.
ate	(at end) Set the stream's position indicator to the end of the stream on opening.
binary	(binary) Consider stream as binary rather than text.
in	(input) Allow input operations on the stream.
out	(output) Allow output operations on the stream.
trunc	(truncate) Any current content is discarded, assuming a length of zero on opening.
    */

    string modeStr;
    auto addMode = [&] (int mask, const char * name)
        {
            if ((mode & mask) == 0)
                return;
            if (!modeStr.empty())
                modeStr += ',';
            modeStr += name;
        };

    addMode(ios_base::app, "app");
    addMode(ios_base::ate, "ate");
    addMode(ios_base::binary, "binary");
    addMode(ios_base::in, "in");
    addMode(ios_base::out, "out");
    addMode(ios_base::trunc, "trunc");

    //cerr << "compression = " << compression << endl;

    std::map<std::string, std::string> result;

    if (!modeStr.empty())
        result["mode"] = modeStr;
    if (!compression.empty())
        result["compression"] = compression;
    if (compressionLevel != -1)
        result["compressionLevel"] = std::to_string(compressionLevel);

    return result;
}

std::ios_base::openmode getMode(const std::map<std::string, std::string> & options)
{
    std::ios_base::openmode result
        = std::ios_base::openmode(0);

    auto it = options.find("mode");
    if (it == options.end())
        return result;

    vector<string> elements = split(it->second, ',');

    for (auto & el: elements) {
        if (el == "app")
            result |= ios_base::app;
        else if (el == "ate")
            result |= ios_base::ate;
        else if (el == "binary")
            result |= ios_base::binary;
        else if (el == "in")
            result |= ios_base::in;
        else if (el == "out")
            result |= ios_base::out;
        else if (el == "trunc")
            result |= ios_base::trunc;
        else throw ML::Exception("unknown filter_stream open mode " + el);
    }

    return result;
}

} // file scope

void
filter_ostream::
open(const std::string & uri, std::ios_base::openmode mode,
     const std::string & compression, int compressionLevel)
{
    //cerr << "uri = " << uri << " compression = " << compression << endl;

    open(uri, createOptions(mode, compression, compressionLevel));
}

void
filter_ostream::
open(const std::string & uri, std::ios_base::openmode mode,
     const std::string & compression, int compressionLevel, 
     unsigned int numThreads)
{
    //cerr << "uri = " << uri << " compression = " << compression << endl;
    std::map<std::string, std::string>  options = 
         createOptions(mode, compression, compressionLevel);
    // add the number of threads to the options
    options["num-threads"] = to_string(numThreads);
    open(uri, options);
}


void
filter_ostream::
open(const std::string & uri,
     const std::map<std::string, std::string> & options)
{
    using namespace boost::iostreams;

    string scheme, resource;
    std::tie(scheme, resource) = getScheme(uri);

    std::ios_base::openmode mode = getMode(options);
    if (!mode)
        mode = std::ios_base::out;

    //cerr << "opening scheme " << scheme << " resource " << resource
    //     << endl;

    const auto & handler = getUriHandler(scheme);
    std::streambuf * buf;
    bool weOwnBuf;
    auto onException = [&]() { this->deferredFailure = true; };
    std::tie(buf, weOwnBuf) = handler(scheme, resource, mode, options,
                                      onException);
    
    return openFromStreambuf(buf, weOwnBuf, resource, options);
}

void
filter_ostream::
openFromStreambuf(std::streambuf * buf,
                  bool weOwnBuf,
                  const std::string & resource,
                  const std::string & compression,
                  int compressionLevel)
{
    openFromStreambuf(buf, weOwnBuf, resource,
                      createOptions(std::ios_base::openmode(0),
                                    compression, compressionLevel));
}    

void
filter_ostream::
openFromStreambuf(std::streambuf * buf,
                  bool weOwnBuf,
                  const std::string & resource,
                  const std::map<std::string, std::string> & options)
{

    // TODO: exception safety for buf

    using namespace boost::iostreams;

    //cerr << "buf = " << (void *)buf << endl;
    //cerr << "weOwnBuf = " << weOwnBuf << endl;

    std::unique_ptr<std::streambuf> sink;
    if (weOwnBuf)
        sink.reset(buf);

    unique_ptr<filtering_ostream> new_stream
        (new filtering_ostream());

    addCompression(*buf, *new_stream, resource, options);

    new_stream->push(*buf);

    this->stream = std::move(new_stream);
    this->sink = std::move(sink);
    rdbuf(this->stream->rdbuf());

    exceptions(ios::badbit | ios::failbit);
}

void filter_ostream::
open(int fd, std::ios_base::openmode mode,
     const std::string & compression, int compressionLevel)
{
    open(fd, createOptions(mode, compression, compressionLevel));
}

void filter_ostream::
open(int fd, const std::map<std::string, std::string> & options)
{
    using namespace boost::iostreams;
    
    unique_ptr<filtering_ostream> new_stream
        (new filtering_ostream());

    stringbuf headerbuf;
    addCompression(headerbuf, *new_stream, "", options);
    string header = headerbuf.str();
    if (!header.empty()) {
        ssize_t rc = ::write(fd, header.c_str(), header.size());
        if (rc < 0) {
            throw ML::Exception(errno, "open", "open");
        }
    }

#if (BOOST_VERSION < 104100)
    new_stream->push(file_descriptor_sink(fd));
#else
    new_stream->push(file_descriptor_sink(fd,
                                          boost::iostreams::never_close_handle));
#endif
    stream.reset(new_stream.release());
    sink.reset();
    rdbuf(stream->rdbuf());

    exceptions(ios::badbit | ios::failbit);
}

void
filter_ostream::
close()
{
    if (stream) {
        boost::iostreams::flush(*stream);
        boost::iostreams::close(*stream);
    }
    exceptions(ios::goodbit);
    rdbuf(0);
    stream.reset();
    sink.reset();
    options.clear();
    if (deferredFailure) {
        deferredFailure = false;
        exceptions(ios::badbit | ios::failbit);
        setstate(ios::badbit);
    }
}

std::string
filter_ostream::
status() const
{
    if (*this) return "good";
    else return format("%s%s%s",
                       fail() ? " fail" : "",
                       bad() ? " bad" : "",
                       eof() ? " eof" : "");
}


/*****************************************************************************/
/* FILTER_ISTREAM                                                            */
/*****************************************************************************/

filter_istream::filter_istream()
    : istream(std::cin.rdbuf()), deferredFailure(false)
{
}

filter_istream::
filter_istream(const std::string & file, std::ios_base::openmode mode,
               const std::string & compression)
    : istream(std::cin.rdbuf()),
      deferredFailure(false)
{
    open(file, mode, compression);
}

filter_istream::
filter_istream(filter_istream && other) noexcept
    : istream(other.rdbuf()),
    stream(std::move(other.stream)),
    sink(std::move(other.sink)),
    deferredFailure(false)
{
}

filter_istream::
~filter_istream()
{
    try {
        close();
    }
    catch (...) {
        cerr << "~filter_istream: ignored exception\n";
    }
}

filter_istream &
filter_istream::
operator = (filter_istream && other)
{
    exceptions(ios::goodbit);
    stream = std::move(other.stream);
    sink = std::move(other.sink);
    rdbuf(other.rdbuf());
    exceptions(other.exceptions());
    other.exceptions(ios::goodbit);
    other.rdbuf(0);

    return *this;
}

void
filter_istream::
open(const std::string & uri,
     std::ios_base::openmode mode,
     const std::string & compression)
{
    exceptions(ios::badbit);

    string scheme, resource;
    std::tie(scheme, resource) = getScheme(uri);

    const auto & handler = getUriHandler(scheme);
    std::streambuf * buf;
    bool weOwnBuf;
    auto onException = [&]() { this->deferredFailure = true; };
    std::tie(buf, weOwnBuf) = handler(scheme, resource, mode,
                                      createOptions(mode, compression, -1),
                                      onException);

    openFromStreambuf(buf, weOwnBuf, resource, compression);
}

void
filter_istream::
openFromStreambuf(std::streambuf * buf,
                  bool weOwnBuf,
                  const std::string & resource,
                  const std::string & compression)
{
    // TODO: exception safety for buf

    using namespace boost::iostreams;

    std::unique_ptr<std::streambuf> sink;
    if (weOwnBuf)
        sink.reset(buf);
    
    unique_ptr<filtering_istream> new_stream
        (new filtering_istream());

    bool gzip = (compression == "gz" || compression == "gzip"
                 || (compression == ""
                     && (ends_with(resource, ".gz")
                         || ends_with(resource, ".gz~"))));
    bool bzip2 = (compression == "bz2" || compression == "bzip2"
                 || (compression == ""
                     && (ends_with(resource, ".bz2")
                         || ends_with(resource, ".bz2~"))));
    bool lzma = (compression == "xz" || compression == "lzma"
                 || (compression == ""
                     && (ends_with(resource, ".xz")
                         || ends_with(resource, ".xz~"))));

    bool lz4 = (compression == "lz4"
                 || (compression == ""
                     && (ends_with(resource, ".lz4")
                         || ends_with(resource, ".lz4~"))));

    if (gzip) new_stream->push(gzip_decompressor());
    if (bzip2) new_stream->push(bzip2_decompressor());
    if (lzma) new_stream->push(lzma_decompressor());
    if (lz4) new_stream->push(lz4_decompressor());

    new_stream->push(*buf);

    this->stream = std::move(new_stream);
    this->sink = std::move(sink);
    rdbuf(this->stream->rdbuf());
}

void
filter_istream::
close()
{
    if (stream) {
        boost::iostreams::flush(*stream);
        boost::iostreams::close(*stream);
    }
    exceptions(ios::goodbit);
    rdbuf(0);
    stream.reset();
    sink.reset();
    if (deferredFailure) {
        deferredFailure = false;
        exceptions(ios::badbit | ios::failbit);
        setstate(ios::badbit);
    }
}

string
filter_istream::
readAll()
{
    string result;

    char buffer[65536];
    while (*this) {
        read(buffer, sizeof(buffer));
        result.append(buffer, gcount());
    }

    return result;
}

/*****************************************************************************/
/* REGISTRATION                                                              */
/*****************************************************************************/

namespace {

std::mutex uriHandlersLock;
std::unordered_map<std::string, UriHandlerFunction> uriHandlers;

} // file scope

void registerUriHandler(const std::string & scheme,
                        const UriHandlerFunction & handler)
{
    if (!handler)
        throw ML::Exception("registerUriHandler: null handler passed");

    std::unique_lock<std::mutex> guard(uriHandlersLock);
    auto it = uriHandlers.find(scheme);
    if (it != uriHandlers.end())
        throw ML::Exception("already have a Uri handler registered for scheme "
                            + scheme);
    uriHandlers[scheme] = handler;
}

const UriHandlerFunction &
getUriHandler(const std::string & scheme)
{
    std::unique_lock<std::mutex> guard(uriHandlersLock);
    auto it = uriHandlers.find(scheme);
    if (it == uriHandlers.end())
        throw ML::Exception("Uri handler not found for scheme " + scheme);
    return it->second;
}

struct RegisterFileHandler {
    static std::pair<std::streambuf *, bool>
    getFileHandler(const std::string & scheme,
                   std::string resource,
                   std::ios_base::open_mode mode,
                   const std::map<std::string, std::string> & options,
                   const OnUriHandlerException & onException)
    {
        if (resource == "")
            resource = "/dev/null";

        if (mode == ios::in) {
            if (resource == "-")
                return make_pair(cin.rdbuf(), false);
            unique_ptr<std::filebuf> buf(new std::filebuf);
            buf->open(resource, ios_base::openmode(mode));

            if (!buf->is_open())
                throw ML::Exception("couldn't open file %s: %s",
                                    resource.c_str(), strerror(errno));

            return make_pair(buf.release(), true);
        }
        else if (mode & ios::out) {
            if (resource == "-")
                return make_pair(cout.rdbuf(), false);

            unique_ptr<std::filebuf> buf(new std::filebuf);
            buf->open(resource, ios_base::openmode(mode));

            if (!buf->is_open())
                throw ML::Exception("couldn't open file %s: %s",
                                    resource.c_str(), strerror(errno));

            return make_pair(buf.release(), true);
        }
        else throw ML::Exception("no way to create file handler for non in/out");
    }

    RegisterFileHandler()
    {
        registerUriHandler("file", getFileHandler);
    }

} registerFileHandler;



/* "mem:" scheme for using memory as storage backend. Only useful for testing. */

namespace {

mutex memStringsLock;
map<string, string> memStrings;

} // "mem" scheme

string &
getMemStreamString(const string & name)
{
    unique_lock<mutex> guard(memStringsLock);

    return memStrings.at(name);    
}

void
setMemStreamString(const std::string & name,
                   const std::string & contents)
{
    unique_lock<mutex> guard(memStringsLock);

    memStrings.insert({name, contents});
}

void
deleteMemStreamString(const std::string & name)
{
    unique_lock<mutex> guard(memStringsLock);

    memStrings.erase(name);    
}

void
deleteAllMemStreamStrings()
{
    unique_lock<mutex> guard(memStringsLock);

    memStrings.clear();
}


struct MemStreamingInOut {
    MemStreamingInOut(string & targetString)
        : open_(true), pos_(0), targetString_(targetString)
    {
    }

    typedef char char_type;

    bool is_open() const
    {
        return open_;
    }

    void close()
    {
        open_ = false;
    }

    bool open_;
    size_t pos_;
    string & targetString_;
};

struct MemStreamingIn : public MemStreamingInOut {
    struct category
        : public boost::iostreams::input,
          public boost::iostreams::device_tag,
          public boost::iostreams::closable_tag
    {
    };

    MemStreamingIn(string & targetString)
        : MemStreamingInOut(targetString)
    {
    }

    streamsize read(char * s, streamsize n)
    {
        streamsize res;
        unique_lock<mutex> guard(memStringsLock);

        size_t maxLen = std::min<size_t>(targetString_.size() - pos_,
                                         n);
        if (maxLen > 0) {
            const char * start = targetString_.c_str() + pos_;
            copy(start, start + maxLen, s);
            pos_ += maxLen;
            res = maxLen;
        }
        else {
            res = -1;
        }

        return res;
    }
};

struct MemStreamingOut : public MemStreamingInOut {
    struct category
        : public boost::iostreams::output,
          public boost::iostreams::device_tag,
          public boost::iostreams::closable_tag
    {
    };

    MemStreamingOut(string & targetString)
        : MemStreamingInOut(targetString)
    {
    }

    streamsize write(const char * s, streamsize n)
    {
        unique_lock<mutex> guard(memStringsLock);

        size_t nextLen = pos_ + n;
        targetString_.reserve(nextLen);
        targetString_.append(s, n);
        pos_ = nextLen;

        return n;
    }
};

struct RegisterMemHandler {
    static pair<streambuf *, bool>
    getMemHandler(const string & scheme,
                  string resource,
                  ios_base::open_mode mode,
                  const map<string, string> & options,
                  const OnUriHandlerException & onException)
    {
        if (scheme != "mem")
            throw ML::Exception("bad scheme name");
        if (resource == "")
            throw ML::Exception("bad resource name");

        unique_lock<mutex> guard(memStringsLock);

        // cerr << "string resource : " + resource + "\n";
        string & targetString = memStrings[resource];

        unique_ptr<std::streambuf> streamBuf;
        if (mode == ios::in) {
            streamBuf.reset(new boost::iostreams::stream_buffer<MemStreamingIn>(MemStreamingIn(targetString),
                                                                                4096));
        }
        else if (mode == ios::out) {
            streamBuf.reset(new boost::iostreams::stream_buffer<MemStreamingOut>(MemStreamingOut(targetString),
                                                                                 4096));
        }
        else {
            throw ML::Exception("unable to create mem handler");
        }
        return make_pair(streamBuf.release(), true);
    }

    RegisterMemHandler()
    {
        registerUriHandler("mem", getMemHandler);
    }

} registerMemHandler;

} // namespace ML
