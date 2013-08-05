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
#include "jml/arch/exception.h"
#include "jml/arch/format.h"
#include <errno.h>
#include <sstream>
#include <thread>
#include <unordered_map>
#include "lzma.h"


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
    : ostream(std::cout.rdbuf())
{
}

filter_ostream::
filter_ostream(filter_ostream && other) noexcept
    : ostream(other.rdbuf()),
    stream(std::move(other.stream)),
    sink(std::move(other.sink))
{
}

filter_ostream::
filter_ostream(const std::string & file, std::ios_base::openmode mode,
               const std::string & compression, int level)
    : ostream(std::cout.rdbuf())
{
    open(file, mode, compression, level);
}

filter_ostream::
filter_ostream(int fd, std::ios_base::openmode mode,
               const std::string & compression, int level)
    : ostream(std::cout.rdbuf())
{
    open(fd, mode);
}

filter_ostream::
~filter_ostream()
{
    close();
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
    else if (compression != "" && compression != "none")
        throw ML::Exception("unknown filter compression " + compression);
    
}

} // file scope

void
filter_ostream::
open(const std::string & uri, std::ios_base::openmode mode,
     const std::string & compression, int compressionLevel)
{
    using namespace boost::iostreams;

    string scheme, resource;
    std::tie(scheme, resource) = getScheme(uri);

    //cerr << "opening scheme " << scheme << " resource " << resource
    //     << endl;

    const auto & handler = getUriHandler(scheme);
    std::streambuf * buf;
    bool weOwnBuf;
    std::tie(buf, weOwnBuf) = handler(scheme, resource, mode);

    return openFromStreambuf(buf, weOwnBuf, resource, compression,
                             compressionLevel);
}

void
filter_ostream::
openFromStreambuf(std::streambuf * buf,
                  bool weOwnBuf,
                  const std::string & resource,
                  const std::string & compression,
                  int compressionLevel)
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

    addCompression(*buf, *new_stream, resource, compression, compressionLevel);

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
    using namespace boost::iostreams;
    
    unique_ptr<filtering_ostream> new_stream
        (new filtering_ostream());

    if (compression.size() > 0) {
        stringbuf headerbuf;
        addCompression(headerbuf, *new_stream, "", compression,
                       compressionLevel);
        string header = headerbuf.str();
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
    stream.reset();
    sink.reset();
    rdbuf(0);
    //stream->close();
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
    : istream(std::cin.rdbuf())
{
}

filter_istream::
filter_istream(const std::string & file, std::ios_base::openmode mode,
               const std::string & compression)
    : istream(std::cin.rdbuf())
{
    open(file, mode, compression);
}

filter_istream::
filter_istream(filter_istream && other) noexcept
    : istream(other.rdbuf()),
    stream(std::move(other.stream)),
    sink(std::move(other.sink))
{
}

filter_istream::
~filter_istream()
{
    close();
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
    std::tie(buf, weOwnBuf) = handler(scheme, resource, mode);

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

    if (gzip) new_stream->push(gzip_decompressor());
    if (bzip2) new_stream->push(bzip2_decompressor());
    if (lzma) new_stream->push(lzma_decompressor());

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
    stream.reset();
    sink.reset();
    rdbuf(0);
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
                   std::ios_base::open_mode mode)
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

} // namespace ML
