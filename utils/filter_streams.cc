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
#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/filter/bzip2.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include <boost/iostreams/device/file.hpp>
#include "jml/arch/exception.h"
#include <errno.h>


using namespace std;


namespace ML {


/*****************************************************************************/
/* FILTER_OSTREAM                                                            */
/*****************************************************************************/

filter_ostream::filter_ostream()
    : ostream(std::cout.rdbuf())
{
}

filter_ostream::
filter_ostream(const std::string & file, std::ios_base::openmode mode)
    : ostream(std::cout.rdbuf())
{
    open(file, mode);
}

namespace {

bool ends_with(const std::string & str, const std::string & what)
{
    string::size_type result = str.rfind(what);
    return result != string::npos
        && result == str.size() - what.size();
}

} // file scope

void filter_ostream::
open(const std::string & file_, std::ios_base::openmode mode)
{
    using namespace boost::iostreams;

    string file = file_;
    if (file == "") file = "/dev/null";

    auto_ptr<filtering_ostream> new_stream
        (new filtering_ostream());

    bool gzip = (ends_with(file, ".gz") || ends_with(file, ".gz~"));
    bool bzip2 = (ends_with(file, ".bz2") || ends_with(file, ".bz2~"));

    if (gzip) new_stream->push(gzip_compressor());
    if (bzip2) new_stream->push(bzip2_compressor());
    
    if (file == "-") {
        new_stream->push(std::cout);
    }
    else {
        new_stream->push(file_sink(file.c_str(), mode));
    }

    stream.reset(new_stream.release());
    rdbuf(stream->rdbuf());

    //stream.reset(new ofstream(file.c_str(), mode));
}

void
filter_ostream::
close()
{
    rdbuf(0);
    //stream->close();
}


/*****************************************************************************/
/* FILTER_ISTREAM                                                            */
/*****************************************************************************/

filter_istream::filter_istream()
    : istream(std::cin.rdbuf())
{
}

filter_istream::
filter_istream(const std::string & file, std::ios_base::openmode mode)
    : istream(std::cin.rdbuf())
{
    open(file, mode);
}

void filter_istream::
open(const std::string & file_, std::ios_base::openmode mode)
{
    using namespace boost::iostreams;

    string file = file_;
    if (file == "") file = "/dev/null";
    
    auto_ptr<filtering_istream> new_stream
        (new filtering_istream());

    bool gzip = (ends_with(file, ".gz") || ends_with(file, ".gz~"));
    bool bzip2 = (ends_with(file, ".bz2") || ends_with(file, ".bz2~"));

    if (gzip) new_stream->push(gzip_decompressor());
    if (bzip2) new_stream->push(bzip2_decompressor());

    if (file == "-") {
        new_stream->push(std::cin);
    }
    else {
        file_source source(file.c_str(), mode);
        if (!source.is_open())
            throw Exception("stream open failed for file %s: %s",
                            file_.c_str(), strerror(errno));
        new_stream->push(source);
    }

    stream.reset(new_stream.release());
    rdbuf(stream->rdbuf());
}

void
filter_istream::
close()
{
    rdbuf(0);
    //stream->close();
}

} // namespace ML
