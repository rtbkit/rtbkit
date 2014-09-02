/* filter_streams.h                                                -*- C++ -*-
   Jeremy Barnes, 12 March 2005
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
   
   Streams that understand "-" syntax.
*/

#ifndef __utils__filter_streams_h__
#define __utils__filter_streams_h__

#include <atomic>
#include <iostream>
#include <fstream>
#include <memory>
#include <map>

namespace ML {


/*****************************************************************************/
/* FILTER OSTREAM                                                            */
/*****************************************************************************/

/** Ostream class that has the following features:
    - It has move semantics so can be passed by reference
    - It can add filters to compress / decompress
    - It can hook into other filesystems (eg s3, ...) based upon an
      extensible API.
*/

class filter_ostream : public std::ostream {
public:
    filter_ostream();
    filter_ostream(const std::string & uri,
                   std::ios_base::openmode mode = std::ios_base::out,
                   const std::string & compression = "",
                   int compressionLevel = -1);
    filter_ostream(int fd,
                   std::ios_base::openmode mode = std::ios_base::out,
                   const std::string & compression = "",
                   int compressionLevel = -1);

    filter_ostream(int fd,
                   const std::map<std::string, std::string> & options);

    filter_ostream(const std::string & uri,
                   const std::map<std::string, std::string> & options);

    filter_ostream(filter_ostream && other) noexcept;

    filter_ostream & operator = (filter_ostream && other);

    ~filter_ostream();

    void open(const std::string & uri,
              std::ios_base::openmode mode = std::ios_base::out,
              const std::string & compression = "",
              int level = -1);
    // have a version where we can specify the number of threads
    void open(const std::string & uri,
              std::ios_base::openmode mode,
              const std::string & compression,
              int level , unsigned int numThreads);

    void open(int fd,
              std::ios_base::openmode mode = std::ios_base::out,
              const std::string & compression = "",
              int level = -1);
    

    void openFromStreambuf(std::streambuf * buf,
                           bool weOwnBuf,
                           const std::string & resource = "",
                           const std::string & compression = "",
                           int compressionLevel = -1);
                           
    /** Open with the given options.  Option keys are interpreted by plugins,
        but include:

        mode = comma separated list of out,append,create
        compression = string (gz, bz2, xz, ...)
        resource = string to be used in error messages
    */
    void open(const std::string & uri,
              const std::map<std::string, std::string> & options);

    void open(int fd,
              const std::map<std::string, std::string> & options);

    void openFromStreambuf(std::streambuf * buf,
                           bool weOwnBuf,
                           const std::string & resource,
                           const std::map<std::string, std::string> & options);
    
    void close();

    std::string status() const;

    /* notifies that an exception occurred in the streambuf */
    void notifyException()
    {
        deferredFailure = true;
    }

private:
    std::unique_ptr<std::ostream> stream;
    std::unique_ptr<std::streambuf> sink;
    std::atomic<bool> deferredFailure;
    std::map<std::string, std::string> options;
};


/*****************************************************************************/
/* FILTER ISTREAM                                                            */
/*****************************************************************************/

class filter_istream : public std::istream {
public:
    filter_istream();
    filter_istream(const std::string & uri,
                   std::ios_base::openmode mode = std::ios_base::in,
                   const std::string & compression = "");

    filter_istream(filter_istream && other) noexcept;

    filter_istream & operator = (filter_istream && other);

    ~filter_istream();

    void open(const std::string & uri,
              std::ios_base::openmode mode = std::ios_base::in,
              const std::string & comparession = "");

    void openFromStreambuf(std::streambuf * buf,
                           bool weOwnBuf,
                           const std::string & resource = "",
                           const std::string & compression = "");

    void close();

    /* read the entire stream into a std::string */
    std::string readAll();

private:
    std::unique_ptr<std::istream> stream;
    std::unique_ptr<std::streambuf> sink;
    std::atomic<bool> deferredFailure;
};


/*****************************************************************************/
/* REGISTRY                                                                  */
/*****************************************************************************/

/* The type of a function a uri handler invokes when an execption is thrown at
 * closing time. This serves as a workaround of the silent catching that
 * boost::iostreams::stream_buffer performs when a streambuf is being
 * destroyed. To be effective, it requires that "close" be called on all
 * streams before destruction. */
typedef std::function<void ()> OnUriHandlerException;

typedef std::function<std::pair<std::streambuf *, bool>
                      (const std::string & scheme,
                       const std::string & resource,
                       std::ios_base::openmode mode,
                       const std::map<std::string, std::string> & options,
                       const OnUriHandlerException & onException)>
UriHandlerFunction;

void registerUriHandler(const std::string & scheme,
                        const UriHandlerFunction & handler);

std::string & getMemStreamString(const std::string & name);
void setMemStreamString(const std::string & name,
                        const std::string & contents);
void deleteMemStreamString(const std::string & name);
void deleteAllMemStreamStrings();

} // namespace ML

#endif /* __utils__filter_streams_h__ */

