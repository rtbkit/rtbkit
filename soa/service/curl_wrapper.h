/* curl_wrapper.h                                                  -*- C++ -*-
   Guy Dumais, 4 September 2015
   Copyright (c) 2015 Datacratic Inc.  All rights reserved.

   A thin wrapper on libcurl.
*/

#pragma once

#include "jml/arch/exception.h"

#include <curl/curl.h>
#include <functional>
#include <stdexcept>
#include <vector>
#include <memory>

namespace Datacratic
{
    class RestParams;
    
namespace CurlWrapper {
    class Easy;

    class RuntimeError : public ML::Exception
    {
    public:
        RuntimeError(const std::string & reason, CURLcode code);

        /**
         * Returns the CURLcode that libcurl returned.
         */
        CURLcode whatCode() const;

    private:
        CURLcode code;

    };
    
    /**
     * Wraps libcurl's easy interface.
     */
    class Easy
    {
    public:
        // this callback MUST process all the input data and return this size
        // any other return value is considered an error by libcurl
        typedef std::function<size_t(char*, size_t, size_t)> CurlCallback;
        
        Easy();

        void add_option(CURLoption option, long value);
        void add_option(CURLoption option, const std::string& value);
        void add_header_option(const RestParams& headers);
        void add_callback_option(CURLoption option, CURLoption userDataOption, CurlCallback & callback);
        void add_data_option(CURLoption option, const void * data);

        /* Perform the request in a blocking manner.  No exception will be
           thrown if there is an error; instead it will be returned in the
           CURLcode that is returned from the result of this command.
        */
        CURLcode perform();

        /* reset the options while keeping the handle.  there could be a speed 
           advantage to reuse the same Easy object */
        void reset();

        /* get the info of the session */
        void get_info(CURLINFO info, long &pdst);
        void get_info(CURLINFO info, double &pdst);

        // this is required until we decide to wrap the multi interface
        operator CURL*() const {return curl.get();}
        
    private:
        /// Function object to cleanup a CURL instance
        struct CurlCleanup {
            void operator () (CURL *);
        };

        /// The CURL handle we're operating with.  We make use of the
        /// unique_ptr to handle its lifetime.
        std::unique_ptr<CURL, CurlCleanup> curl;

        /// List of headers for the *current* request.  These are reset
        /// whenever we call reset().
        struct curl_slist *header_list;
        
        /// To be called after an attempt.  This will create an error if
        /// it failed with the given error text.
        inline void attempt(CURLcode res, const std::string errortext)
        {
            if (res != CURLE_OK)
                throw RuntimeError(errortext, res);
        }

    };
}
}

    
