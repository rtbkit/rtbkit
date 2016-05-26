/* curl_wrapper.h
   Guy Dumais, 4 September 2015
   Copyright (c) 2015 Datacratic Inc.  All rights reserved.

   A thin wrapper on libcurl.
*/

#include "curl_wrapper.h"
#include "jml/utils/exc_assert.h"
#include "jml/compiler/compiler.h"
#include "soa/service/http_header.h"
#include "soa/service/openssl_threading.h"

#include <numeric>

using namespace std;


namespace Datacratic {

namespace CurlWrapper {


    RuntimeError::RuntimeError(const std::string & what, CURLcode code) :
        ML::Exception(what), code(code)
    {}
    
    CURLcode RuntimeError::whatCode() const
    {
        return code;
    }

    /// Static function used to actually perform the callback
    static size_t doCallback(char *buffer, size_t size, size_t nitems, void *userdata)
    {
        Easy::CurlCallback * callback = (Easy::CurlCallback *)userdata;
        try {
            return (*callback)(buffer, size, nitems);
        } JML_CATCH_ALL {
            return size * nitems + 1;  // a number different than the input size indicates an error
        }
    }

    Easy::Easy()
        : curl(curl_easy_init()),
          header_list(nullptr)
    {
        ExcAssert(curl.get());
    }
    
    void Easy::add_data_option(CURLoption option, const void * data)
    {
        attempt(curl_easy_setopt(curl.get(), option, data),
                "easy.add_option: failed to set data option");
    }

    void Easy::add_option(CURLoption option, const std::string & value) 
    {
        attempt(curl_easy_setopt(curl.get(), option, value.c_str()),
                "easy.add_option: failed to set string option");
    }

    void Easy::add_option(CURLoption option, long value)
    {
        attempt(curl_easy_setopt(curl.get(), option, value), "easy.add_option: failed to set long option");
    }
    
    void Easy::add_header_option(const RestParams& headers)
    {
        if (header_list != nullptr)
            throw ML::Exception("header option has already been set. Call reset to before reusing an Easy object");
        
        for (const auto & header : headers) {
            header_list = curl_slist_append(header_list,
                                            (header.first + ":" + header.second).c_str());
        }

        attempt(curl_easy_setopt(curl.get(), CURLOPT_HTTPHEADER, header_list),
                "easy.add_header_option failed to set headers");
    }

    void Easy::add_callback_option(CURLoption option, CURLoption userDataOption, CurlCallback & callback)
    {
        CURLcode res = curl_easy_setopt(curl.get(), userDataOption, &callback);
    
        if (res != CURLE_OK) {
            throw RuntimeError(curl_easy_strerror(res), res);
        }

        res = curl_easy_setopt(curl.get(), option, &doCallback);

        if (res != CURLE_OK) {
            throw RuntimeError(curl_easy_strerror(res), res);
        }
    }
    
    void Easy::get_info(CURLINFO info, long &pdst)
    {
        attempt(curl_easy_getinfo(curl.get(), info, &pdst), "easy.get_info: failed to get info");
    }

    void Easy::get_info(CURLINFO info, double &pdst)
    {
        attempt(curl_easy_getinfo(curl.get(), info, &pdst), "easy.get_info: failed to get info");
    }

    CURLcode Easy::perform()
    {
        return curl_easy_perform(curl.get());
    }
    
    void Easy::reset()
    {
        if (header_list) {
            curl_slist_free_all(header_list);
            header_list = nullptr;
        }
        
        curl_easy_reset(curl.get());
    }

void Easy::CurlCleanup::operator () (CURL * c)
{
    // TODO: check error code... an error indicates an earlier problem
    curl_easy_cleanup(c);
}

namespace {

static struct AtInit {
    AtInit()
    {
        initOpenSSLThreading();
        curl_global_init(CURL_GLOBAL_DEFAULT);
    }
    
    ~AtInit()
    {
        curl_global_cleanup();
    }
} atInit;

} // file scope

} // namespace CurlWrapper
} // namespace Datacratic

    
