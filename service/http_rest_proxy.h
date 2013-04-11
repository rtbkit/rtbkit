/* http_rest_proxy.h                                               -*- C++ -*-
   Jeremy Barnes, 10 April 2013
   Copyright (c) 2013 Datacratic Inc.  All rights reserved.

*/

#pragma once

#include "soa/service/http_endpoint.h"
#include "jml/utils/vector_utils.h"
#include <boost/make_shared.hpp>


namespace Datacratic {


/*****************************************************************************/
/* HTTP REST PROXY                                                           */
/*****************************************************************************/

/** A class that can be used to perform queries against an HTTP service.
    Normally used to consume REST-like APIs, hence the name.
*/

struct HttpRestProxy {

    HttpRestProxy(const std::string & serviceUri = "")
        : serviceUri(serviceUri)
    {
    }

    /** The response of a request.  Has a return code and a body. */
    struct Response {
        Response()
            : code_(0)
        {
        }

        /** Return code of the REST call. */
        int code() const {
            return code_;
        }

        /** Body of the REST call. */
        std::string body() const
        {
            if (code_ < 200 || code_ >= 300)
                throw ML::Exception("invalid http code returned");
            return body_;
        }

        /** Get the given response header of the REST call. */
        std::string getHeader(const std::string & name) const
        {
            auto it = header_.headers.find(name);
            if (it == header_.headers.end())
                throw ML::Exception("required header " + name + " not found");
            return it->second;
        }

        long code_;
        std::string body_;
        HttpHeader header_;
    };

    /** Structure used to hold content for a POST request. */
    struct Content {
        Content()
            : data(0), size(0), hasContent(false)
        {
        }

        Content(const std::string & str,
                const std::string & contentType = "")
            : str(str), data(str.c_str()), size(str.size()),
              hasContent(true), contentType(contentType)
        {
        }

        Content(const char * data, uint64_t size,
                const std::string & contentType = "",
                const std::string & contentMd5 = "")
            : data(data), size(size), hasContent(true),
              contentType(contentType), contentMd5(contentMd5)
        {
        }

        std::string str;

        const char * data;
        uint64_t size;
        bool hasContent;

        std::string contentType;
        std::string contentMd5;
    };

    /** Perform a POST request from end to end. */
    Response post(const std::string & resource,
                  const Content & content = Content(),
                  const RestParams & queryParams = RestParams(),
                  const RestParams & headers = RestParams(),
                  int timeout = -1) const
    {
        return perform("POST", resource, content, queryParams, headers,
                       timeout);
    }

    /** Perform a synchronous GET request from end to end. */
    Response get(const std::string & resource,
                 const RestParams & queryParams = RestParams(),
                 const RestParams & headers = RestParams(),
                 int timeout = -1) const
    {
        return perform("GET", resource, Content(), queryParams, headers,
                       timeout);
    }

    /** Perform a synchronous request from end to end. */
    Response perform(const std::string & verb,
                     const std::string & resource,
                     const Content & content = Content(),
                     const RestParams & queryParams = RestParams(),
                     const RestParams & headers = RestParams(),
                     int timeout = -1) const;
    
    /** URI that will be automatically prepended to resources passed in to
        the perform() methods
    */
    std::string serviceUri;
};

} // namespace Datacratic

