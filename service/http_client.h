/* http_client.h                                                   -*- C++ -*-
   Wolfgang Sourdeau, January 2014
   Copyright (c) 2014 Datacratic.  All rights reserved.

   An asynchronous HTTP client.
*/

#pragma once

#include "sys/epoll.h"

#include <string>
#include <vector>

#include <curlpp/Easy.hpp>
#include <curlpp/Multi.hpp>
#include <curlpp/Types.hpp>

#include "jml/arch/wakeup_fd.h"
#include "jml/utils/ring_buffer.h"

#include "soa/service/http_endpoint.h"
#include "soa/service/http_header.h"
#include "soa/jsoncpp/reader.h"


/* Notes:
   - mostly similar to http_rest_proxy
   - lacks support for cookies yet
   - no support for EPOLLONESHOT yet */

namespace Datacratic {


/** Header for an HTTP request.  Just a place to dump the data. */

struct HttpResponseHeader {
    HttpResponseHeader()
        : code_(0), contentLength_(-1)
    {
    }

    int code()
        const
    {
        return code_;
    }
    int code_; // HTTP status code

    std::string version()
        const
    {
        return version_;
    }
    std::string version_; // HTTP version

    void clear();

    std::string contentType()
        const
    {
        return contentType_;
    }
    std::string contentType_;

    size_t contentLength()
        const
    {
        return contentLength_;
    }
    ssize_t contentLength_;

    std::string getHeader(const std::string & key)
        const
    {
        // std::string & value = headers_.at(key);
        // return value;
        return std::string();
    }

    std::string tryGetHeader(const std::string & key) const
    {
        return std::string();
        // auto it = headers_.find(key);
        // if (it == headers_.end())
        //     return "";
        // return *it->second;
    }
    std::map<std::string, std::string> headers_;

    /* curl helper */
    void parseLine(const std::string & headerLine);
};

/** The response of a request.  Has a return code and a body. */
struct HttpClientResponse {
    enum Error {
        NONE,
        UNKNOWN,
        TIMEOUT,
        HOST_NOT_FOUND,
        COULD_NOT_CONNECT,
    };
    static std::string errorMessage(Error errorCode);

    HttpClientResponse()
        : errorCode_(Error::NONE)
    {}

    /** Body of the REST call. */
    std::string body()
        const
    {
        return body_;
    }

    Json::Value jsonBody()
        const
    {
        return Json::parse(body_);
    }

    void clear()
    {
        errorCode_ = Error::NONE;
        header_.clear();
        body_ = "";
    }

    Error errorCode_;
    HttpResponseHeader header_;
    std::string body_;
};

/* Representation of an HTTP request. */
struct HttpRequest {
    typedef std::function<void (const HttpClientResponse &,
                                const HttpRequest &)> OnResponse;

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

        Content(const Json::Value & content,
                const std::string & contentType = "application/json")
            : str(content.toString()), data(str.c_str()),
              size(str.size()), hasContent(true),
              contentType(contentType)
        {
        }

        std::string str;
        const char * data;
        uint64_t size;

        bool hasContent;

        std::string contentType;
        std::string contentMd5;
    };

    HttpRequest()
        : timeout_(-1)
    {
    }

    HttpRequest(const std::string & verb, const std::string & url,
                const OnResponse & onResponse,
                const Content & content, const RestParams & headers,
                int timeout = -1)
        noexcept
        : verb_(verb), url_(url), onResponse_(onResponse),
          content_(content), headers_(headers),
          timeout_(timeout)
    {
    }

    HttpRequest(const HttpRequest & other)
        noexcept
        : verb_(other.verb_), url_(other.url_),
          onResponse_(other.onResponse_), content_(other.content_),
          headers_(other.headers_), timeout_(other.timeout_)
    {
    }

    HttpRequest(HttpRequest && other)
        noexcept
    {
        if (&other != this) {
            verb_ = std::move(other.verb_);
            url_ = std::move(other.url_);
            onResponse_ = std::move(other.onResponse_);
            content_ = std::move(other.content_);
            headers_ = std::move(other.headers_);
            timeout_ = other.timeout_;
        }
    }

    HttpRequest & operator = (const HttpRequest & other)
        noexcept
    {
        if (&other != this) {
            verb_ = other.verb_;
            url_ = other.url_;
            onResponse_ = other.onResponse_,
            content_ = other.content_;
            headers_ = other.headers_;
            timeout_ = other.timeout_;
        }

        return *this;
    }

    void clear()
    {
        *this = HttpRequest();
    }

    std::string verb_;
    std::string url_;
    OnResponse onResponse_;
    Content content_;
    RestParams headers_;
    int timeout_;
};


struct HttpConnection {
    HttpConnection();

    HttpConnection(const HttpConnection & other) = delete;

    void clear()
    {
        easy_.reset();
        request_.clear();
        response_.clear();
        afterContinue_ = false;
        uploadOffset_ = 0;
    }
    void perform(bool noSSLChecks, bool debug);

    curlpp::types::WriteFunctionFunctor onHeader_;
    curlpp::types::WriteFunctionFunctor onWrite_;
    size_t onCurlHeader(char * data, size_t size) noexcept;
    size_t onCurlWrite(char * data, size_t size) noexcept;

    curlpp::types::ReadFunctionFunctor onRead_;
    size_t onCurlRead(char * data, size_t size, size_t max) noexcept;

    HttpRequest request_;

    curlpp::Easy easy_;
    HttpClientResponse response_;
    bool afterContinue_;
    size_t uploadOffset_;

    struct HttpConnection *next;
};

struct HttpClient : public AsyncEventSource {
    HttpClient(const std::string & baseUrl,
               int numParallel = 4, size_t queueSize = 32);
    ~HttpClient();

    bool post(const std::string & resource,
              const HttpRequest::OnResponse & onResponse,
              const HttpRequest::Content & content = HttpRequest::Content(),
              const RestParams & queryParams = RestParams(),
              const RestParams & headers = RestParams(),
              int timeout = -1)
    {
        return enqueueRequest("POST", resource, onResponse, content,
                              queryParams, headers, timeout);
    }

    bool put(const std::string & resource,
             const HttpRequest::OnResponse & onResponse,
             const HttpRequest::Content & content = HttpRequest::Content(),
             const RestParams & queryParams = RestParams(),
             const RestParams & headers = RestParams(),
             int timeout = -1)
    {
        return enqueueRequest("PUT", resource, onResponse, content,
                              queryParams, headers, timeout);
    }

    bool get(const std::string & resource,
             const HttpRequest::OnResponse & onResponse,
             const RestParams & queryParams = RestParams(),
             const RestParams & headers = RestParams(),
             int timeout = -1)
    {
        // std::cerr << "get\n";
        return enqueueRequest("GET", resource, onResponse,
                              HttpRequest::Content(),
                              queryParams, headers, timeout);
    }

    // /** Add a cookie to the connection. */
    // void setCookie(const std::string & value)
    // {
    //     headers_.emplace_back("Cookie", value);
    // }

    // /** Add a cookie to the connection that comes in from the response. */
    // void setCookieFromResponse(const HttpClientResponse & r)
    // {
    //     cookies_.emplace_back("Cookie: "
    //                           + r.header_.getHeader("set-cookie"));
    // }

    /* AsyncEventSource */
    virtual int selectFd() const;
    virtual bool processOne();
    // virtual bool poll() const;

    /* internal */
    bool enqueueRequest(const std::string & verb,
                        const std::string & resource,
                        const HttpRequest::OnResponse & onResponse,
                        const HttpRequest::Content & content,
                        const RestParams & queryParams,
                        const RestParams & headers,
                        int timeout = -1);

    void handleEvents();
    void handleEvent(const ::epoll_event & event);

    void checkMultiInfos();

    int onCurlSocketEvent(CURL *e, curl_socket_t s, int what, void *sockp);
    int onCurlTimerEvent(long timeout_ms);

    void addFd(int fd, bool isMod, int flags) const;
    void removeFd(int fd) const;

    HttpConnection * getConnection();
    void releaseConnection(HttpConnection * connection);

    std::string baseUrl_;

    /** SSL checks */
    bool noSSLChecks;

    /** Are we debugging? */
    bool debug;

    int fd_;
    ML::Wakeup_Fd wakeup_;
    int timerFd_;

    curlpp::Multi multi_;
    ::CURLM * handle_;

    HttpConnection * connections_;
    std::vector<HttpConnection> connectionStash_;

    ML::RingBufferSRMW<HttpRequest> queue_; /* queued requests */
};

}
