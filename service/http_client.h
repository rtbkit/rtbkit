/* http_client.h                                                   -*- C++ -*-
   Wolfgang Sourdeau, January 2014
   Copyright (c) 2014 Datacratic.  All rights reserved.

   An asynchronous HTTP client.

   HttpClient is meant to provide a featureful asynchronous HTTP client class.
   It supports strictly asynchronous (non-blocking) operations, HTTP
   pipelining and concurrent requests while enabling streaming responses via a
   callback mechanism. It is meant to be subclassed whenever a synchronous
   interface or a one-shot response mechanism is required. In general, the
   code should be complete enough that existing and similar classes could be
   subclassed gradually (HttpRestProxy, s3 internals).

   Caveat:
   - no support for EPOLLONESHOT yet
   - has not been tweaked for performance yet
   - does and will not provide any support for cookies per se
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

#include "soa/jsoncpp/value.h"
#include "soa/service/async_event_source.h"


namespace Datacratic {

struct HttpClientCallbacks;


/* HTTPREQUEST */

/* Representation of an HTTP request. */
struct HttpRequest {
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
        : callbacks_(nullptr), timeout_(-1)
    {
    }

    HttpRequest(const std::string & verb, const std::string & url,
                const HttpClientCallbacks & callbacks,
                const Content & content, const RestParams & headers,
                int timeout = -1)
        noexcept
        : verb_(verb), url_(url), callbacks_(&callbacks),
          content_(content), headers_(headers),
          timeout_(timeout)
    {
    }

    HttpRequest(const HttpRequest & other)
        noexcept
        : verb_(other.verb_), url_(other.url_),
          callbacks_(other.callbacks_), content_(other.content_),
          headers_(other.headers_), timeout_(other.timeout_)
    {
    }

    HttpRequest(HttpRequest && other)
        noexcept
    {
        if (&other != this) {
            verb_ = std::move(other.verb_);
            url_ = std::move(other.url_);
            callbacks_ = other.callbacks_;
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
            callbacks_ = other.callbacks_,
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
    const HttpClientCallbacks * callbacks_;
    Content content_;
    RestParams headers_;
    int timeout_;
};


/* HTTPCLIENT */

struct HttpClient : public AsyncEventSource {
    HttpClient(const std::string & baseUrl,
               int numParallel = 4, size_t queueSize = 32);
    ~HttpClient();

    bool post(const std::string & resource,
              const HttpClientCallbacks & callbacks,
              const HttpRequest::Content & content = HttpRequest::Content(),
              const RestParams & queryParams = RestParams(),
              const RestParams & headers = RestParams(),
              int timeout = -1)
    {
        return enqueueRequest("POST", resource, callbacks, content,
                              queryParams, headers, timeout);
    }

    bool put(const std::string & resource,
             const HttpClientCallbacks & callbacks,
             const HttpRequest::Content & content = HttpRequest::Content(),
             const RestParams & queryParams = RestParams(),
             const RestParams & headers = RestParams(),
             int timeout = -1)
    {
        return enqueueRequest("PUT", resource, callbacks, content,
                              queryParams, headers, timeout);
    }

    bool get(const std::string & resource,
             const HttpClientCallbacks & callbacks,
             const RestParams & queryParams = RestParams(),
             const RestParams & headers = RestParams(),
             int timeout = -1)
    {
        return enqueueRequest("GET", resource, callbacks,
                              HttpRequest::Content(),
                              queryParams, headers, timeout);
    }

    /* AsyncEventSource */
    virtual int selectFd() const;
    virtual bool processOne();
    // virtual bool poll() const;

    /* internal */
    bool enqueueRequest(const std::string & verb,
                        const std::string & resource,
                        const HttpClientCallbacks & callbacks,
                        const HttpRequest::Content & content,
                        const RestParams & queryParams,
                        const RestParams & headers,
                        int timeout = -1);

    void handleEvents();
    void handleEvent(const ::epoll_event & event);
    void handleWakeupEvent();
    void handleTimerEvent();
    void handleMultiEvent(const ::epoll_event & event);

    void checkMultiInfos();

    int onCurlSocketEvent(CURL *e, curl_socket_t s, int what, void *sockp);
    int onCurlTimerEvent(long timeout_ms);

    void addFd(int fd, bool isMod, int flags) const;
    void removeFd(int fd) const;

    struct HttpConnection {
        HttpConnection();

        HttpConnection(const HttpConnection & other) = delete;

        void clear()
        {
            easy_.reset();
            request_.clear();
            afterContinue_ = false;
            uploadOffset_ = 0;
        }
        void perform(bool noSSLChecks, bool debug);

        /* header and body write callbacks */
        curlpp::types::WriteFunctionFunctor onHeader_;
        curlpp::types::WriteFunctionFunctor onWrite_;
        size_t onCurlHeader(const char * data, size_t size) noexcept;
        size_t onCurlWrite(const char * data, size_t size) noexcept;

        /* body read callback */
        curlpp::types::ReadFunctionFunctor onRead_;
        size_t onCurlRead(char * buffer, size_t bufferSize) noexcept;

        HttpRequest request_;

        curlpp::Easy easy_;
        // HttpClientResponse response_;
        bool afterContinue_;
        size_t uploadOffset_;

        struct HttpConnection *next;
    };

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


/* HTTPCLIENTCALLBACKS */

struct HttpClientCallbacks {
    enum Error {
        NONE,
        UNKNOWN,
        TIMEOUT,
        HOST_NOT_FOUND,
        COULD_NOT_CONNECT,
    };

    typedef std::function<void (const HttpRequest &,
                                const std::string &,
                                int code)> OnResponseStart;
    typedef std::function<void (const HttpRequest &,
                                const std::string &)> OnData;
    typedef std::function<void (const HttpRequest & rq,
                                Error errorCode)> OnDone;

    HttpClientCallbacks(OnResponseStart onResponseStart = nullptr,
                        OnData onHeader = nullptr,
                        OnData onData = nullptr,
                        OnDone onDone = nullptr)
        : onResponseStart_(onResponseStart),
          onHeader_(onHeader), onData_(onData),
          onDone_(onDone)
    {
    }

    virtual ~HttpClientCallbacks()
    {
    }

    static const std::string & errorMessage(Error errorCode);

    /* initiates a response */
    virtual void onResponseStart(const HttpRequest & rq,
                                 const std::string & httpVersion,
                                 int code) const;

    /* callback for header lines */
    virtual void onHeader(const HttpRequest & rq,
                          const std::string & header) const;

    /* callback for body data */
    virtual void onData(const HttpRequest & rq,
                        const std::string & data) const;

    /* callback for operation completions */
    virtual void onDone(const HttpRequest & rq,
                        Error errorCode) const;

    OnResponseStart onResponseStart_;
    OnData onHeader_;
    OnData onData_;
    OnDone onDone_;
};

} // namespace Datacratic

