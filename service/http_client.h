/* http_client.h                                                   -*- C++ -*-
   Wolfgang Sourdeau, January 2014
   Copyright (c) 2014 Datacratic.  All rights reserved.

   An asynchronous HTTP client.

   HttpClient is meant to provide a featureful, generic and asynchronous HTTP
   client class. It supports strictly asynchronous (non-blocking) operations,
   HTTP pipelining and concurrent requests while enabling streaming responses
   via a callback mechanism. It is meant to be subclassed whenever a
   synchronous interface or a one-shot response mechanism is required. In
   general, the code should be complete enough that existing and similar
   classes could be subclassed gradually (HttpRestProxy, s3 internals). As a
   generic class, it does not make assumptions on the transferred contents.
   Finally, it is based on the interface of HttpRestProxy.

   Caveat:
   - since those require header interpretation, there is not support for
     cookies per se
*/

#pragma once

#include <memory>
#include <string>

#include "soa/jsoncpp/value.h"
#include "soa/service/async_event_source.h"
#include "soa/service/http_header.h"


namespace Datacratic {

/* Forward declarations */

struct HttpClientCallbacks;


/****************************************************************************/
/* HTTP REQUEST                                                             */
/****************************************************************************/

/* Representation of an HTTP request. */

struct HttpRequest {
    /** Structure used to hold content for a POST request. */
    struct Content {
        Content() = default;

        Content(const std::string & str,
                const std::string & contentType = "")
            : str(str), contentType(contentType)
        {
        }

        Content(const char * data, uint64_t size,
                const std::string & contentType = "")
            : str(data, size), contentType(contentType)
        {
        }

        Content(const Json::Value & content,
                const std::string & contentType = "application/json")
            : str(content.toString()), contentType(contentType)
        {
        }

        std::string str;
        std::string contentType;
    };

    HttpRequest()
        : timeout_(-1)
    {
    }

    HttpRequest(const std::string & verb, const std::string & url,
                const std::shared_ptr<HttpClientCallbacks> & callbacks,
                const Content & content, const RestParams & headers,
                int timeout = -1)
        noexcept
        : verb_(verb), url_(url), callbacks_(callbacks),
          content_(content), headers_(headers),
          timeout_(timeout)
    {
    }

    void clear()
    {
        verb_ = "";
        url_ = "";
        callbacks_ = nullptr;
        content_ = Content();
        headers_ = RestParams();
        timeout_ = -1;
    }

    std::string verb_;
    std::string url_;
    std::shared_ptr<HttpClientCallbacks> callbacks_;
    Content content_;
    RestParams headers_;
    int timeout_;
};


/****************************************************************************/
/* HTTP CLIENT IMPL                                                         */
/****************************************************************************/

struct HttpClientImpl : public AsyncEventSource {
    HttpClientImpl(const std::string & baseUrl,
                   int numParallel = 1024, int queueSize = 0)
        : AsyncEventSource()
    {
    }

    HttpClientImpl(HttpClientImpl && other) = default;

    virtual ~HttpClientImpl()
    {}

    /** Enable debugging */
    virtual void enableDebug(bool value) = 0;

    /** SSL checks */
    virtual void enableSSLChecks(bool value) = 0;

    /** Enable the TCP_NODELAY option, also known as the Nagle's algorithm */
    virtual void enableTcpNoDelay(bool value) = 0;

    /** Use with servers that support HTTP pipelining */
    virtual void enablePipelining(bool value) = 0;

    /** Enqueue (or perform) the specified request */
    virtual bool enqueueRequest(const std::string & verb,
                                const std::string & resource,
                                const std::shared_ptr<HttpClientCallbacks> & callbacks,
                                const HttpRequest::Content & content,
                                const RestParams & queryParams,
                                const RestParams & headers,
                                int timeout = -1) = 0;

    /* Returns the number of requests in the queue */
    virtual size_t queuedRequests() const = 0;
};


/****************************************************************************/
/* HTTP CLIENT ERROR                                                        */
/****************************************************************************/

enum struct HttpClientError {
    None,
    Unknown,
    Timeout,
    HostNotFound,
    CouldNotConnect,
    SendError,
    RecvError
};

std::ostream & operator << (std::ostream & stream, HttpClientError error);


/****************************************************************************/
/* HTTP CLIENT                                                              */
/****************************************************************************/

struct HttpClient : public AsyncEventSource {
    /* This sets the requested version of the underlying HttpClientImpl. By
     * default, this value is deduced from the "HTTP_CLIENT_VERSION"
     * environment variable. It not set, this falls back to 1. */
    static void setHttpClientImplVersion(int version);

    /* "baseUrl": scheme, hostname and port (scheme://hostname[:port]) that
       will be used as base for all requests
       "numParallels": number of requests that can be handled simultaneously
       "queueSize": size of the backlog of pending requests, after which
       operations will be refused (0 = infinite)
       "implVersion": use version X of the HttpClientImpl, fallback
       to HTTP_CLIENT_IMPL
    */
    HttpClient(const std::string & baseUrl,
               int numParallel = 1024, int queueSize = 0,
               int implVersion = 0);
    HttpClient(HttpClient && other) noexcept
    {
        *this = std::move(other);
    }
    HttpClient(const HttpClient & other) = delete;

    virtual int selectFd()
        const
    {
        return impl->selectFd();
    }

    virtual bool processOne()
    {
        return impl->processOne();
    }

    /** Enable debugging */
    void enableDebug(bool value)
    {
        impl->enableDebug(value);
    }

    /** SSL checks */
    void enableSSLChecks(bool value)
    {
        impl->enableSSLChecks(value);
    }

    /** Enable the TCP_NODELAY option, also known as the Nagle's algorithm */
    void enableTcpNoDelay(bool value)
    {
        impl->enableTcpNoDelay(value);
    }

    /** Enable the requesting of "100 Continue" responses in preparation of
     * a PUT request */
    void sendExpect100Continue(bool value);

    /** Use with servers that support HTTP pipelining */
    void enablePipelining(bool value)
    {
        impl->enablePipelining(value);
    }

    /** Performs a GET request, with "resource" as the location of the
     *  resource on the server indicated in "baseUrl". Query parameters
     *  should preferably be passed via "queryParams".
     *
     *  Returns "true" when the request could successfully be enqueued.
     */
    bool get(const std::string & resource,
             const std::shared_ptr<HttpClientCallbacks> & callbacks,
             const RestParams & queryParams = RestParams(),
             const RestParams & headers = RestParams(),
             int timeout = -1)
    {
        return enqueueRequest("GET", resource, callbacks,
                              HttpRequest::Content(),
                              queryParams, headers, timeout);
    }

    /** Performs a POST request, using similar parameters as get with the
     * addition of "content" which defines the contents body and type.
     *
     *  Returns "true" when the request could successfully be enqueued.
     */
    bool post(const std::string & resource,
              const std::shared_ptr<HttpClientCallbacks> & callbacks,
              const HttpRequest::Content & content = HttpRequest::Content(),
              const RestParams & queryParams = RestParams(),
              const RestParams & headers = RestParams(),
              int timeout = -1)
    {
        return enqueueRequest("POST", resource, callbacks, content,
                              queryParams, headers, timeout);
    }

    /** Performs a PUT request in a similar fashion to "post" above.
     *
     *  Returns "true" when the request could successfully be enqueued.
     */
    bool put(const std::string & resource,
             const std::shared_ptr<HttpClientCallbacks> & callbacks,
             const HttpRequest::Content & content = HttpRequest::Content(),
             const RestParams & queryParams = RestParams(),
             const RestParams & headers = RestParams(),
             int timeout = -1)
    {
        return enqueueRequest("PUT", resource, callbacks, content,
                              queryParams, headers, timeout);
    }

    /** Performs a DELETE request. Note that this method cannot be named
     * "delete", which is a reserved keyword in C++.
     *
     *  Returns "true" when the request could successfully be enqueued.
     */
    bool del(const std::string & resource,
             const std::shared_ptr<HttpClientCallbacks> & callbacks,
             const RestParams & queryParams = RestParams(),
             const RestParams & headers = RestParams(),
             int timeout = -1)
    {
        return enqueueRequest("DELETE", resource, callbacks,
                              HttpRequest::Content(),
                              queryParams, headers, timeout);
    }

    /** Enqueue (or perform) the specified request */
    bool enqueueRequest(const std::string & verb,
                        const std::string & resource,
                        const std::shared_ptr<HttpClientCallbacks> & callbacks,
                        const HttpRequest::Content & content,
                        const RestParams & queryParams,
                        const RestParams & headers,
                        int timeout = -1)
    {
        return impl->enqueueRequest(verb, resource, callbacks, content,
                                    queryParams, headers, timeout);
    }

    size_t queuedRequests()
        const
    {
        return impl->queuedRequests();
    }

    HttpClient & operator = (HttpClient && other) noexcept
    {
        if (&other != this) {
            impl = std::move(other.impl);
        }

        return *this;
    }

private:
    std::unique_ptr<HttpClientImpl> impl;
};


/****************************************************************************/
/* HTTP CLIENT CALLBACKS                                                    */
/****************************************************************************/

struct HttpClientCallbacks {
    typedef std::function<void (const HttpRequest &,
                                const std::string &,
                                int code)> OnResponseStart;
    typedef std::function<void (const HttpRequest &,
                                const char * data, size_t size)> OnData;
    typedef std::function<void (const HttpRequest & rq,
                                HttpClientError errorCode)> OnDone;

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

    static const std::string & errorMessage(HttpClientError errorCode);

    /* initiates a response */
    virtual void onResponseStart(const HttpRequest & rq,
                                 const std::string & httpVersion,
                                 int code);

    /* callback for header lines, one invocation per line */
    virtual void onHeader(const HttpRequest & rq,
                          const char * data, size_t size);

    /* callback for body data, one invocation per chunk */
    virtual void onData(const HttpRequest & rq,
                        const char * data, size_t size);

    /* callback for operation completions, implying that no other call will
     * be performed for the same request */
    virtual void onDone(const HttpRequest & rq,
                        HttpClientError errorCode);

private:
    OnResponseStart onResponseStart_;
    OnData onHeader_;
    OnData onData_;
    OnDone onDone_;
};


/****************************************************************************/
/* HTTP CLIENT SIMPLE CALLBACKS                                             */
/****************************************************************************/

/* This class is a child of HttpClientCallbacks and offers a simplified
 * interface when support for progressive responses is not necessary. */

struct HttpClientSimpleCallbacks : public HttpClientCallbacks
{
    typedef std::function<void (const HttpRequest &,  /* request */
                                HttpClientError,      /* error code */
                                int,                  /* status code */
                                std::string &&,       /* headers */
                                std::string &&)>      /* body */
        OnResponse;
    HttpClientSimpleCallbacks(const OnResponse & onResponse = nullptr);

    /* HttpClientCallbacks overrides */
    virtual void onResponseStart(const HttpRequest & rq,
                                 const std::string & httpVersion, int code);
    virtual void onHeader(const HttpRequest & rq,
                          const char * data, size_t size);
    virtual void onData(const HttpRequest & rq,
                        const char * data, size_t size);
    virtual void onDone(const HttpRequest & rq, HttpClientError errorCode);

    virtual void onResponse(const HttpRequest & rq,
                            HttpClientError error,
                            int status,
                            std::string && headers,
                            std::string && body);

private:
    OnResponse onResponse_;

    int statusCode_;
    std::string headers_;
    std::string body_;
};

} // namespace Datacratic
