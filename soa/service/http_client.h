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
   - no support for EPOLLONESHOT yet
   - has not been tweaked for performance yet
   - since those require header interpretation, there is not support for
     cookies per se
*/

#pragma once

#include "sys/epoll.h"

#include <mutex>
#include <queue>
#include <string>
#include <vector>

#include <curlpp/Easy.hpp>
#include <curlpp/Multi.hpp>
#include <curlpp/Types.hpp>

#include "jml/arch/wakeup_fd.h"

#include "soa/jsoncpp/value.h"
#include "soa/service/async_event_source.h"
#include "soa/service/http_header.h"


namespace Datacratic {

struct MessageLoop;

struct HttpClientCallbacks;

/* HTTPREQUEST */

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


/* HTTPCLIENT */

struct HttpClient : public AsyncEventSource {

    /* "baseUrl": scheme, hostname and port (scheme://hostname[:port]) that
       will be used as base for all requests
       "numParallels": number of requests that can be handled simultaneously
       "queueSize": size of the backlog of pending requests, after which
       operations will be refused */
    HttpClient(const std::string & baseUrl,
               int numParallel = 4);
    HttpClient(HttpClient && other) noexcept;
    HttpClient(const HttpClient & other) = delete;

    ~HttpClient();

    /** SSL checks */
    bool noSSLChecks;

    /** Use with servers that support HTTP pipelining */
    void enablePipelining();

    /** Performs a POST request, with "resource" as the location of the
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

    HttpClient & operator = (HttpClient && other) noexcept;

private:
    /* AsyncEventSource */
    virtual int selectFd() const;
    virtual bool processOne();

    /* Local */
    bool enqueueRequest(const std::string & verb,
                        const std::string & resource,
                        const std::shared_ptr<HttpClientCallbacks> & callbacks,
                        const HttpRequest::Content & content,
                        const RestParams & queryParams,
                        const RestParams & headers,
                        int timeout = -1);
    std::vector<HttpRequest> popRequests(size_t number);

    void handleEvents();
    void handleEvent(const ::epoll_event & event);
    void handleWakeupEvent();
    void handleTimerEvent();
    void handleMultiEvent(const ::epoll_event & event);

    void checkMultiInfos();

    static int socketCallback(CURL *e, curl_socket_t s, int what,
                              void *clientP, void *sockp);
    int onCurlSocketEvent(CURL *e, curl_socket_t s, int what, void *sockp);

    static int timerCallback(CURLM *multi, long timeoutMs, void *clientP);
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

    int fd_;
    ML::Wakeup_Fd wakeup_;
    int timerFd_;

    curlpp::Multi multi_;
    ::CURLM * handle_;

    std::vector<HttpConnection> connectionStash_;
    std::vector<HttpConnection *> avlConnections_;
    size_t nextAvail_;

    typedef std::mutex Mutex;
    typedef std::unique_lock<Mutex> Guard;
    Mutex queueLock_;
    std::queue<HttpRequest> queue_; /* queued requests */
};


enum struct HttpClientError {
    None,
    Unknown,
    Timeout,
    HostNotFound,
    CouldNotConnect,
};

std::ostream & operator << (std::ostream & stream, HttpClientError error);

/* HTTPCLIENTCALLBACKS */

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


/* SIMPLE CALLBACKS */

/* This class enables to simplify the interface use by clients which do not
 * need support for progressive responses. */
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
