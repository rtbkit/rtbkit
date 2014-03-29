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

#include <string>
#include <vector>

#include <curlpp/Easy.hpp>
#include <curlpp/Multi.hpp>
#include <curlpp/Types.hpp>

#include "jml/arch/wakeup_fd.h"
#include "jml/utils/ring_buffer.h"

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
        : callbacks_(nullptr), timeout_(-1)
    {
    }

    HttpRequest(const std::string & verb, const std::string & url,
                HttpClientCallbacks & callbacks,
                const Content & content, const RestParams & headers,
                int timeout = -1)
        noexcept
        : verb_(verb), url_(url), callbacks_(&callbacks),
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
    HttpClientCallbacks * callbacks_;
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
               int numParallel = 4, size_t queueSize = 32);
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
             HttpClientCallbacks & callbacks,
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
              HttpClientCallbacks & callbacks,
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
             HttpClientCallbacks & callbacks,
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

    void fixConnectionStash();

    /* Local */
    bool enqueueRequest(const std::string & verb,
                        const std::string & resource,
                        HttpClientCallbacks & callbacks,
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
                                 int code);

    /* callback for header lines, one invocation per line */
    virtual void onHeader(const HttpRequest & rq,
                          const std::string & header);

    /* callback for body data, one invocation per chunk */
    virtual void onData(const HttpRequest & rq,
                        const std::string & data);

    /* callback for operation completions, implying that no other call will
     * be performed for the same request */
    virtual void onDone(const HttpRequest & rq,
                        Error errorCode);

private:
    OnResponseStart onResponseStart_;
    OnData onHeader_;
    OnData onData_;
    OnDone onDone_;
};


/* HTTP CLIENT POOL */

/* In general, there is one socket per HttpClient instance and requests are
 * queued until that socket become available. With HttpClientPool, it is
 * ensured that requests are distributed more or less equally across different
 * instances, in a round-robin fashion. This enables a certain amount of
 * parallelism. */

struct HttpClientPool
{
    HttpClientPool(const std::string & baseUrl, size_t numClients = 8) noexcept;

    void registerClients(MessageLoop & loop);
    void unregisterClients(MessageLoop & loop);

    bool get(const std::string & resource,
             HttpClientCallbacks & callbacks,
             const RestParams & queryParams = RestParams(),
             const RestParams & headers = RestParams(),
             int timeout = -1);

    bool put(const std::string & resource,
             HttpClientCallbacks & callbacks,
             const HttpRequest::Content & content = HttpRequest::Content(),
             const RestParams & queryParams = RestParams(),
             const RestParams & headers = RestParams(),
             int timeout = -1);

    bool post(const std::string & resource,
              HttpClientCallbacks & callbacks,
              const HttpRequest::Content & content = HttpRequest::Content(),
              const RestParams & queryParams = RestParams(),
              const RestParams & headers = RestParams(),
              int timeout = -1);

private:
    std::vector<HttpClient> clients_;

    size_t getNextClientNbr();
    size_t nextClientNbr_;
};

} // namespace Datacratic
