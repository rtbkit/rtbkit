/* http_client_v1.h                                                -*- C++ -*-
   Wolfgang Sourdeau, January 2014
   Copyright (c) 2014 Datacratic.  All rights reserved.

   V1 of the HTTP client, based on libcurl:
   - has support for https
   - slow
*/

#pragma once

#include "sys/epoll.h"

#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <vector>

#include <curlpp/Easy.hpp>
#include <curlpp/Multi.hpp>
#include <curlpp/Types.hpp>

#include "jml/arch/wakeup_fd.h"
#include "soa/service/async_event_source.h"
#include "soa/service/http_header.h"
#include "soa/service/http_client.h"


namespace Datacratic {

/****************************************************************************/
/* HTTP CLIENT V1                                                           */
/****************************************************************************/

struct HttpClientV1 : public HttpClientImpl {
    HttpClientV1(const std::string & baseUrl,
                 int numParallel, int queueSize);

    ~HttpClientV1();

    /* AsyncEventSource */
    virtual int selectFd() const;
    virtual bool processOne();

    /* HttpClientImpl */
    void enableDebug(bool value);
    void enableSSLChecks(bool value);
    void enableTcpNoDelay(bool value);
    void enablePipelining(bool value);

    bool enqueueRequest(const std::string & verb,
                        const std::string & resource,
                        const std::shared_ptr<HttpClientCallbacks> & callbacks,
                        const HttpRequest::Content & content,
                        const RestParams & queryParams,
                        const RestParams & headers,
                        int timeout = -1);

    size_t queuedRequests() const;

private:
    void cleanupFds() noexcept;

    /* Local */
    std::vector<std::shared_ptr<HttpRequest>> popRequests(size_t number);

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
            request_.reset();
            afterContinue_ = false;
            uploadOffset_ = 0;
        }
        void perform(bool noSSLChecks, bool tcpNoDelay, bool debug);

        /* header and body write callbacks */
        curlpp::types::WriteFunctionFunctor onHeader_;
        curlpp::types::WriteFunctionFunctor onWrite_;
        size_t onCurlHeader(const char * data, size_t size) noexcept;
        size_t onCurlWrite(const char * data, size_t size) noexcept;

        /* body read callback */
        curlpp::types::ReadFunctionFunctor onRead_;
        size_t onCurlRead(char * buffer, size_t bufferSize) noexcept;

        std::shared_ptr<HttpRequest> request_;

        curlpp::Easy easy_;
        // HttpClientResponse response_;
        bool afterContinue_;
        size_t uploadOffset_;

        struct HttpConnection *next;
    };

    HttpConnection * getConnection();
    void releaseConnection(HttpConnection * connection);

    std::string baseUrl_;
    bool expect100Continue_;
    bool tcpNoDelay_;
    bool noSSLChecks_;

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
    mutable Mutex queueLock_;
    std::queue<std::shared_ptr<HttpRequest>> queue_; /* queued requests */
};

} // namespace Datacratic
