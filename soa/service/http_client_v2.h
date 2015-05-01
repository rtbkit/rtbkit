/* http_client_v2.h                                                -*- C++ -*-
   Wolfgang Sourdeau, April 2014
   Copyright (c) 2014 Datacratic.  All rights reserved.

   V2 of the HTTP client. Based on in-house tcp library.
   This model uses TcpClient as transport class and HttpParser for parsing
   requests. Faster than v1 and than curl::Easy in the thread-model.
*/

#pragma once

/* TODO:
   nice to have:
   - connect timeout
   - activity timeout
   - parser:
     - needs better validation (header key size, ...)
   - compression
   - auto disconnect (keep-alive)
   - SSL support
   - pipelining
 */

#include <string>
#include <vector>

#include "soa/jsoncpp/value.h"
#include "soa/service/http_client.h"
#include "soa/service/http_header.h"
#include "soa/service/http_parsers.h"
#include "soa/service/message_loop.h"
#include "soa/service/typed_message_channel.h"
#include "soa/service/tcp_client.h"


namespace Datacratic {

/****************************************************************************/
/* HTTP CONNECTION                                                          */
/****************************************************************************/

struct HttpConnection : TcpClient {
    typedef std::function<void (TcpConnectionCode)> OnDone;

    enum HttpState {
        IDLE,
        PENDING
    };

    HttpConnection();

    HttpConnection(const HttpConnection & other) = delete;

    ~HttpConnection();

    void clear();
    void perform(HttpRequest && request);

    const HttpRequest & request() const
    {
        return request_;
    }

    OnDone onDone;

private:
    /* tcp_socket overrides */
    virtual void onClosed(bool fromPeer,
                          const std::vector<std::string> & msgs);
    virtual void onReceivedData(const char * data, size_t size);
    virtual void onException(const std::exception_ptr & excPtr);

    void onParserResponseStart(const std::string & httpVersion, int code);
    void onParserHeader(const char * data, size_t size);
    void onParserData(const char * data, size_t size);
    void onParserDone(bool onClose);

    void startSendingRequest();

    void handleEndOfRq(TcpConnectionCode code, bool requireClose);
    void finalizeEndOfRq(TcpConnectionCode code);

    HttpResponseParser parser_;

    HttpState responseState_;
    HttpRequest request_;
    bool requestEnded_;

    /* Connection: close */
    TcpConnectionCode lastCode_;

    /* request timeouts */
    void armRequestTimer();
    void cancelRequestTimer();
    void handleTimeoutEvent(const ::epoll_event & event);

    int timeoutFd_;
};


/****************************************************************************/
/* HTTP CLIENT V2                                                           */
/****************************************************************************/

struct HttpClientV2 : public HttpClientImpl {
    HttpClientV2(const std::string & baseUrl,
                 int numParallel, size_t queueSize);

    ~HttpClientV2();

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

    size_t queuedRequests()
        const
    {
        return queue_.size();
    }

    HttpClient & operator = (HttpClient && other) = delete;
    HttpClient & operator = (const HttpClient & other) = delete;

private:
    void handleQueueEvent();

    void handleHttpConnectionDone(HttpConnection * connection,
                                  TcpConnectionCode result);

    HttpConnection * getConnection();
    void releaseConnection(HttpConnection * connection);

    MessageLoop loop_;

    std::string baseUrl_;

    std::vector<HttpConnection *> avlConnections_;
    size_t nextAvail_;

    TypedMessageQueue<HttpRequest> queue_; /* queued requests */

    HttpConnection::OnDone onHttpConnectionDone_;
};

} // namespace Datacratic
