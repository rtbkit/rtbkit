/* http_client_v2.cc
   Wolfgang Sourdeau, April 2014
   Copyright (c) 2014 Datacratic.  All rights reserved.
*/

#include <errno.h>
#include <sys/timerfd.h>

#include "jml/arch/exception.h"
#include "jml/utils/exc_assert.h"

#include "soa/types/url.h"
#include "message_loop.h"
#include "http_header.h"
#include "http_parsers.h"

#include "http_client_v2.h"


using namespace std;
using namespace Datacratic;


namespace {

HttpClientError
translateError(TcpConnectionCode code)
{
    HttpClientError error;

    switch (code) {
    case Success:
        error = HttpClientError::None;
        break;
    case Timeout:
        error = HttpClientError::Timeout;
        break;
    case HostUnknown:
        error = HttpClientError::HostNotFound;
        break;
    case ConnectionFailure:
        error = HttpClientError::CouldNotConnect;
        break;
    case ConnectionEnded:
        error = HttpClientError::Unknown;
        break;
    default:
        ::fprintf(stderr, "returning 'unknown' for code %d\n", code);
        error = HttpClientError::Unknown;
    }

    return error;
}

bool getExpectResponseBody(const HttpRequest & request)
{
    return (request.verb_ != "HEAD");
}

string
makeRequestStr(const HttpRequest & request)
{
    string requestStr;
    requestStr.reserve(10000);

    Url url(request.url_);
    requestStr = request.verb_ + " " + url.path();
    string query = url.query();
    if (query.size() > 0) {
        requestStr += "?" + query;
    }
    requestStr += " HTTP/1.1\r\n";
    requestStr += "Host: "+ url.host();
    int port = url.port();
    if (port > 0) {
        requestStr += ":" + to_string(port);
    }
    requestStr += "\r\nAccept: */*\r\n";
    for (const auto & header: request.headers_) {
        requestStr += header.first + ":" + header.second + "\r\n";
    }
    const auto & content = request.content_;
    if (!content.str.empty()) {
        requestStr += ("Content-Length: "
                       + to_string(content.str.size()) + "\r\n");
        requestStr += "Content-Type: " + content.contentType + "\r\n";
    }
    requestStr += "\r\n";

    return requestStr;
}

} // file scope


/****************************************************************************/
/* HTTP CONNECTION                                                          */
/****************************************************************************/

HttpConnection::
HttpConnection()
    : responseState_(IDLE), requestEnded_(false), lastCode_(Success),
      timeoutFd_(-1)
{
    // cerr << "HttpConnection(): " << this << "\n";

    /* Apart with pipelining, there is no real interest in using the Nagle
       algorithm with HTTP, since we will want to send everything in one shot
       as soon as possible. */
    setUseNagle(false);

    parser_.onResponseStart = [&] (const string & httpVersion,
                                   int code) {
        this->onParserResponseStart(httpVersion, code);
    };
    parser_.onHeader = [&] (const char * data, size_t size) {
        this->onParserHeader(data, size);
    };
    parser_.onData = [&] (const char * data, size_t size) {
        this->onParserData(data, size);
    };
    parser_.onDone = [&] (bool doClose) {
        this->onParserDone(doClose);
    };
}

HttpConnection::
~HttpConnection()
{
    // cerr << "~HttpConnection: " << this << "\n";
    cancelRequestTimer();
    if (responseState_ != IDLE) {
        ::fprintf(stderr,
                  "destroying non-idle connection: %d",
                  responseState_);
        abort();
    }
}

void
HttpConnection::
clear()
{
    responseState_ = IDLE;
    requestEnded_ = false;
    request_.clear();
    lastCode_ = Success;
}

void
HttpConnection::
perform(HttpRequest && request)
{
    // cerr << "perform: " << this << endl;

    if (responseState_ != IDLE) {
        throw ML::Exception("%p: cannot process a request when state is not"
                            " idle: %d", this, responseState_);
    }

    request_ = move(request);

    if (queueEnabled()) {
        startSendingRequest();
    }
    else {
        auto onConnectionResult = [&] (TcpConnectionResult result) {
            if (result.code == TcpConnectionCode::Success) {
                startSendingRequest();
            }
            else {
                handleEndOfRq(result.code, false);
            }
        };
        connect(onConnectionResult);
    }
}

void
HttpConnection::
startSendingRequest()
{
    /* This controls the maximum body size from which the body will be written
       separately from the request headers. This tend to improve performance
       by removing a potential allocation and a large copy. 65536 appears to
       be a reasonable value on my installation, but this would need to be
       tested on different setups. */
    static constexpr size_t TwoStepsThreshold(65536);

    parser_.setExpectBody(getExpectResponseBody(request_));
    string rqData = makeRequestStr(request_);

    bool twoSteps(false);

    const HttpRequest::Content & content = request_.content_;
    if (content.str.size() > 0) {
        if (content.str.size() < TwoStepsThreshold) {
            rqData.append(content.str);
        }
        else {
            twoSteps = true;
        }
    }
    responseState_ = PENDING;

    auto onWriteResultFinal = [&] (AsyncWriteResult result) {
        if (result.error == 0) {
            ExcAssertEqual(responseState_, PENDING);
            responseState_ = IDLE;
        }
        else {
            throw ML::Exception("unhandled error");
        }
    };

    if (twoSteps) {
        auto onWriteResultFirst
            = [this, onWriteResultFinal] (AsyncWriteResult result) {
            if (result.error == 0) {
                ExcAssertEqual(responseState_, PENDING);
                const HttpRequest::Content & content = request_.content_;
                write(move(content.str), onWriteResultFinal);
            }
            else {
                throw ML::Exception("unhandled error");
            }
        };
        write(move(rqData), onWriteResultFirst);
    }
    else {
        write(move(rqData), onWriteResultFinal);
    }

    armRequestTimer();
}

void
HttpConnection::
onReceivedData(const char * data, size_t size)
{
    // cerr << "onReceivedData: " + string(data, size) + "\n";
    parser_.feed(data, size);
}

void
HttpConnection::
onException(const exception_ptr & excPtr)
{
    cerr << "http client received exception\n";
    abort();
}

void
HttpConnection::
onParserResponseStart(const string & httpVersion, int code)
{
    // ::fprintf(stderr, "%p: onParserResponseStart\n", this);
    request_.callbacks_->onResponseStart(request_, httpVersion, code);
}

void
HttpConnection::
onParserHeader(const char * data, size_t size)
{
    // cerr << "onParserHeader: " << this << endl;
    request_.callbacks_->onHeader(request_, data, size);
}

void
HttpConnection::
onParserData(const char * data, size_t size)
{
    // cerr << "onParserData: " << this << endl;
    request_.callbacks_->onData(request_, data, size);
}

void
HttpConnection::
onParserDone(bool doClose)
{
    handleEndOfRq(Success, doClose);
}

/* This method handles end of requests: callback invocation, timer
 * cancellation etc. It may request the closing of the connection, in which
 * case the HttpConnection will be ready for a new request only after
 * finalizeEndOfRq is invoked. */
void
HttpConnection::
handleEndOfRq(TcpConnectionCode code, bool requireClose)
{
    if (requestEnded_) {
        // cerr << "ignoring extraneous end of request\n";
        ;
    }
    else {
        requestEnded_ = true;
        cancelRequestTimer();
        if (requireClose) {
            lastCode_ = code;
            requestClose();
        }
        else {
            finalizeEndOfRq(code);
        }
    }
}

void
HttpConnection::
finalizeEndOfRq(TcpConnectionCode code)
{
    request_.callbacks_->onDone(request_, translateError(code));
    clear();
    onDone(code);
}

void
HttpConnection::
onClosed(bool fromPeer, const std::vector<std::string> & msgs)
{
    if (fromPeer) {
        handleEndOfRq(ConnectionEnded, false);
    }
    else {
        finalizeEndOfRq(lastCode_);
    }
}

void
HttpConnection::
armRequestTimer()
{
    if (request_.timeout_ > 0) {
        if (timeoutFd_ == -1) {
            timeoutFd_ = timerfd_create(CLOCK_MONOTONIC,
                                        TFD_NONBLOCK | TFD_CLOEXEC);
            if (timeoutFd_ == -1) {
                throw ML::Exception(errno, "timerfd_create");
            }
            auto handleTimeoutEventCb = [&] (const struct epoll_event & event) {
                this->handleTimeoutEvent(event);
            };
            registerFdCallback(timeoutFd_, handleTimeoutEventCb);
            // cerr << " timeoutFd_: "  + to_string(timeoutFd_) + "\n";
            addFdOneShot(timeoutFd_, true, false);
            // cerr << "timer armed\n";
        }
        else {
            // cerr << "timer rearmed\n";
            modifyFdOneShot(timeoutFd_, true, false);
        }

        itimerspec spec;
        ::memset(&spec, 0, sizeof(itimerspec));

        spec.it_interval.tv_sec = 0;
        spec.it_value.tv_sec = request_.timeout_;
        int res = timerfd_settime(timeoutFd_, 0, &spec, nullptr);
        if (res == -1) {
            throw ML::Exception(errno, "timerfd_settime");
        }
    }
}

void
HttpConnection::
cancelRequestTimer()
{
    // cerr << "cancel request timer " << this << "\n";
    if (timeoutFd_ != -1) {
        // cerr << "  was active\n";
        removeFd(timeoutFd_);
        unregisterFdCallback(timeoutFd_, true);
        ::close(timeoutFd_);
        timeoutFd_ = -1;
    }
    // else {
    //     cerr << "  was not active\n";
    // }
}

void
HttpConnection::
handleTimeoutEvent(const ::epoll_event & event)
{
    if (timeoutFd_ == -1) {
        return;
    }

    if ((event.events & EPOLLIN) != 0) {
        while (true) {
            uint64_t expiries;
            int res = ::read(timeoutFd_, &expiries, sizeof(expiries));
            if (res == -1) {
                if (errno == EAGAIN) {
                    break;
                }

                throw ML::Exception(errno, "read");
            }
        }
        handleEndOfRq(Timeout, true);
    }
}


/****************************************************************************/
/* HTTP CLIENT V2                                                           */
/****************************************************************************/

HttpClientV2::
HttpClientV2(const string & baseUrl, int numParallel, size_t queueSize)
    : HttpClientImpl(baseUrl, numParallel, queueSize),
      loop_(1, 0, -1),
      baseUrl_(baseUrl),
      avlConnections_(numParallel),
      nextAvail_(0),
      queue_([&]() { this->handleQueueEvent(); return false; }, queueSize)
{
    ExcAssert(baseUrl.compare(0, 8, "https://") != 0);

    /* available connections */
    for (size_t i = 0; i < numParallel; i++) {
        HttpConnection * connPtr = new HttpConnection();
        shared_ptr<HttpConnection> connection(connPtr);
        connection->init(baseUrl);
        connection->onDone = [&, connPtr] (TcpConnectionCode result) {
            handleHttpConnectionDone(connPtr, result);
        };
        loop_.addSource("connection" + to_string(i), connection);
        avlConnections_[i] = connPtr;
    }
    loop_.addSource("queue", queue_);
}

HttpClientV2::
~HttpClientV2()
{
    // cerr << "~HttpClient: " << this << "\n";
}

int
HttpClientV2::
selectFd()
    const
{
    return loop_.selectFd();
}

bool
HttpClientV2::
processOne()
{
    return loop_.processOne();
}

void
HttpClientV2::
enableDebug(bool value)
{
    debug(value);
}

void
HttpClientV2::
enableSSLChecks(bool value)
{
}

void
HttpClientV2::
enableTcpNoDelay(bool value)
{
}

void
HttpClientV2::
enablePipelining(bool value)
{
    if (value) {
        throw ML::Exception("pipeline is not supported");
    }
}

bool
HttpClientV2::
enqueueRequest(const string & verb, const string & resource,
               const shared_ptr<HttpClientCallbacks> & callbacks,
               const HttpRequest::Content & content,
               const RestParams & queryParams, const RestParams & headers,
               int timeout)
{
    string url = baseUrl_ + resource + queryParams.uriEscaped();
    HttpRequest request(verb, url, callbacks, content, headers, timeout);

    return queue_.push_back(std::move(request));
}

void
HttpClientV2::
handleQueueEvent()
{
    size_t numConnections = avlConnections_.size() - nextAvail_;
    if (numConnections > 0) {
        /* "0" has a special meaning for pop_front and must be avoided here */
        auto requests = queue_.pop_front(numConnections);
        for (auto request: requests) {
            HttpConnection * conn = getConnection();
            if (!conn) {
                cerr << ("nextAvail_: "  + to_string(nextAvail_)
                         + "; num conn: "  + to_string(numConnections)
                         + "; num reqs: "  + to_string(requests.size())
                         + "\n");
                throw ML::Exception("inconsistency in count of available"
                                    " connections");
            }
            conn->perform(move(request));
        }
    }
}

void
HttpClientV2::
handleHttpConnectionDone(HttpConnection * connection,
                         TcpConnectionCode result)
{
    auto requests = queue_.pop_front(1);
    if (requests.size() > 0) {
        // cerr << "emptying queue...\n";
        connection->perform(move(requests[0]));
    }
    else {
        releaseConnection(connection);
    }
}

HttpConnection *
HttpClientV2::
getConnection()
{
    HttpConnection * conn;

    if (nextAvail_ < avlConnections_.size()) {
        conn = avlConnections_[nextAvail_];
        nextAvail_++;
    }
    else {
        conn = nullptr;
    }

    // cerr << " returning conn: " << conn << "\n";

    return conn;
}

void
HttpClientV2::
releaseConnection(HttpConnection * oldConnection)
{
    if (nextAvail_ > 0) {
        nextAvail_--;
        avlConnections_[nextAvail_] = oldConnection;
    }
}
