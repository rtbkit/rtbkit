/* tcp_socket.cc
   Wolfgang Sourdeau, April 2014
   Copyright (c) 2014 Datacratic.  All rights reserved.

   A helper base class for handling tcp connections.
*/

#include <netdb.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/epoll.h>
#include <sys/types.h>
#include <sys/socket.h>

#include "googleurl/src/gurl.h"
#include "jml/utils/exc_assert.h"
#include "jml/utils/guard.h"
#include "soa/types/url.h"

#include "tcp_client.h"

using namespace std;
using namespace Datacratic;

TcpClient::
TcpClient(OnConnectionResult onConnectionResult,
          OnClosed onClosed,
          OnWriteResult onWriteResult,
          OnReceivedData onReceivedData,
          OnException onException,
          size_t maxMessages,
          size_t recvBufSize)
    : AsyncWriterSource(onClosed, onWriteResult, onReceivedData,
                        onException, maxMessages, recvBufSize),
      port_(-1),
      state_(TcpClientState::Disconnected),
      noNagle_(false),
      onConnectionResult_(onConnectionResult)
{
}

TcpClient::
~TcpClient()
{
}

void
TcpClient::
init(const string & url)
{
    init(Url(url));
}

void
TcpClient::
init(const Url & url)
{
    int port = url.url->EffectiveIntPort();
    init(url.host(), port);
}

void
TcpClient::
init(const string & address, int port)
{
    if (state_ == TcpClientState::Connecting
        || state_ == TcpClientState::Connected) {
        throw ML::Exception("connection already pending or established");
    }
    if (address.empty()) {
        throw ML::Exception("invalid address: " + address);
    }
    if (port < 1) {
        throw ML::Exception("invalid port: " + to_string(port));
    }
    address_ = address;
    port_ = port;
}

void
TcpClient::
setUseNagle(bool useNagle)
{
    if (state() != Disconnected) {
        throw ML::Exception("socket already created");
    }

    noNagle_ = !useNagle;
}

void
TcpClient::
connect()
{
    // cerr << "connect...\n";
    ExcCheck(state() == Disconnected, "socket is not closed");
    ExcCheck(!address_.empty(), "no address set");

    state_ = TcpClientState::Connecting;
    ML::futex_wake(state_);

    int res = ::socket(AF_INET,
                       SOCK_STREAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0);
    if (res == -1) {
        state_ = TcpClientState::Disconnected;
        ML::futex_wake(state_);
        throw ML::Exception(errno, "socket");
    }

    int socketFd = res;
    // cerr << "socket created\n";

    /* cleanup */
    bool success(false);
    auto cleanup = [&] () {
        if (!success) {
            ::close(socketFd);
            state_ = TcpClientState::Disconnected;
            ML::futex_wake(state_);
        }
    };
    ML::Call_Guard guard(cleanup);

    /* nagle */
    if (noNagle_) {
        int flag = 1;
        res = setsockopt(socketFd,
                         IPPROTO_TCP, TCP_NODELAY,
                         (char *) &flag, sizeof(int));
        if (res == -1) {
            throw ML::Exception(errno, "setsockopt TCP_NODELAY");
        }
    }

    /* address resolution */
    struct sockaddr_in addr;
    addr.sin_port = htons(port_);
    addr.sin_family = AF_INET;

    // cerr << " connecting to host: " + address_ + "\n";
    res = ::inet_aton(address_.c_str(), &addr.sin_addr);
    if (res == 0) {
        // cerr << "host is not an ip\n";
        struct hostent hostentry;
        struct hostent * hostentryP;
        int hErrnoP;

        char buffer[1024];
        res = gethostbyname_r(address_.c_str(),
                              &hostentry,
                              buffer, sizeof(buffer),
                              &hostentryP, &hErrnoP);
        if (res == -1 || hostentry.h_addr_list == nullptr) {
            onConnectionResult(ConnectionResult::HostUnknown, {});
            return;
        }
        addr.sin_family = hostentry.h_addrtype;
        addr.sin_addr.s_addr = *(in_addr_t *) hostentry.h_addr_list[0];
    }

    /* connection */
    res = ::connect(socketFd,
                    (const struct sockaddr *) &addr, sizeof(sockaddr_in));
    if (res == -1) {
        if (errno != EINPROGRESS) {
            onConnectionResult(ConnectionResult::ConnectionFailure,
                               {});
            return;
        }
        handleConnectionEventCb_
            = [&, socketFd] (const ::epoll_event & event) {
            this->handleConnectionEvent(socketFd, event);
        };
        registerFdCallback(socketFd, handleConnectionEventCb_);
        addFdOneShot(socketFd, false, true);
        enableQueue();
        state_ = TcpClientState::Connecting;
        // cerr << "connection in progress\n";
    }
    else {
        // cerr << "connection established\n";
        setFd(socketFd);
        onConnectionResult(ConnectionResult::Success, {});
        state_ = TcpClientState::Connected;
    }
    ML::futex_wake(state_);

    /* no cleanup required */
    success = true;
}

void
TcpClient::
handleConnectionEvent(int socketFd, const ::epoll_event & event)
{
    // cerr << "handle connection result\n";
    int32_t result;
    socklen_t len(sizeof(result));
    int res = getsockopt(socketFd, SOL_SOCKET, SO_ERROR,
                         (void *) &result, &len);
    if (res == -1) {
        throw ML::Exception(errno, "getsockopt");
    }

    ConnectionResult connResult;
    if (result == 0) {
        connResult = Success;
    }
    else if (result == ENETUNREACH) {
        connResult = HostUnknown;
    }
    else if (result == ECONNREFUSED
             || result == EHOSTDOWN
             || result == EHOSTUNREACH) {
        connResult = ConnectionFailure;
    }
    else {
        throw ML::Exception("unhandled error:" + to_string(result));
    }

    vector<string> lostMessages;
    removeFd(socketFd);
    unregisterFdCallback(socketFd);
    if (connResult == Success) {
        errno = 0;
        setFd(socketFd);
        // cerr << "connection successful\n";
        state_ = TcpClientState::Connected;
    }
    else {
        disableQueue();
        ::close(socketFd);
        state_ = TcpClientState::Disconnected;
        lostMessages = emptyMessageQueue();
    }
    ML::futex_wake(state_);
    onConnectionResult(connResult, lostMessages);
}

void
TcpClient::
onConnectionResult(ConnectionResult result, const vector<string> & msgs)
{
    if (onConnectionResult_) {
        onConnectionResult_(result, msgs);
    }
}
