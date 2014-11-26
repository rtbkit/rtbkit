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
TcpClient(const OnClosed & onClosed, const OnReceivedData & onReceivedData,
          const OnException & onException,
          size_t maxMessages, size_t recvBufSize)
    : AsyncWriterSource(onClosed, onReceivedData, onException,
                        maxMessages, recvBufSize),
      port_(-1),
      state_(TcpClientState::Disconnected),
      noNagle_(false)
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
init(const string & hostname, int port)
{
    if (state_ == TcpClientState::Connecting
        || state_ == TcpClientState::Connected) {
        throw ML::Exception("connection already pending or established");
    }
    if (hostname.empty()) {
        throw ML::Exception("hostname is empty");
    }
    if (port < 1) {
        throw ML::Exception("invalid port: " + to_string(port));
    }
    hostname_ = hostname;
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
connect(const OnConnectionResult & onConnectionResult)
{
    // cerr << "connect...\n";
    ExcCheck(getFd() == -1, "socket is not closed");
    ExcCheck(!hostname_.empty(), "no hostname set");

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

    /* host resolution */
    struct sockaddr_in addr;
    addr.sin_port = htons(port_);
    addr.sin_family = AF_INET;

    // cerr << " connecting to host: " + hostname_ + "\n";
    res = ::inet_aton(hostname_.c_str(), &addr.sin_addr);
    if (res == 0) {
        // cerr << "hostname is not an ip\n";
        struct hostent hostentry;
        struct hostent * hostentryP;
        int hErrnoP;

        char buffer[1024];
        res = gethostbyname_r(hostname_.c_str(),
                              &hostentry,
                              buffer, sizeof(buffer),
                              &hostentryP, &hErrnoP);
        if (res == -1 || hostentry.h_addr_list == nullptr) {
            onConnectionResult(TcpConnectionCode::HostUnknown);
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
            onConnectionResult(TcpConnectionCode::ConnectionFailure);
            return;
        }
        handleConnectionEventCb_ = [=] (const ::epoll_event & event) {
            this->handleConnectionEvent(socketFd, onConnectionResult);
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
        onConnectionResult(TcpConnectionCode::Success);
        state_ = TcpClientState::Connected;
    }
    ML::futex_wake(state_);

    /* no cleanup required */
    success = true;
}

void
TcpClient::
handleConnectionEvent(int socketFd, OnConnectionResult onConnectionResult)
{
    // cerr << "handle connection result\n";
    int32_t result;
    socklen_t len(sizeof(result));
    int res = getsockopt(socketFd, SOL_SOCKET, SO_ERROR,
                         (void *) &result, &len);
    if (res == -1) {
        throw ML::Exception(errno, "getsockopt");
    }

    TcpConnectionCode connCode;
    vector<string> lostMessages;
    if (result == 0) {
        connCode = Success;
    }
    else if (result == ENETUNREACH) {
        connCode = HostUnknown;
    }
    else if (result == ECONNREFUSED
             || result == EHOSTDOWN
             || result == EHOSTUNREACH) {
        connCode = ConnectionFailure;
    }
    else {
        throw ML::Exception("unhandled error:" + to_string(result));
    }

    removeFd(socketFd);
    unregisterFdCallback(socketFd, true);
    if (connCode == Success) {
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

    TcpConnectionResult connResult(connCode, move(lostMessages));
    onConnectionResult(move(connResult));
}
