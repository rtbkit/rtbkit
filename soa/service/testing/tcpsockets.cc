/* Wolfgang Sourdeau - April 2013 */

#include <fcntl.h>
#include <unistd.h>
#include <netinet/tcp.h>
#include <sys/epoll.h>
#include <sys/types.h>
#include <sys/socket.h>

#include "jml/arch/cmp_xchg.h"
#include "soa/service/rest_service_endpoint.h"

#include "tcpsockets.h"


using namespace std;
using namespace Datacratic;


const int bufferSizePow = 16;
const int tcpBufferSizePow = 13;

/* CHARRINGBUFFER */
size_t
CharRingBuffer::
availableForWriting()
    const
{
    size_t available,
        myReadPosition(readPosition), myWritePosition(writePosition);

    if (myReadPosition <= myWritePosition) {
        available = bufferSize;
    }
    else {
        available = 0;
    }
    available += myReadPosition - myWritePosition - 1;
    
    if (available >= bufferSize) {
        cerr << "writing: bufferSize: " << bufferSize
             << "; readPosition: " << myReadPosition
             << "; writePosition: " << myWritePosition
             << "; available: " << available
             << endl;
        ExcAssert(available < bufferSize);
    }

    return available;
}

size_t
CharRingBuffer::
availableForReading()
    const
{
    size_t available,
        myReadPosition(readPosition), myWritePosition(writePosition);

    if (myWritePosition < myReadPosition) {
        available = bufferSize;
    }
    else {
        available = 0;
    }
    available += myWritePosition - myReadPosition;

    if (available >= bufferSize) {
        cerr << "reading: bufferSize: " << bufferSize
             << "; readPosition: " << myReadPosition
             << "; writePosition: " << myWritePosition
             << "; available: " << available
             << endl;
    }
    ExcAssert(available < bufferSize);
    
    return available;
}

void
CharRingBuffer::
write(const char *newBytes, size_t len)
{
    size_t maxLen = availableForWriting();
    if (len > maxLen)
        throw ML::Exception("no room left");

    size_t myWritePosition(writePosition);
    size_t bytesLeftRight = bufferSize - myWritePosition;
    if (len > bytesLeftRight) {
        memcpy(buffer + myWritePosition, newBytes, bytesLeftRight);
        memcpy(buffer, newBytes + bytesLeftRight, len - bytesLeftRight);
    }
    else {
        memcpy(buffer + myWritePosition, newBytes, len);
    }

    size_t newWritePosition = (myWritePosition + len) & bufferMask;
    if (!ML::cmp_xchg(writePosition, myWritePosition, newWritePosition)) {
        throw ML::Exception("write position changed unexpectedly");
    }
}

void
CharRingBuffer::
read(char *bytes, size_t len, bool peek)
{
    size_t maxLen = availableForReading();
    if (len > maxLen)
        throw ML::Exception("nothing left to read");

    size_t myReadPosition(readPosition);
    size_t bytesLeftRight = bufferSize - myReadPosition;

    if (len > bytesLeftRight) {
        memcpy(bytes, buffer + myReadPosition, bytesLeftRight);
        memcpy(bytes + bytesLeftRight, buffer, len - bytesLeftRight);
    }
    else {
        memcpy(bytes, buffer + myReadPosition, len);
    }

    if (!peek) {
        size_t newReadPosition = (myReadPosition + len) & bufferMask;
        if (!ML::cmp_xchg(readPosition, myReadPosition, newReadPosition)) {
            throw ML::Exception("read position changed unexpectedly");
        }
    }
}

/* CHARMESSAGERINGBUFFER */
bool
CharMessageRingBuffer::
writeMessage(const std::string & newMessage)
{
    bool rc(true);

    char msgSize = newMessage.size();
    ssize_t totalSize = msgSize + 1;
    if (totalSize <= availableForWriting()) {
        write(&msgSize, 1);
        write(newMessage.c_str(), msgSize);
    }
    else {
        // cerr << "writeMessage: no buffer room\n";
        rc = false;
    }

    return rc;
}

bool
CharMessageRingBuffer::
readMessage(std::string & message)
{
    bool rc(true);
    size_t available = availableForReading();

    if (available > 0) {
        char msgLenChar;
        read(&msgLenChar, 1, true);
        size_t msgLen(msgLenChar);
        available--;
        if (msgLen > available) {
            rc = false;
        }
        else {
            /* first byte will contain size */
            char buffer[msgLen+1];
            read(buffer, msgLen + 1);
            // cerr << "msgLen: " << msgLen << endl;
            message = string(buffer + 1, msgLen);
        }
    }
    else {
        rc = false;
    }

    return rc;
}

/* FULLPOLLER */
FullPoller::
FullPoller()
    : epollSocket_(-1), shutdown_(false)
{
}

FullPoller::
~FullPoller()
{
    shutdown();
    disconnect();
}

void
FullPoller::
init()
{
    epollSocket_ = epoll_create(1);
}

void
FullPoller::
shutdown()
{
    shutdown_ = true;
    if (epollSocket_ != -1) {
        ::close(epollSocket_);
        epollSocket_ = -1;
    }
}

void
FullPoller::
addFd(int fd, void *data)
{
    struct epoll_event ev;
    
    ev.events = EPOLLIN | EPOLLOUT | EPOLLHUP;
    ev.data.fd = fd;
    int rc = epoll_ctl(epollSocket_, EPOLL_CTL_ADD, fd, &ev);
    if (rc == -1) {
        throw ML::Exception(errno, "addFd", "error");
    }
    // cerr << "adding socket " << fd << " to epoll set " << this << "\n";
}

void
FullPoller::
removeFd(int fd)
{
    epoll_ctl(epollSocket_, EPOLL_CTL_DEL, fd, NULL);
}

void
FullPoller::
handleEvents()
{
    epoll_event events[64];
    memset(events, 0, sizeof(events));

    int res = epoll_wait(epollSocket_, events, 64, 0);
    if (res == -1) {
        if (errno == EBADF) {
            cerr << "got bad FD" << endl;
            return;
        }
        else if (errno != EINTR)
            throw ML::Exception(errno, "epoll_wait");
    }
    else {
        for (unsigned i = 0; i < res; i++) {
            // cerr << "handling event on fd: " << events[i].data.fd << endl;
            handleEvent(events[i]);
        }
    }
}

bool
FullPoller::
poll()
    const
{
    struct epoll_event ev;
    return !shutdown_ && (epoll_wait(epollSocket_, &ev, 1, 0) > 0);
}

/* TCPNAMEDENDPOINT */

TcpNamedEndpoint::
TcpNamedEndpoint()
    : NamedEndpoint(), FullPoller(),
      recvBuffer(bufferSizePow), sendBuffer(bufferSizePow)
{
    needsPoll = true;
}

TcpNamedEndpoint::
~TcpNamedEndpoint()
{
    shutdown();
}

void
TcpNamedEndpoint::
init(shared_ptr<ConfigurationService> config,
     const string & endpointName)
{
    NamedEndpoint::init(config, endpointName);
    FullPoller::init();

    socket_ = socket(AF_INET, SOCK_STREAM | SOCK_NONBLOCK, 0);
    // cerr << "endpoint socket: " << socket_ << endl;
    uint32_t value = 1 << tcpBufferSizePow;
    setsockopt(socket_, SOL_TCP, SO_SNDBUF, &value, sizeof(value));
    setsockopt(socket_, SOL_TCP, SO_RCVBUF, &value, sizeof(value));

    addFd(socket_);
}

void
TcpNamedEndpoint::
onDisconnect(int fd)
{
}

void
TcpNamedEndpoint::
handleEvent(epoll_event & event)
{
    if (event.data.fd == socket_) {
        if ((event.events & EPOLLIN) == EPOLLIN) {
            int newFd = accept4(socket_, NULL, NULL, SOCK_NONBLOCK);
            if (newFd == -1) {
                throw ML::Exception(errno, "accept", "failure in handleEvent");
            }
#if 0
            int flags = fcntl(newFd, F_GETFL);
            flags |= O_NONBLOCK;
            fcntl(newFd, F_SETFL, &flags);
#endif
            // cerr << "epoll connected\n";
            onConnect(newFd);
        }
        else {
            throw ML::Exception("unhandled");
        }
    }
    else if ((event.events & EPOLLIN) == EPOLLIN) {
        // cerr << "epollin on socket" << event.data.fd << endl;
        /* fill the ring buffer */
        ssize_t nBytes = recvBuffer.availableForWriting();
        if (nBytes > 0) {
            // cerr << "reading " << nBytes << " from client socket\n";
            while (nBytes > 0) {
                char buffer[nBytes];
                int rc = ::read(event.data.fd, buffer, nBytes);
                // cerr << "read returned  " << rc << "\n";
                if (rc == -1) {
                    if (errno == EAGAIN)
                        break;
                    cerr << "errno = " << errno << endl;
                    throw ML::Exception(errno, "read", "handleEvent");
                }
                if (rc == 0)
                    break;
                recvBuffer.write(buffer, rc);
                nBytes -= rc;
            }
            flushMessages();
        }
    }
    else if ((event.events & EPOLLHUP) == EPOLLHUP) {
        onDisconnect(event.data.fd);
    }
}

void
TcpNamedEndpoint::
flushMessages()
{
    string message;
    while (recvBuffer.readMessage(message)) {
        onMessage_(message);
    }
    // cerr << "flushMessages done\n";
}

void
TcpNamedEndpoint::
onConnect(int newFd)
{
    addFd(newFd);
}

void
TcpNamedEndpoint::
bindTcp(int port)
{
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    addr.sin_addr.s_addr = inet_addr("127.0.0.1");

    if (::bind(socket_, (struct sockaddr *)&addr, sizeof(addr)) == -1) {
        throw ML::Exception(errno, "failure", "bind");
    }
    if (::listen(socket_, 1024) == -1) {
        throw ML::Exception(errno, "failure", "listen");
    }
    cerr << "listening\n";
}

void
TcpNamedEndpoint::
shutdown()
{
    FullPoller::shutdown();
    if (socket_ != -1) {
        // shutdown(socket_, SHUT_RDRW);
        ::close(socket_);
        socket_ = -1;
    }
}

/* TCPNAMEDPROXY */
TcpNamedProxy::
TcpNamedProxy()
    : FullPoller(),
      recvBuffer(bufferSizePow), sendBuffer(bufferSizePow)
{
    needsPoll = true;
}

TcpNamedProxy::
~TcpNamedProxy()
{
    shutdown();
}

void
TcpNamedProxy::
init(shared_ptr<ConfigurationService> config)
{
    FullPoller::init();

    socket_ = socket(AF_INET, SOCK_STREAM, 0);
    // uint32_t value(1);
    // setsockopt(socket_, SOL_TCP, TCP_NODELAY, &value, sizeof(value));
    uint32_t value = 1 << tcpBufferSizePow;
    setsockopt(socket_, SOL_TCP, SO_SNDBUF, &value, sizeof(value));
    setsockopt(socket_, SOL_TCP, SO_RCVBUF, &value, sizeof(value));

    addFd(socket_);
}

void
TcpNamedProxy::
connectTo(string host, int port)
{
    struct sockaddr_in addr;

    bzero((char *) &addr, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    addr.sin_addr.s_addr = inet_addr(host.c_str());
    if (addr.sin_addr.s_addr == -1) {
        throw ML::Exception(errno, "lookup failed", "inet_addr");
    }

    int rc = connect(socket_,
                     (const struct sockaddr *) &addr, sizeof(addr));
    cerr << "connect rc: " << rc << endl;
    if (rc == 0) {
        state_ = CONNECTED;
        int flags = fcntl(socket_, F_GETFL);
        flags |= O_NONBLOCK;
        fcntl(socket_, F_SETFL, &flags);
    }
    else if (rc == -1) {
        throw ML::Exception(errno, "connection failed", "connectTo");
    }
    else {
        throw ML::Exception(errno, "unexpected return code");
    }
}

void
TcpNamedProxy::
shutdown()
{
    FullPoller::shutdown();
    if (socket_ != -1) {
        // shutdown(socket_, SHUT_RDRW);
        removeFd(socket_);
        ::close(socket_);
        socket_ = -1;
    }
}

bool
TcpNamedProxy::
isConnected()
    const
{
    return state_ == CONNECTED;
}

bool
TcpNamedProxy::
sendMessage(const string & message)
{
    return sendBuffer.writeMessage(message);
}

void
TcpNamedProxy::
handleEvent(epoll_event & event)
{
    // cerr << "handleEvent: " << event.events << endl;
    if ((event.events & EPOLLIN) == EPOLLIN) {
        // cerr << "proxy pollin\n";
    }
    else if ((event.events & EPOLLOUT) == EPOLLOUT) {
        int nBytes = sendBuffer.availableForReading();
        if (nBytes > 0) {
            // cerr << "available for reading: " << nBytes << endl;
            char buffer[nBytes];
            sendBuffer.read(buffer, nBytes);
            int rc = ::write(socket_, buffer, nBytes);
            if (rc == -1) {
                throw ML::Exception(errno, "handleEvent", "write");
            }
            // ExcAssertEqual(rc, nBytes);
            // nBytes = sendBuffer.availableForReading();
            // cerr << "available for reading after send: " << nBytes << endl;
        }
    }
    else if ((event.events & EPOLLHUP) == EPOLLHUP) {
        cerr << "proxy pollhup\n";
    }
}

void
TcpNamedProxy::
onMessage(string && newMessage)
{
    cerr << "received message: " << newMessage << endl;
}

void
TcpNamedProxy::
onDisconnect(int fd)
{
    cerr << "socket disconnected\n";
}
