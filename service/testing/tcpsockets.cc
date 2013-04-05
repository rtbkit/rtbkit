/* Wolfgang Sourdeau - April 2013 */

#include <fcntl.h>
#include <unistd.h>
#include <netinet/tcp.h>
#include <sys/epoll.h>
#include <sys/types.h>
#include <sys/socket.h>

#include "jml/arch/atomic_ops.h"
#include "soa/service/rest_service_endpoint.h"

#include "tcpsockets.h"


using namespace std;
using namespace Datacratic;


/* CHARRINGBUFFER */
ssize_t
CharRingBuffer::
availableForWriting()
    const
{
    ML::memory_barrier();

    ssize_t available(readPosition - writePosition);
    if (writePosition >= readPosition) {
        available += bufferSize;
    }
    
    return available;
}

ssize_t
CharRingBuffer::
availableForReading()
    const
{
    ML::memory_barrier();

    ssize_t available(writePosition - readPosition);
    if (writePosition < readPosition) {
        available += bufferSize - 1;
    }
    
    return available;
}

void
CharRingBuffer::
write(const char *newBytes, size_t len)
{
    ML::memory_barrier();

    size_t maxLen = availableForWriting();
    if (len > maxLen)
        throw ML::Exception("no room left");

    int bytesLeftRight = bufferSize - writePosition;
    int bytesCopied;
    if (len > bytesLeftRight) {
        memcpy(buffer + writePosition, newBytes, bytesLeftRight);
        len -= bytesLeftRight;
        bytesCopied = bytesLeftRight;
        writePosition = 0;
    }
    else {
        bytesCopied = 0;
    }
    memcpy(buffer + writePosition, newBytes + bytesCopied, len);

    writePosition = (writePosition + len) & bufferMask;
}

void
CharRingBuffer::
read(char *bytes, size_t len, bool peek)
{
    ML::memory_barrier();

    size_t maxLen = availableForReading();
    if (len > maxLen)
        throw ML::Exception("nothing left to read");

    int bytesLeftRight = bufferSize - readPosition;

    if (len > bytesLeftRight) {
        memcpy(bytes, buffer + readPosition, bytesLeftRight);
        memcpy(bytes + bytesLeftRight, buffer, len - bytesLeftRight);
    }
    else {
        memcpy(bytes, buffer + readPosition, len);
    }

    if (!peek) {
        readPosition = (readPosition + len) & bufferMask;
    }
}

/* FULLPOLLER */
void
FullPoller::
init()
{
    epollSocket_ = epoll_create(1);
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

int
FullPoller::
handleEvents()
{
    for (;;) {
        epoll_event events[64];
        memset(events, 0, sizeof(events));

        int res = epoll_wait(epollSocket_, events, 64, 0);
        if (res == 0) return 0;

        // cerr << this << ": events to handle: " << res << endl;

        // sys call interrupt
        if (res == -1 && errno == EINTR) continue;
        if (res == -1 && errno == EBADF) {
            cerr << "got bad FD" << endl;
            return -1;
        }

        if (res == -1)
            throw ML::Exception(errno, "epoll_wait");

        for (unsigned i = 0; i < res; i++) {
            // cerr << "handling event on fd: " << events[i].data.fd << endl;
            handleEvent(events[i]);
        }

        return poll();
    }
}

bool
FullPoller::
poll()
    const
{
    struct epoll_event ev;
    return (epoll_wait(epollSocket_, &ev, 1, 0) > 0);
}

/* TCPNAMEDENDPOINT */

TcpNamedEndpoint::
TcpNamedEndpoint()
    : NamedEndpoint(), FullPoller(),
      recvBuffer(20), sendBuffer(20)
{
    needsPoll = true;
}

TcpNamedEndpoint::
~TcpNamedEndpoint()
{
}

void
TcpNamedEndpoint::
init(shared_ptr<ConfigurationService> config,
     const string & endpointName)
{
    NamedEndpoint::init(config, endpointName);
    FullPoller::init();

    socket_ = socket(AF_INET, SOCK_STREAM | SOCK_NONBLOCK, 0);
    cerr << "endpoint socket: " << socket_ << endl;
    addFd(socket_);
}

void
TcpNamedEndpoint::
onDisconnect(int fd)
{
}

bool
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
            cerr << "epoll connected\n";
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
                int rc = read(event.data.fd, buffer, nBytes);
                // cerr << "read returned  " << rc << "\n";
                if (rc == -1) {
                    if (errno == EAGAIN || errno == ENOTCONN)
                        break;
                    cerr << "errno = " << errno << endl;
                    throw ML::Exception(errno, "read", "handleEvent");
                }
                recvBuffer.write(buffer, rc);
                nBytes -= rc;
            }
            flushMessages();
        }
    }
    else if ((event.events & EPOLLHUP) == EPOLLHUP) {
        onDisconnect(event.data.fd);
    }

    return false; /* unused */
}

void
TcpNamedEndpoint::
flushMessages()
{
    /* message handling, should be put in subclass */
    size_t readLen = recvBuffer.availableForReading();
    if (readLen == 0) {
        return;
    }
    // cerr << "processing bytes:" << readLen << endl;
    while (readLen > 0) {
        char msgLen;
        recvBuffer.read(&msgLen, 1, true);
        readLen--;
        if (msgLen > readLen) {
            break;
        }
        /* first byte will contain size */
        char buffer[msgLen+1];
        recvBuffer.read(buffer, 1 + msgLen);
        readLen -= msgLen;

        string message(buffer + 1, msgLen);
        // cerr << "onMessage: " << message << endl;
        onMessage_(message);
    }
}

void
TcpNamedEndpoint::
onConnect(int newFd)
{
    addFd(newFd);
}

void
TcpNamedEndpoint::
bindTcp()
{
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(9876);
    addr.sin_addr.s_addr = INADDR_ANY;

    bind(socket_, (struct sockaddr *)&addr, sizeof(addr));
    listen(socket_, 1024);
}

void
TcpNamedEndpoint::
shutdown()
{
    if (socket_ > -1) {
        // shutdown(socket_, SHUT_RDRW);
        ::close(socket_);
        socket_ = -1;
    }
}

/* TCPNAMEDPROXY */
TcpNamedProxy::
TcpNamedProxy()
    : FullPoller(),
      recvBuffer(20), sendBuffer(20)
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

    socket_ = socket(AF_INET, SOCK_STREAM | SOCK_NONBLOCK, 0);
    // uint32_t nonagle(1);
    // setsockopt(socket_, SOL_TCP, TCP_NODELAY, &nonagle, sizeof(nonagle));

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
    if (rc == 0)
        state_ = CONNECTED;
    else if (rc == -1) {
        if (errno == EINPROGRESS) {
            state_ = CONNECTED; /* should be CONNECTING */
        }
        else {
            throw ML::Exception(errno, "connection failed", "connectTo");
        }
    }
    else {
        throw ML::Exception(errno, "unexpected return code");
    }
}

void
TcpNamedProxy::
shutdown()
{
    if (socket_ > -1) {
        // shutdown(socket_, SHUT_RDRW);
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

void
TcpNamedProxy::
sendMessage(const string & message)
{
    // cerr << "availableForWriting: " << sendBuffer.availableForWriting() << endl;
    char msgSize = message.size();
    ssize_t totalSize = msgSize + 1;
    if (totalSize <= sendBuffer.availableForWriting()) {
        sendBuffer.write(&msgSize, 1);
        sendBuffer.write(message.c_str(), msgSize);
        
        // cerr << "written " << totalSize << " bytes\n";
    }
    else {
        cerr << "sendMessage: no buffer room\n";
    }
}

bool
TcpNamedProxy::
handleEvent(epoll_event & event)
{
    // cerr << "handleEvent: " << event.events << endl;
    if ((event.events & EPOLLIN) == EPOLLIN) {
        cerr << "proxy pollin\n";
    }
    else if ((event.events & EPOLLOUT) == EPOLLOUT) {
        ssize_t nBytes = sendBuffer.availableForReading();
        if (nBytes > 0) {
            // cerr << "available for reading: " << nBytes << endl;
            char buffer[nBytes];
            sendBuffer.read(buffer, nBytes);
            int rc = write(socket_, buffer, nBytes);
            // cerr << "written " << rc << " bytes\n";
            if (rc == -1) {
                throw ML::Exception(errno, "handleEvent", "write");
            }
            if (rc != nBytes) {
                throw ML::Exception("inconsistency");
            }
            // nBytes = sendBuffer.availableForReading();
            // cerr << "available for reading after send: " << nBytes << endl;
        }
    }
    else if ((event.events & EPOLLHUP) == EPOLLHUP) {
        cerr << "proxy pollhup\n";
    }

    return false; /* unused */
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
