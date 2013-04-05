/* Wolfgang Sourdeau - April 2013 */

#include <memory>
#include <string>

#include "soa/service/service_base.h"
#include "soa/service/epoller.h"
#include "jml/arch/spinlock.h"
#include "jml/utils/ring_buffer.h"


namespace Datacratic {

struct CharRingBuffer {
    CharRingBuffer(size_t sizePower)
    : bufferSize(1 << sizePower),
      buffer(new char[bufferSize]),
      bufferMask(bufferSize - 1),
      readPosition(0), writePosition(0)
    {
    }

    ~CharRingBuffer()
    {
        delete[] buffer;
    }

    size_t bufferSize;
    char *buffer;
    size_t bufferMask;
    int readPosition;
    int writePosition;

#if 0
    typedef boost::lock_guard<ML::Spinlock> Guard;
    mutable ML::Spinlock lock;
    // typedef std::mutex Lock;
    // typedef std::unique_lock<Lock> Guard;
    // mutable Lock lock;
#endif

    ssize_t availableForWriting() const;
    ssize_t availableForReading() const;

    void write(const char *newBytes, size_t len);
    void read(char *bytes, size_t len, bool peek = false);
};

struct FullPoller: public AsyncEventSource {
    void init();

    void addFd(int fd, void * data = 0);
    
    virtual bool handleEvent(epoll_event & event) = 0;

    int handleEvents();
    virtual bool poll() const;

    virtual bool processOne()
    {
        handleEvents();
        return poll();
    }

    virtual int selectFd() const
    {
        return -1;
    }
    
    int epollSocket_;
};

struct TcpNamedEndpoint : public NamedEndpoint, public FullPoller {
    TcpNamedEndpoint();
    ~TcpNamedEndpoint();

    typedef std::function<void (const std::string &)> MessageHandler;

    void init(std::shared_ptr<ConfigurationService> config,
              const std::string & endpointName);
    void shutdown();

    void onConnect(int newFd);
    void bindTcp();

    void onDisconnect(int fd);

    bool handleEvent(epoll_event & event);
    void flushMessages();

    int socket_;
    MessageHandler onMessage_;

    CharRingBuffer recvBuffer;
    CharRingBuffer sendBuffer;
};

struct TcpNamedProxy: public FullPoller {
    enum ConnectionState {
        LOOKUP,
        DISCONNECTED,
        CONNECTING,
        CONNECTED
    };

    TcpNamedProxy();
    ~TcpNamedProxy();

    void init(std::shared_ptr<ConfigurationService> config);
    void shutdown();

    void connectTo(std::string host, int port);
    bool isConnected() const;
    void sendMessage(const std::string & message);

    void onMessage(std::string && newMessage);
    void onDisconnect(int fd);

    bool handleEvent(epoll_event & event);

    int socket_;
    enum ConnectionState state_;

    CharRingBuffer recvBuffer;
    CharRingBuffer sendBuffer;
};

} // namespace Datacratic
