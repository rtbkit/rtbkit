/* async_writer_bench */

#include <fcntl.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>

#include <memory>

#include "jml/arch/timers.h"
#include "jml/utils/file_functions.h"
#include "soa/service/message_loop.h"
#include "soa/service/async_writer_source.h"
#include "soa/utils/print_utils.h"

using namespace std;
using namespace Datacratic;


struct WriterSource : public AsyncWriterSource {
    WriterSource(int fd, size_t maxMessages)
        : AsyncWriterSource(nullptr, nullptr, nullptr, maxMessages, 0)
    {
        setFd(fd);
        enableQueue();
    }
};

struct ReaderSource : public AsyncWriterSource {
    ReaderSource(int fd, const OnReceivedData & onReaderData,
                 size_t readBufferSize)
        : AsyncWriterSource(nullptr, onReaderData, nullptr, 0, readBufferSize)
    {
        setFd(fd);
    }
};

/* order is : {writer, reader} */

pair<int, int> makePipePair()
{
    int fds[2];
    if (pipe2(fds, O_NONBLOCK) == -1) {
        throw ML::Exception(errno, "pipe2");
    }

    return {fds[1], fds[0]};
}

pair<int, int> makeUnixSocketPair()
{
    int fds[2];
    socketpair(AF_UNIX, SOCK_STREAM | SOCK_NONBLOCK, 0, fds);
    return {fds[0], fds[1]};
}

pair<int, int> makeTcpSocketPair()
{
    /* listener */
    struct sockaddr_in addr;
    socklen_t addrLen = sizeof(addr);
    addr.sin_port = htons(0);
    addr.sin_family = AF_INET;
    inet_aton("127.0.0.1", &addr.sin_addr);

    int listener = socket(AF_INET, SOCK_STREAM, 0);
    if (listener == -1) {
        throw ML::Exception(errno, "socket");
    }

    if (bind(listener, (const struct sockaddr *) &addr, addrLen) == -1) {
        throw ML::Exception(errno, "bind");
    }

    if (listen(listener, 666) == -1) {
        throw ML::Exception(errno, "listen");
    }

    if (getsockname(listener, (sockaddr *) &addr, &addrLen) == -1) {
        throw ML::Exception(errno, "getsockname");
    }

    /* writer */
    int writer = socket(AF_INET, SOCK_STREAM, 0);
    if (writer == -1) {
        throw ML::Exception(errno, "socket");
    }

#if 0
    {
        int flag = 1;
        if (setsockopt(writer,
                       IPPROTO_TCP, TCP_NODELAY,
                       (char *) &flag, sizeof(int)) == -1) {
            throw ML::Exception(errno, "setsockopt TCP_NODELAY");
        }
    }
#endif

    if (connect(writer, (const struct sockaddr *) &addr, addrLen) == -1) {
        throw ML::Exception(errno, "connect");
    }
    ML::set_file_flag(writer, O_NONBLOCK);

    int reader = accept(listener, (struct sockaddr *) &addr, &addrLen);
    if (reader == -1) {
        throw ML::Exception(errno, "accept");
    }
    ML::set_file_flag(reader, O_NONBLOCK);

    close(listener);

    return {writer, reader};
}

void doBench(const string & label,
             int writerFd, int readerFd,
             int numMessages, size_t msgSize)
{
    string message = randomString(msgSize);
    MessageLoop writerLoop, readerLoop;

    /* writer setup */
    writerLoop.start();
    Date lastWrite;
    Date lastWriteResult;
    int numWriteResults(0);
    int numWritten(0);
    int numMissed(0);
    auto onWriteResult = [&] (AsyncWriteResult result) {
        if (result.error != 0) {
            throw ML::Exception("write error");
        }
        numWriteResults++;
        if (numWriteResults == numMessages) {
            lastWriteResult = Date::now();
            ML::futex_wake(numWriteResults);
        }
    };

    auto writer = make_shared<WriterSource>(writerFd, 1000);
    writerLoop.addSource("writer", writer);

    /* reader setup */
    readerLoop.start();
    Date lastRead;
    size_t totalBytes = msgSize * numMessages;
    size_t bytesRead(0);
    auto onReaderData = [&] (const char * data, size_t size) {
        bytesRead += size;
        if (bytesRead == totalBytes) {
            lastRead = Date::now();
        }
    };
    auto reader = make_shared<ReaderSource>(readerFd, onReaderData, 262144);
    readerLoop.addSource("reader", reader);

    writer->waitConnectionState(AsyncEventSource::CONNECTED);
    reader->waitConnectionState(AsyncEventSource::CONNECTED);

    Date start = Date::now();
    ML::memory_barrier();
    for (numWritten = 0 ; numWritten < numMessages;) {
        if (writer->write(message, onWriteResult)) {
            numWritten++;
        }
        else {
            numMissed++;
        }
    }
    ML::memory_barrier();
    lastWrite = Date::now();

    while (numWriteResults < numMessages) {
        int old = numWriteResults;
        ML::futex_wait(numWriteResults, old);
    }

    while (bytesRead < totalBytes) {
        ML::sleep(1.0);
    }

    double totalTime = lastRead - start;
    ::printf("%s,%d,%lu,%lu,%d,%f,%f,%f,%f,%f\n",
             label.c_str(),
             numMessages, msgSize, bytesRead, numMissed,
             (lastWrite - start),
             (lastWriteResult - start),
             totalTime,
             (double(numMessages) / totalTime),
             (double(totalBytes) / totalTime));

    readerLoop.shutdown();
    writerLoop.shutdown();
}

void benchFunction(const string & label,
                   std::function<pair<int, int> ()> f)
{
    int multiplier(1);
    for (int i = 0; i < 4; i++) {
        multiplier *= 10;
        auto fds = f();
        doBench(label, fds.first, fds.second,
                10000000 / multiplier, 50 * multiplier);
    }
}

int main()
{
    ::printf("label,msgs_count,msg_size,bytes_xfer,miss_count,"
             "delta_last_write,delta_last_written,delta_last_read,"
             "msg_rate,byte_rate\n");

    benchFunction("pipe", makePipePair);
    benchFunction("unix", makeUnixSocketPair);
    benchFunction("tcp4", makeTcpSocketPair);

    return 0;
}
