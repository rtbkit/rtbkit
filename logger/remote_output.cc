/* remote_output.cc
   Jeremy Barnes, 23 May 2011
   Copyright (c) 2011 Datacratic.  All rights reserved.

*/

#include "remote_output.h"
#include <boost/iostreams/categories.hpp>
#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/filter/bzip2.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include "filter.h"

using namespace std;
using namespace ML;
using namespace boost::iostreams;

namespace Datacratic {



/*****************************************************************************/
/* REMOTE LOG CONNECTION                                                     */
/*****************************************************************************/

struct RemoteOutputConnection
    : public PassiveConnectionHandler {

    RemoteOutputConnection()
        : messageSerial(0), messageWritten(0)
    {
        std::shared_ptr<ZlibCompressor> filter
            (new ZlibCompressor());
        filter->onOutput = boost::bind(&RemoteOutputConnection::write,
                                       this,
                                       _1, _2, _4);
        this->filter = filter;
    }

    ~RemoteOutputConnection()
    {
        cerr << "destroying remote output connection" << endl;
    }

    virtual void onGotTransport()
    {
        cerr << "on got transport" << endl;
        startReading();
    }

    virtual void handleData(const std::string & data)
    {
        // We shouldn't get anything back on this (yet)
        cerr << "warning: handleData: shouldn't have received anything"
             << " but got " << data << endl;
    }

    virtual void handleError(const std::string & error)
    {
        cerr << "Remote Log Connection got an error: " << error << endl;
        //abort();
    }

    virtual void onCleanup()
    {
        cerr << "onCleanup" << endl;
    }

    virtual void handleDisconnect()
    {
        cerr << "handleDisconnect()" << endl;
        closeWhenHandlerFinished();
    }

    void logMessage(const std::string & channel,
                    const std::string & message,
                    boost::function<void ()> onMessageDone)
    {
        string buf = format("%s\t%s\n\r", channel.c_str(), message.c_str());
        filter->process(buf, FLUSH_SYNC, onMessageDone);
    }

    void flush(FlushLevel level = FLUSH_FULL,
               boost::function<void ()> onFlushDone
                   = boost::function<void ()>())
    {
        filter->flush(level, onFlushDone);
    }

    void close(boost::function<void ()> onCloseDone
                   = boost::function<void ()>())
    {
        cerr << "closing" << endl;
        filter->flush(FLUSH_FINISH, onCloseDone);
    }

    size_t messageSerial;
    size_t messageWritten;

    std::shared_ptr<Filter> filter;

    // TODO: how to deal with dropped messages?
    std::streamsize write(const char * s, size_t n,
                          boost::function<void ()> onWriteFinished)
    {
        size_t serial JML_UNUSED = ++messageSerial;
        //cerr << "sending " << n << " bytes" << endl;

        std::string toSend(s, n);
        
        auto onSendFinished = [=] ()
            {
                ++messageWritten;
                if (onWriteFinished)
                    onWriteFinished();
            };
        
        auto doSend = [=] ()
            {
                this->send(toSend, NEXT_CONTINUE, onSendFinished);
            };
        
        doAsync(doSend, "doSendLog");
        
        return n;
    }
    
};


/*****************************************************************************/
/* REMOTE OUTPUT                                                             */
/*****************************************************************************/

RemoteOutput::
RemoteOutput()
    : ActiveEndpointT<SocketTransport>("remoteOutput")
{
    shuttingDown = false;
}

RemoteOutput::
~RemoteOutput()
{
    shutdown();
}

void
RemoteOutput::
connect(int port, const std::string & hostname, double timeout)
{
    Guard guard(lock);

    this->port = port;
    this->hostname = hostname;
    this->timeout = timeout;

    init(port, hostname);

    ACE_Semaphore sem(0);
    string error;

    auto onConnectionDone = [&] ()
        {
            sem.release();
        };

    auto onConnectionError = [&] (const std::string & err)
        {
            error = err;
            sem.release();
        };

    guard.release();

    reconnect(onConnectionDone, onConnectionError, timeout);
    sem.acquire();
    
    if (error != "")
        throw Exception("RemoteOutput::connect(): connection error: "
                        + error);
}

void
RemoteOutput::
reconnect(boost::function<void ()> onFinished,
          boost::function<void (const std::string &)> onError,
          double timeout)
{
    newConnection(boost::bind<void>(&RemoteOutput::setupConnection,
                                    this, _1, onFinished, onError),
                  onError, timeout);
}

void
RemoteOutput::
setupConnection(std::shared_ptr<TransportBase> transport,
                boost::function<void ()> onFinished,
                boost::function<void (const std::string &)> onError)
{
    cerr << "got new connection" << endl;
    
    auto finishConnect = [=] ()
        {
            try {
                std::shared_ptr<RemoteOutputConnection> connection
                    (new RemoteOutputConnection());
                transport->associate(connection);

                Guard guard(this->lock);
                this->connection = connection;
                if (onFinished) onFinished();
            } catch (const std::exception & exc) {
                onError("setupConnection: error: " + string(exc.what()));
            }
        };
    
    // Create a new connection and associate it
    transport->doAsync(finishConnect, "finishConnect");
}

void
RemoteOutput::
barrier()
{
    Guard guard(lock);

    ACE_Semaphore sem(0);

    auto onBarrierDone = [&] ()
        {
            sem.release();
        };

    auto finishBarrier = [&] ()
        {
            this->connection->flush(FLUSH_NONE, onBarrierDone);
        };

    connection->doAsync(finishBarrier, "finishBarrier");
    
    sem.acquire();
}

void
RemoteOutput::
sync()
{
    Guard guard(lock);

    ACE_Semaphore sem(0);

    auto onSyncDone = [&] ()
        {
            sem.release();
        };

    auto finishSync = [&] ()
        {
            this->connection->flush(FLUSH_SYNC, onSyncDone);
        };

    connection->doAsync(finishSync, "finishSync");
    
    sem.acquire();
}

void
RemoteOutput::
flush()
{
    Guard guard(lock);

    ACE_Semaphore sem(0);

    auto onFlushDone = [&] ()
        {
            sem.release();
        };

    auto finishFlush = [&] ()
        {
            this->connection->flush(FLUSH_FULL, onFlushDone);
        };

    connection->doAsync(finishFlush, "finishFlush");
    
    sem.acquire();
}

void
RemoteOutput::
close()
{
    Guard guard(lock);

    ACE_Semaphore sem(0);

    auto onCloseDone = [&] ()
        {
            sem.release();
        };

    auto finishClose = [&] ()
        {
            this->connection->close(onCloseDone);
        };

    connection->doAsync(finishClose, "finishClose");
    
    sem.acquire();
}

void
RemoteOutput::
shutdown()
{
    shuttingDown = true;

    if (connection)
        connection->flush();

    ActiveEndpointT<SocketTransport>::shutdown();

    shuttingDown = false;
}

void
RemoteOutput::
logMessage(const std::string & channel,
           const std::string & message)
{
    Guard guard(lock);

    if (shuttingDown)
        throw Exception("attempt to log message whilst shutting down");

    if (!connection) {
        cerr << "adding to backlog; currently " << backlog.size()
             << " messages" << endl;
        backlog.push_back(make_pair(channel, message));
        return;
    }

    auto onMessageLogged = [] () {};

    // Safe to call from any thread on the connection
    connection->doAsync(std::bind(&RemoteOutputConnection::logMessage,
                                  connection,
                                  channel,
                                  message,
                                  onMessageLogged),
                        "logMessage");
}

void
RemoteOutput::
notifyCloseTransport(const std::shared_ptr<TransportBase> & transport)
{
    Guard guard(lock);

    ActiveEndpointT<SocketTransport>::notifyCloseTransport(transport);

    this->connection.reset();

    if (shuttingDown) return;

    cerr << "transport was closed; reconnecting" << endl;

    auto onConnectDone = [=] ()
        {
            cerr << "new connection done" << endl;
        };

    auto onConnectError = [=] (const std::string & error)
        {
            cerr << "reconnection had error: " << error << endl;
            if (this->onConnectionError)
                this->onConnectionError(error);
        };

    reconnect(onConnectDone, onConnectError, timeout);
}

} // namespace Datacratic
