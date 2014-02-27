/* remote_input.cc
   Jeremy Barnes, 26 May 2011
   Copyright (c) 2011 Datacratic.  All rights reserved.

   Remote input endpoint (listens for a remote output connection).
*/

#include "remote_input.h"
#include <boost/iostreams/categories.hpp>
#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/filter/bzip2.hpp>
#include <boost/iostreams/filter/gzip.hpp>

using namespace std;


namespace Datacratic {

/*****************************************************************************/
/* REMOTE LOG CONNECTION                                                     */
/*****************************************************************************/

struct RemoteInputConnection : public PassiveConnectionHandler {

    RemoteInputConnection()
        : bytes_in(0)
    {
#if 0
        using namespace boost::iostreams;

        auto_ptr<filtering_ostream> new_stream
            (new filtering_ostream());

        bool gzip = true;
        bool bzip2 = false;

        if (gzip)
            new_stream->push(gzip_decompressor());
        else if (bzip2)
            new_stream->push(bzip2_decompressor());
    
        new_stream->push(ConnectionSink(this));

        stream.reset(new_stream.release());
#endif
    }

    ~RemoteInputConnection()
    {
        
        cerr << "input got total of " << bytes_in << " bytes" << endl;
    }

    virtual void onGotTransport()
    {
        cerr << "on input got transport" << endl;
        startReading();
    }

    virtual void handleData(const std::string & data)
    {
        // We shouldn't get anything back on this (yet)
        //cerr << "warning: input handleData: shouldn't have received anything"
        //     << " but got " << data.size() << " bytes" << endl;
        bytes_in += data.size();
    }

    virtual void handleError(const std::string & error)
    {
        cerr << "Remote Log Input Connection got an error: " << error << endl;
        //abort();
    }

    virtual void onCleanup()
    {
        cerr << "input onCleanup" << endl;
    }

    virtual void handleDisconnect()
    {
        cerr << "input handleDisconnect()" << endl;
        //closePeer();
        closeWhenHandlerFinished();
    }

    uint64_t bytes_in;

#if 0
    void logMessage(const std::string & channel,
                    const std::string & message)
    {
        //cerr << "remoteOutputConnection::logMessage()" << endl;
        (*stream) << channel << '\t' << message << flush;
        //this->send(channel + "-" + message, NEXT_CONTINUE, onSendFinished);
    }

    boost::scoped_ptr<std::ostream> stream;

    // TODO: how to deal with dropped messages?
    struct ConnectionSink : public boost::iostreams::sink {
        ConnectionSink(RemoteOutputConnection * connection)
            : connection(connection)
        {
        }
        
        RemoteOutputConnection * connection;

        std::streamsize write(const char * s, std::streamsize n)
        {
            auto onSendFinished = [=] ()
                {
                    cerr << "finished sending remote log message" << endl;
                    abort();
                };
            
            connection->send(std::string(s, n),
                             NEXT_CONTINUE,
                             onSendFinished);

            return n;
        }
    };
#endif    
};



/*****************************************************************************/
/* REMOTE INPUT                                                              */
/*****************************************************************************/

RemoteInput::
RemoteInput()
    : endpoint("RemoteInput")
{
}

RemoteInput::
~RemoteInput()
{
    shutdown();
}

void
RemoteInput::
listen(int port,
       const std::string & hostname,
       boost::function<void ()> onShutdown)
{
    shutdown();

    endpoint.onMakeNewHandler
        = [=] () -> std::shared_ptr<ConnectionHandler>
        {
            return ML::make_std_sp(new RemoteInputConnection());
        };

    endpoint.onAcceptError = [=] (const std::string & str)
        {
            cerr << "error in accept: " << str << endl;
        };

    cerr << "start endpoint init" << endl;

    endpoint.init(port, hostname,
                  1    /* num_threads */,
                  true /* sync */,
                  10   /* backlog */);
    
    cerr << "finished endpoint init" << endl;

    this->onShutdown = onShutdown;
}

void
RemoteInput::
shutdown()
{
    if (onShutdown) {
        onShutdown();
        onShutdown.clear();
    }
    endpoint.shutdown();
}

} // namespace Datacratic
