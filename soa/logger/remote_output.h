/* remote_output.h                                                 -*- C++ -*-
   Jeremy Barnes, 23 May 2011
   Copyright (c) 2011 Datacratic.  All rights reserved.

   Output source that goes to a remote sink.  TCP/IP.

   
*/

#pragma once

#include "logger.h"
#include "soa/service/active_endpoint.h"


namespace Datacratic {


struct RemoteOutputConnection;


/*****************************************************************************/
/* REMOTE OUTPUT                                                             */
/*****************************************************************************/

/** Logging output class that establishes a connection to another machine and
    sends zipped versions of the log file to that machine.
*/

struct RemoteOutput
    : public LogOutput,
      protected ActiveEndpointT<SocketTransport> {

    RemoteOutput();

    virtual ~RemoteOutput();

    /** Connect to the remote endpoint and start sending on down those logs. */
    void connect(int port, const std::string & hostname, double timeout = 10.0);
    
    /** Close everything down. */
    void shutdown();

    /** Make sure that everything that's pending has already been sent before
        returning from this function. */
    void barrier();

    /** Flush out the current messages, waiting until everything is done
        before we stop. */
    void flush();

    /** Sync all data and wait for it to finish */
    void sync();

    /** Close the connection. */
    void close();

    virtual void logMessage(const std::string & channel,
                            const std::string & message);

    /** Notification that a connection was closed.  This can be used to give a
        new set of data.
    */
    virtual void
    notifyCloseTransport(const std::shared_ptr<TransportBase> & transport);

    /** Thing to be called back on a connection error.  Used to hook in to
        allow notification of problems.
    */
    boost::function<void (const std::string)> onConnectionError;

private:
    /** Internal helper function used to reconnect to the remote server. */
    void reconnect(boost::function<void ()> onFinished,
                   boost::function<void (const std::string &)> onError,
                   double timeout);

    /** Internal helper function used as a callback from the reconnect.  It
        will set up the internal connection and then call onFinished.  If
        there is an error, it will call onError.
    */
    void setupConnection(std::shared_ptr<TransportBase> transport,
                         boost::function<void ()> onFinished,
                         boost::function<void (const std::string &)> onError);

    int port;
    std::string hostname;
    double timeout;
    std::shared_ptr<RemoteOutputConnection> connection;
    bool shuttingDown;

    /** Backlog of messages to send whilst connection is down. */
    std::vector<std::pair<std::string, std::string> > backlog;
};

} // namespace Datacratic
