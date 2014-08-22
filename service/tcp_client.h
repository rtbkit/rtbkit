/* tcp_client.h                                                    -*- C++ -*-
   Wolfgang Sourdeau, April 2014
   Copyright (c) 2014 Datacratic.  All rights reserved.

   A helper base class for handling tcp sockets.
*/

#pragma once

#include <atomic>
#include <string>

#include "jml/arch/wakeup_fd.h"
#include "jml/utils/ring_buffer.h"

#include "async_writer_source.h"


namespace Datacratic {

struct Url;


/****************************************************************************/
/* CLIENT TCP SOCKET CONNECTION RESULT                                      */
/****************************************************************************/

enum ConnectionResult {
    Success = 0,
    UnknownError = 1,
    ConnectionFailure = 2,
    HostUnknown = 3,
    Timeout = 4
};


/****************************************************************************/
/* CLIENT TCP SOCKET STATE                                                  */
/****************************************************************************/

enum TcpClientState {
    Disconnected,
    Connecting,
    Connected
};


/****************************************************************************/
/* CLIENT TCP SOCKET                                                        */
/****************************************************************************/

/* A class that handles the asynchronous opening and connection of TCP
 * sockets. */

struct TcpClient : public AsyncWriterSource
{
    typedef std::function<void(ConnectionResult, const std::vector<std::string> &)>
        OnConnectionResult;

    TcpClient(OnConnectionResult onConnectionResult = nullptr,
                    OnClosed onClosed = nullptr,
                    OnWriteResult onWriteResult = nullptr,
                    OnReceivedData onReceivedData = nullptr,
                    OnException onException = nullptr,
                    size_t maxMessages = 32,
                    size_t recvBufSize = 65536);

    virtual ~TcpClient();

    /* utility functions to defined the target service */
    void init(const std::string & url);
    void init(const Url & url);
    void init(const std::string & address, int port);

    /* disable the Nagle algorithm (TCP_NODELAY) */
    void setUseNagle(bool useNagle);

    /* initiate or restore a connection to the target service */
    void connect();

    /* invoked when the status of the connection becomes available */
    virtual void onConnectionResult(ConnectionResult result,
                                    const std::vector<std::string> & msgs);

    /* state of the connection */
    TcpClientState state() const
    { return TcpClientState(state_); }

private:
    void handleConnectionEvent(int socketFd, const ::epoll_event & event);
    void handleConnectionResult();

    std::string address_;
    int port_;
    int state_; /* TcpClientState */
    bool noNagle_;

    EpollCallback handleConnectionEventCb_;

    OnConnectionResult onConnectionResult_;
};

} // namespace Datacratic
