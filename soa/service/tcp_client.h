/* tcp_client.h                                                    -*- C++ -*-
   Wolfgang Sourdeau, April 2014
   Copyright (c) 2014 Datacratic.  All rights reserved.

   A helper base class for handling tcp sockets.
*/

#pragma once

#include <functional>
#include <string>
#include <vector>

#include "jml/arch/wakeup_fd.h"
#include "jml/utils/ring_buffer.h"

#include "async_writer_source.h"


namespace Datacratic {

struct Url;


/****************************************************************************/
/* TCP CONNECTION CODE                                                      */
/****************************************************************************/

enum TcpConnectionCode {
    Success = 0,
    UnknownError = 1,
    ConnectionFailure = 2,
    HostUnknown = 3,
    Timeout = 4,
    ConnectionEnded = 5
};


/****************************************************************************/
/* TCP CONNECTION RESULT                                                    */
/****************************************************************************/

struct TcpConnectionResult {
    TcpConnectionResult()
        : code(Success)
    {
    }

    TcpConnectionResult(TcpConnectionCode newCode)
        : code(newCode)
    {
    }

    TcpConnectionResult(TcpConnectionCode newCode,
                        std::vector<std::string> newMessages)
        : code(newCode), messages(std::move(newMessages))
    {
    }

    /* status code of the connect operation */
    TcpConnectionCode code;

    /* messages that could not be sent due in the event of a connection
       error */ 
    std::vector<std::string> messages;
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
    typedef std::function<void(TcpConnectionResult)> OnConnectionResult;

    TcpClient(const OnClosed & onClosed = nullptr,
              const OnReceivedData & onReceivedData = nullptr,
              const OnException & onException = nullptr,
              size_t maxMessages = 32,
              size_t recvBufSize = 65536);

    virtual ~TcpClient();

    /* utility functions to defined the target service */
    void init(const std::string & url);
    void init(const Url & url);
    void init(const std::string & hostname, int port);

    /* disable the Nagle algorithm (TCP_NODELAY) */
    void setUseNagle(bool useNagle);

    /* initiate or restore a connection to the target service */
    void connect(const OnConnectionResult & onConnectionResult);

    /* state of the connection */
    TcpClientState state() const
    { return TcpClientState(state_); }

private:
    void handleConnectionEvent(int socketFd,
                               OnConnectionResult onConnectionResult);
    void handleConnectionResult();

    std::string hostname_;
    int port_;
    int state_; /* TcpClientState */
    bool noNagle_;

    EpollCallback handleConnectionEventCb_;
};

} // namespace Datacratic
