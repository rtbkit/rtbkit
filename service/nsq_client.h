/* nsq_client.h                                                    -*- C++ -*-
   Wolfgang Sourdeau, August 2014
   Copyright (c) 2014 Datacratic.  All rights reserved.

   A class handling the NSQ protocol.
   See http://nsq.io/clients/tcp_protocol_spec.html.

*/

#pragma once

#include <string>

#include "soa/types/date.h"
#include "soa/types/value_description.h"

#include "soa/service/tcp_client.h"


namespace Datacratic {

/****************************************************************************/
/* NSQ IDENTIFY PAYLOAD                                                     */
/****************************************************************************/

struct NsqIdentifyPayload {
    NsqIdentifyPayload();

    std::string clientId;
    std::string hostname;
    bool featureNegotiation;
    // int heartbeatInterval;
    // int outputBufferSize;
    // int outputBufferTimeout;
    // bool tlsV1;
    // bool snappy;
    // bool deflate;
    // bool deflateLevel;
    // int sampleRate;
    std::string userAgent;
    // int msgTimeout;
};

CREATE_STRUCTURE_DESCRIPTION(NsqIdentifyPayload);


/****************************************************************************/
/* NSQ FRAME                                                                */
/****************************************************************************/

enum NsqFrameType {
    Response = 0,
    Error = 1,
    Message = 2
};

struct NsqFrame {
    NsqFrameType type;
    std::string data;
};


/****************************************************************************/
/* NSQ CLIENT                                                               */
/****************************************************************************/

struct NsqClient : public TcpClient {
    typedef std::function<void (const NsqFrame &)> OnFrame;
    typedef std::function<void (Date, uint16_t,
                                const std::string &,
                                const std::string &)> OnMessage;

    NsqClient(OnClosed onClosed = nullptr,
              const OnMessage & onMessage = nullptr)
        : TcpClient(onClosed, nullptr, nullptr, 0),
          parserStep_(0), parserRemaining_(0),
          onMessage_(onMessage), remainingRdy_(0)
    {
        setUseNagle(true);
    }

    TcpConnectionResult connectSync();

    void nop();
    void cls(const OnFrame & onFrame);

    void identify(const OnFrame & onFrame = nullptr);

    void sub(const std::string & topic, const std::string & channel,
             const OnFrame & onFrame = nullptr);
    void rdy(int count);

    void pub(const std::string & topic, const std::string & message,
             const OnFrame & onFrame = nullptr);

    void fin(const std::string & messageId);

    virtual void onMessage(Date ts, uint16_t attempts,
                           const std::string & messageId,
                           const std::string & message);

private:
    void onReceivedData(const char * buffer, size_t bufferSize);

    void forceWrite(std::string data);

    void handleFrame();
    void handleCommandFrame();
    void handleNsqMessage();

    void resetRdy()
    {
        remainingRdy_ = 1000;
        rdy(1000);
    }

    /* response parsing */
    int parserStep_; /* 0 = size; 1 = type; 2 = message */
    uint32_t parserRemaining_; /* size missing from message */
    std::string parserBuffer_;
    NsqFrame parserFrame_;

    std::mutex callbacksLock_;
    std::queue<OnFrame> callbacks_;

    OnMessage onMessage_;
    int remainingRdy_;
};

} // namespace Datacratic
