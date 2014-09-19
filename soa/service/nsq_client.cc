/* nsq_client.cc
   Wolfgang Sourdeau, June 2014
   Copyright (c) 2014 Datacratic Inc.  All rights reserved.
*/

#include <endian.h>
#include <arpa/inet.h>

#include <iostream>
#include <mutex>
#include <vector>

#include "jml/arch/timers.h"
#include "jml/utils/string_functions.h"

#include "soa/types/basic_value_descriptions.h"

#include "nsq_client.h"

using namespace std;
using namespace Datacratic;


/* NSQ IDENTIFY PAYLOAD */

NsqIdentifyPayload::
NsqIdentifyPayload()
    : featureNegotiation(false),
      userAgent("DatacraticNsq/0.1")
{
}

NsqIdentifyPayloadDescription::
NsqIdentifyPayloadDescription()
{
    addField("client_id", &NsqIdentifyPayload::clientId, "");
    addField("hostname", &NsqIdentifyPayload::hostname, "");
    addField("feature_negotiation", &NsqIdentifyPayload::featureNegotiation, "");
    addField("user_agent", &NsqIdentifyPayload::userAgent, "");
}


/* NSQ CLIENT */

void
NsqClient::
onReceivedData(const char * bufferData, size_t bufferSize)
{
    const char * data;
    size_t dataSize;
    bool fromBuffer;

    if (parserBuffer_.size() > 0) {
        parserBuffer_.append(bufferData, bufferSize);
        data = parserBuffer_.c_str();
        fromBuffer = true;
        dataSize = parserBuffer_.size();
    }
    else {
        data = bufferData;
        fromBuffer = false;
        dataSize = bufferSize;
    }

    auto parseUInt32 = [&] (uint32_t & value) {
        if (dataSize < 4) {
            if (!fromBuffer) {
                parserBuffer_.assign(data, dataSize);
            }
            return false;
        }
        value = ntohl(*(uint32_t *) data);
        data += 4;
        dataSize -= 4;
        parserStep_++;

        return true;
    };

    int responseHandled(0);
    while (dataSize > 0) {
        if (parserStep_ == 0) {
            if (!parseUInt32(parserRemaining_)) {
                return;
            }
        }
        if (parserStep_ == 1) {
            uint32_t responseType;
            if (!parseUInt32(responseType)) {
                return;
            }
            parserFrame_.type = (NsqFrameType) responseType;
            parserRemaining_ -= 4;
        }
        if (parserStep_ == 2) {
            size_t chunkSize = dataSize;
            if (chunkSize > parserRemaining_) {
                chunkSize = parserRemaining_;
            }
            parserFrame_.data.append(data, chunkSize);
            parserRemaining_ -= chunkSize;
            if (parserRemaining_ == 0) {
                handleFrame();
                responseHandled++;
                parserStep_ = 0;
            }
            data += chunkSize;
            dataSize -= chunkSize;
            if (dataSize > 0) {
                if (fromBuffer) {
                    parserBuffer_.assign(data, dataSize);
                    data = parserBuffer_.c_str();
                }
            }
            else {
                parserBuffer_.clear();
                return;
            }
        }
    }
    cerr << "response handled: " + to_string(responseHandled) + "\n";
}

void
NsqClient::
forceWrite(string data)
{
    bool paused(false);
    while (!write(move(data), nullptr)) {
        if (!paused) {
            paused = true;
            ::fprintf(stderr, "queue is full (%s)...\n", data.c_str());
        }
        ML::sleep(0.1);
    }
    if (paused) {
        ::fprintf(stderr, "queue reestablished\n");
    }
}

void
NsqClient::
handleFrame()
{
    switch (parserFrame_.type) {
    case NsqFrameType::Response:
    case NsqFrameType::Error: {
        handleCommandFrame();
        break;
    }
    case NsqFrameType::Message: {
        handleNsqMessage();
        break;
    }
    default: 
        throw ML::Exception("unhandled response type");
    };

    parserFrame_.data.clear();
}

void 
NsqClient::
handleCommandFrame()
{
    if (parserFrame_.data == "_heartbeat_") {
        nop();
    }
    else {
        OnFrame onFrame;

        {
            unique_lock<mutex> guard(callbacksLock_);
            ExcAssert(!callbacks_.empty());
            onFrame = callbacks_.front();
            callbacks_.pop();
        }

        if (onFrame) {
            onFrame(parserFrame_);
        }
    }
}

void 
NsqClient::
handleNsqMessage()
{
    const char * data = parserFrame_.data.c_str();

    uint64_t nanos = be64toh(*(uint64_t *) data);
    double nanoDouble = (double) nanos;
    Date ts = Date::fromSecondsSinceEpoch(nanoDouble / 1000000000);
    uint16_t attempts = ntohs(*(uint16_t *) (data + 8));
    string messageId(data + 10, 16);
    string message(data + 26,
                   parserFrame_.data.size() - 26);

    onMessage(ts, attempts, messageId, message);

    remainingRdy_--;
    if (remainingRdy_ == 0) {
        resetRdy();
    }
}

void 
NsqClient::
onMessage(Date ts, uint16_t attempts,
          const string & messageId, const string & message)
{
    if (onMessage_) {
        onMessage_(ts, attempts, messageId, message);
    }
}

TcpConnectionResult
NsqClient::
connectSync()
{
    TcpConnectionResult result;

    int connected(false);
    ML::memory_barrier();

    auto onConnectionResult = [&, this] (TcpConnectionResult newResult) {
        if (newResult.code == TcpConnectionCode::Success) {
            forceWrite("  V2");
        }
        result = move(newResult);
        ML::memory_barrier();
        connected = true;
        ML::futex_wake(connected);
    };
    cerr << "connectSync start: " << this << endl;
    connect(onConnectionResult);

    while (!connected) {
        int old = connected;
        ML::futex_wait(connected, old);
    }

    return result;
}

void
NsqClient::
nop()
{
    forceWrite("NOP\n");
}

void
NsqClient::
cls(const OnFrame & onFrame)
{
    unique_lock<mutex> guard(callbacksLock_);
    callbacks_.emplace(onFrame);
    forceWrite("CLS\n");
}

void
NsqClient::
identify(const OnFrame & onFrame)
{
    unique_lock<mutex> guard(callbacksLock_);
    callbacks_.emplace(onFrame);

    string identifyMsg = "IDENTIFY\n";
    NsqIdentifyPayload payload;
    string json = jsonEncodeStr(payload);
    uint32_t dataSize = htonl(json.size());
    identifyMsg.append((char *) &dataSize, sizeof(dataSize));
    identifyMsg.append(json);
    forceWrite(move(identifyMsg));
}

void
NsqClient::
sub(const string & topic, const string & channel,
    const OnFrame & onFrame)
{
    unique_lock<mutex> guard(callbacksLock_);
    callbacks_.emplace(onFrame);
    forceWrite("SUB " + topic + " " + channel + "\n");
    resetRdy();
}

void
NsqClient::
pub(const string & topic, const string & message,
    const OnFrame & onFrame)
{
    unique_lock<mutex> guard(callbacksLock_);
    callbacks_.emplace(onFrame);
    string pubMsg = "PUB " + topic + "\n";
    uint32_t dataSize = htonl(message.size());
    pubMsg.append((char *) &dataSize, sizeof(dataSize));
    pubMsg.append(message);
    forceWrite(move(pubMsg));
}

void
NsqClient::
fin(const string & messageId)
{
    forceWrite("FIN " + messageId + "\n");
}

void
NsqClient::
rdy(int count)
{
    forceWrite("RDY " + to_string(count) + "\n");
}
