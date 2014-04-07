/* zmq_utils.h                                                     -*- C++ -*-
   Jeremy Barnes, 15 March 2011
   Copyright (c) 2011 Datacratic.  All rights reserved.

   Utilities for zmq.
*/

#pragma once

#include <unistd.h>
#include <string>
#include <iostream>
#include <cstdio>
#include <memory>
#include <boost/utility.hpp>
#include "soa/service/zmq.hpp"
#include "soa/jsoncpp/value.h"
#include "soa/types/date.h"
#include "soa/types/string.h"
#include "soa/service/port_range_service.h"
#include "jml/arch/format.h"
#include "jml/arch/exception.h"
#include "jml/compiler/compiler.h"

#if 0
#define BLOCK_FLAG 0
#else
#define BLOCK_FLAG ZMQ_DONTWAIT
#endif

namespace Datacratic {

inline void setIdentity(zmq::socket_t & sock, const std::string & identity)
{
    sock.setsockopt(ZMQ_IDENTITY, (void *)identity.c_str(), identity.size());
}

inline void subscribeChannel(zmq::socket_t & sock, const std::string & channel)
{
    sock.setsockopt(ZMQ_SUBSCRIBE, channel.c_str(), channel.length());
}

inline void unsubscribeChannel(zmq::socket_t & sock, const std::string & channel)
{
    sock.setsockopt(ZMQ_UNSUBSCRIBE, channel.c_str(), channel.length());
}

inline void setHwm(zmq::socket_t & sock, int hwm)
{
    sock.setsockopt(ZMQ_SNDHWM, &hwm, sizeof(hwm));
    sock.setsockopt(ZMQ_RCVHWM, &hwm, sizeof(hwm));
}

inline void throwSocketError(const char *data)
{
    throw ML::Exception(errno,
                        "unhandled error (" + std::to_string(errno) + ")",
                        data);
}

/** Returns events available on a socket: (input, output). */
inline std::pair<bool, bool>
getEvents(zmq::socket_t & sock)
{
    int events = 0;
    size_t events_size = sizeof(events);
    sock.getsockopt(ZMQ_EVENTS, &events, &events_size);
    return std::make_pair(events & ZMQ_POLLIN, events & ZMQ_POLLOUT);
}

inline std::string recvMesg(zmq::socket_t & sock)
{
     zmq::message_t message;
     while (!sock.recv(&message, 0)) ;

     return std::string((const char *)message.data(),
                        ((const char *)message.data()) + message.size());
}

inline std::pair<std::string, bool>
recvMesgNonBlocking(zmq::socket_t & sock)
{
    zmq::message_t message;
    bool got = sock.recv(&message, ZMQ_NOBLOCK);
    if (!got)
        return std::make_pair("", false);
    return std::make_pair
        (std::string((const char *)message.data(),
                     ((const char *)message.data()) + message.size()),
         true);
}

inline void recvMesg(zmq::socket_t & sock, char & val)
{
    zmq::message_t message;
    while (!sock.recv(&message, 0)) ;
    if (message.size() != 1)
        throw ML::Exception("invalid char message size");
    val = *(const char *)message.data();
}

inline std::vector<std::string> recvAll(zmq::socket_t & sock)
{
    std::vector<std::string> result;

    int64_t more = 1;
    size_t more_size = sizeof (more);

    while (more) {
        result.push_back(recvMesg(sock));
        sock.getsockopt(ZMQ_RCVMORE, &more, &more_size);
    }

    return result;
}

inline std::vector<std::string>
recvAllNonBlocking(zmq::socket_t & sock)
{
    std::vector<std::string> result;

    std::string msg0;
    bool got;
    std::tie(msg0, got) = recvMesgNonBlocking(sock);
    if (got) {
        result.push_back(msg0);

        for (;;) {
            int64_t more = 1;
            size_t more_size = sizeof (more);
            sock.getsockopt(ZMQ_RCVMORE, &more, &more_size);
            //using namespace std;
            //cerr << "result.size() " << result.size()
            //     << " more " << more << endl;
            if (!more) break;
            result.push_back(recvMesg(sock));
        }
    }
    
    return result;
}

inline zmq::message_t encodeMessage(const std::string & message)
{
    return message;
}

inline zmq::message_t encodeMessage(const Utf8String & message)
{
    return message.rawString();
}

inline zmq::message_t encodeMessage(const Utf32String & message)
{
    return message.utf8String();
}

inline zmq::message_t encodeMessage(const char * msg)
{
    size_t sz = strlen(msg);
    zmq::message_t zmsg(sz);
    std::copy(msg, msg + sz, (char *)zmsg.data());
    return zmsg;
}

inline zmq::message_t encodeMessage(const Date & date)
{
    return ML::format("%.5f", date.secondsSinceEpoch());
}

inline zmq::message_t encodeMessage(int i)
{
    return ML::format("%d", i);
}

inline zmq::message_t encodeMessage(unsigned int i)
{
    return ML::format("%u", i);
}

inline zmq::message_t encodeMessage(double i)
{
    return ML::format("%f", i);
}

inline zmq::message_t encodeMessage(long i)
{
    return ML::format("%ld", i);
}

inline zmq::message_t encodeMessage(unsigned long i)
{
    return ML::format("%lu", i);
}

inline zmq::message_t encodeMessage(char c)
{
    return ML::format("%c", c);
}

inline std::string chomp(const std::string & s)
{
    const char * start = s.c_str();
    const char * end = start + s.length();
    
    while (end > start && end[-1] == '\n') --end;
    
    if (end == start + s.length()) return s;
    return std::string(start, end);
}

inline zmq::message_t encodeMessage(const Json::Value & j)
{
    return chomp(j.toString());
}

inline bool sendMesg(zmq::socket_t & sock,
                     const std::string & msg,
                     int options = 0)
{
    zmq::message_t msg1(msg.size());
    std::copy(msg.begin(), msg.end(), (char *)msg1.data());
    return sock.send(msg1, options);
}
    
inline bool sendMesg(zmq::socket_t & sock,
                     const char * msg,
                     int options = 0)
{
    size_t sz = strlen(msg);
    zmq::message_t msg1(sz);
    std::copy(msg, msg + sz, (char *)msg1.data());
    return sock.send(msg1, options);
}
    
inline bool sendMesg(zmq::socket_t & sock,
                     const void * msg,
                     size_t sz,
                     int options = 0)
{
    zmq::message_t msg1(sz);
    memcpy(msg1.data(), msg, sz);
    return sock.send(msg1, options);
}

template<typename T>
inline bool sendMesg(zmq::socket_t & sock,
                     const T & obj,
                     int options = 0)
{
    return sock.send(encodeMessage(obj), options);
}

template<typename T>
bool sendMesg(zmq::socket_t & socket, const std::shared_ptr<T> & val,
              int flags = 0)
{
    return sendMesg(socket, sharedPtrToMessage(val), flags);
}

inline void sendAll(zmq::socket_t & sock,
                    const std::vector<std::string> & message,
                    int lastFlags = 0)
{
    if (message.empty())
        throw ML::Exception("can't send an empty message vector");

    for (unsigned i = 0;  i < message.size() - 1;  ++i)
        if (!sendMesg(sock, message[i], ZMQ_SNDMORE | BLOCK_FLAG)) {
            throwSocketError(__FUNCTION__);
        }
    if (!sendMesg(sock, message.back(), lastFlags | BLOCK_FLAG)) {
        throwSocketError(__FUNCTION__);
    }
}

inline void sendAll(zmq::socket_t & sock,
                    const std::initializer_list<std::string> & message,
                    int lastFlags = 0)
{
    sendAll(sock, std::vector<std::string>(message));
}

#if 0
template<typename T>
inline void sendAll(zmq::socket_t & socket,
                    const std::vector<T> & vals,
                    int lastFlags)
{
    if (vals.empty()) {
        throw ML::Exception("can't send empty vector");
    }
    for (int i = 0;  i < vals.size() - 1;  ++i)
        sendMesg(socket, vals[i], ZMQ_SNDMORE | BLOCK_FLAG);
    sendMesg(socket, vals.back(), lastFlags | BLOCK_FLAG);
}
#endif

template<typename Arg1>
void sendMessage(zmq::socket_t & socket,
                 const Arg1 & arg1)
{
    if (!sendMesg(socket, arg1, 0)) {
        throwSocketError(__FUNCTION__);
    }
}

inline void sendMessage(zmq::socket_t & socket,
                        const std::vector<std::string> & args)
{
    sendAll(socket, args, 0);
}

template<typename Arg1, typename... Args>
void sendMessage(zmq::socket_t & socket,
                 const Arg1 & arg1,
                 Args... args)
{
    if (!sendMesg(socket, arg1, ZMQ_SNDMORE | BLOCK_FLAG)) {
        throwSocketError(__FUNCTION__);
    }
    sendMessage(socket, args...);
}

/* non-throwing versions, where EAGAIN cases would return false */
template<typename Arg1>
bool trySendMessage(zmq::socket_t & socket, const Arg1 & arg1)
{
    if (!sendMesg(socket, arg1, 0)) {
        if (errno == EAGAIN)
            return false;
        else
            throwSocketError(__FUNCTION__);
    }

    return true;
}

template<typename Arg1, typename... Args>
bool trySendMessage(zmq::socket_t & socket, const Arg1 & arg1, Args... args)
{
    if (!sendMesg(socket, arg1, ZMQ_SNDMORE | BLOCK_FLAG)) {
        if (errno == EAGAIN)
            return false;
        else
            throwSocketError(__FUNCTION__);
    }
    return trySendMessage(socket, args...);
}

inline bool trySendAll(zmq::socket_t & sock,
                       const std::vector<std::string> & message,
                       int lastFlags = 0)
{
    if (message.empty())
        throw ML::Exception("can't send an empty message vector");

    for (unsigned i = 0; i < message.size() - 1;  ++i) {
        if (!sendMesg(sock, message[i], ZMQ_SNDMORE | BLOCK_FLAG)) {
            if (errno == EAGAIN)
                return false;
            else
                throwSocketError(__FUNCTION__);
        }
    }

    if (!sendMesg(sock, message.back(), lastFlags | BLOCK_FLAG)) {
        if (errno == EAGAIN)
            return false;
        else
            throwSocketError(__FUNCTION__);
    }

    return true;
}

inline bool trySendAll(zmq::socket_t & sock,
                        const std::initializer_list<std::string> & message,
                        int lastFlags = 0)
{
    return trySendAll(sock, std::vector<std::string>(message), lastFlags);
}

/* We take a copy of the shared pointer in a heap-allocated object that
   makes sure that it continues to have a reference.  The control
   connection then takes control of the pointer.  This allows us to
   effectively transfer a shared pointer over a zeromq socket.
*/
template<typename T>
std::string sharedPtrToMessage(std::shared_ptr<T> ptr)
{
    static const int mypid = getpid();

    std::auto_ptr<std::shared_ptr<T> > ptrToTransfer
        (new std::shared_ptr<T>(ptr));
    
    std::string ptrMsg
        = ML::format("%p:%d:%p",
                     ptrToTransfer.get(), mypid,
                     typeid(T).name());
    
    ptrToTransfer.release();
    
    return ptrMsg;
}

template<typename T>
std::shared_ptr<T>
sharedPtrFromMessage(const std::string & message)
{
    static const int mypid = getpid();

    std::shared_ptr<T> * ptr;
    int pid;
    const char * name;

    int res = std::sscanf(message.c_str(), "%p:%d:%p", &ptr, &pid, &name);
    if (res != 3)
        throw ML::Exception("failure reconstituting auction");
    if (pid != mypid)
        throw ML::Exception("message comes from different pid");
    if (name != typeid(T).name())
        throw ML::Exception("wrong name for type info: %s vs %s",
                            name, typeid(T).name());

    std::auto_ptr<std::shared_ptr<T> > ptrHolder(ptr);
    return *ptr;
}

template<typename T>
void recvMesg(zmq::socket_t & sock, std::shared_ptr<T> & val)
{
    return sharedPtrFromMessage<T>(recvMesg(sock));
}

inline void close(zmq::socket_t & sock)
{
    zmq_close(sock);
}

inline int
bindAndReturnOpenTcpPort(zmq::socket_t & socket, PortRange const & portRange, const std::string & host) {
    std::string uri;
    int port = portRange.bindPort([&](int port) {
        uri = ML::format("tcp://%s:%d", host.c_str(), port);
        return zmq_bind(socket, uri.c_str()) == 0;
    });

    if(port == -1)
        throw ML::Exception("no open TCP port '%s': %s %s",
                            uri.c_str(),
                            zmq_strerror(zmq_errno()),
                            strerror(errno));
    return port;
}

inline std::string
bindToOpenTcpPort(zmq::socket_t & socket, PortRange const & portRange, const std::string & host) {
    int port = bindAndReturnOpenTcpPort(socket, portRange, host);
    return ML::format("tcp://%s:%d", host.c_str(), port);
}

/** Return a human readable string for a zeromq event name. */
std::string printZmqEvent(int event);

/** Return if the given error event represents some kind of failure or
    not.
*/
bool zmqEventIsError(int event);

} // namespace Datacratic
