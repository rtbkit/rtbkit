/* connection_handler.cc
   Jeremy Barnes, 27 February 2011
   Copyright (c) 2011 Datacratic.  All rights reserved.

   Implementation of connection handler.
*/

#include "connection_handler.h"
#include "jml/arch/demangle.h"
#include "jml/arch/backtrace.h"
#include "jml/utils/guard.h"
#include "soa/service//endpoint.h"


using namespace std;
using namespace ML;


namespace Datacratic {


/*****************************************************************************/
/* CONNECTION HANDLER                                                        */
/*****************************************************************************/

uint32_t ConnectionHandler::created = 0;
uint32_t ConnectionHandler::destroyed = 0;

void
ConnectionHandler::
handleInput()
{
    throw Exception("ConnectionHandler of type %s needs to override "
                    "handle_input()", type_name(*this).c_str());
}

void
ConnectionHandler::
handleOutput()
{
    throw Exception("ConnectionHandler of type %s needs to override "
                    "handle_output()", type_name(*this).c_str());
}

void
ConnectionHandler::
handlePeerShutdown()
{
    // By default we close...
    closeWhenHandlerFinished();
}

void
ConnectionHandler::
handleTimeout(Date time, size_t)
{
    throw Exception("ConnectionHandler of type %s needs to override "
                    "handle_timeout()", type_name(*this).c_str());
}

void
ConnectionHandler::
closeConnection()
{
    int code = closePeer();
    if (code == -1)
        doError("close: " + string(strerror(errno)));
}

void
ConnectionHandler::
startReading()
{
    addActivityS("startReading");
    transport().startReading();
}

void
ConnectionHandler::
stopReading()
{
    addActivityS("stopReading");
    transport().stopReading();
}

void
ConnectionHandler::
startWriting()
{
    addActivityS("startWriting");
    transport().startWriting();
}

void
ConnectionHandler::
stopWriting()
{
    addActivityS("stopWriting");
    transport().stopWriting();
}

void
ConnectionHandler::
setTransport(TransportBase * transport)
{
    //cerr << "setTransport : transport = " << transport << " transport_ = "
    //     << transport_ << endl;

    if (transport_)
        throw Exception("can't switch transports from %8p to %8p",
                        transport_, transport);
    transport_ = transport;
}

void
ConnectionHandler::
addActivity(const std::string & activity)
{
    if (!transport_ || !transport().debug) return;
    transport().addActivity(activity);
}

void
ConnectionHandler::
addActivityS(const char * act)
{
    if (!transport_ || !transport().debug) return;
    transport().addActivity(act);
}

void
ConnectionHandler::
addActivity(const char * fmt, ...)
{
    if (!transport_ || !transport().debug) return;
    va_list ap;
    va_start(ap, fmt);
    ML::Call_Guard cleanupAp([&] () { va_end(ap); });
    transport().addActivity(ML::vformat(fmt, ap));
}

void
ConnectionHandler::
checkMagic() const
{
    if (magic != 0x1234)
        throw Exception("attempt to use deleted ConnectionHandler");
}


/*****************************************************************************/
/* PASSIVE CONNECTION HANDLER                                                */
/*****************************************************************************/

void
PassiveConnectionHandler::
doError(const std::string & error)
{
    this->error = error;
    handleError(error);
    closeWhenHandlerFinished();
}

void
PassiveConnectionHandler::
handleInput()
{
    //cerr << "handle_input on " << fd << " for handler " << ML::type_name(*this)
    //<< endl;
    transport().assertLockedByThisThread();

    size_t chunk_size = 8192;

    string buf;
    size_t done = 0;

    ssize_t bytes_read = 0;

    bool disconnected = false;

    do {
        if (buf.length() - done < chunk_size)
            buf.resize(done + chunk_size);

        errno = 0;

        //cerr << "receiving " << chunk_size << " on top of "
        //     << buf.length() << " already there" << endl;

        bytes_read = recv(&buf[done], chunk_size, MSG_DONTWAIT);

        //int err = errno;

        //cerr << "bytes_read = " << bytes_read 
        //     << " errno " << strerror(errno) << endl;

        if (bytes_read == 0) {
            // Disconnect
            addActivity("readDisconnect");
            //cerr << "**** DISCONNECT ON TRANSPORT " << &transport() << endl;
            disconnected = true;
            break;
        }
        if (bytes_read == -1) {
            // Error, interrupted or no data
            if (errno == EINTR) continue; // interrupted
            if (errno == EAGAIN || errno == EWOULDBLOCK) break; // no data

            if (done) handleData(buf);
            doError("read on " + get_endpoint()->name() + ": " + string(strerror(errno)));
            return;
        }
        if (bytes_read > chunk_size)
            throw Exception("too many bytes read");
        done += bytes_read;
    } while (bytes_read > 0);

    try {
        if (done > buf.length())
            throw Exception("buffer is too long");

        buf.resize(done);

        if (done) handleData(buf);
    } catch (...) {
        if (disconnected) {
            handleDisconnect();
        }
        throw;
    }

    if (disconnected) {
        handleDisconnect();
    }
}

void
PassiveConnectionHandler::
handleOutput()
{
    transport().assertLockedByThisThread();
        
    if (toWrite.empty()) {
#if 0
        // For some reason, we sometimes get a bogus call here.  Deal with
        // it.
        //cerr << "BOGUS handle_output" << endl;
        //transport().activities.dump();
        reactor()->remove_handler(get_handler(),
                                  ACE_Event_Handler::WRITE_MASK
                                  | ACE_Event_Handler::DONT_CALL);
        //stopWriting();
        return;
#endif
        transport().activities.dump();
        
        throw Exception("handle_output with empty buffer");
    }

    //double elapsed = Date::now().secondsSince(toWrite.front().date);
    //cerr << "output: elapsed = " << format("%.1fms", elapsed * 1000)
    //     << endl;

    const string & str = toWrite.front().data;

    //cerr << "writing " << str << endl;

    int len = str.length();

    if (done < 0 || (done >= len && len != 0))
        throw Exception("invalid done");

    /* Send data */
    ssize_t written
        = ConnectionHandler::
        send(str.c_str() + done, len - done, MSG_NOSIGNAL | MSG_DONTWAIT);

    if (written == -1 && errno == EWOULDBLOCK) {
        //cerr << "write would block" << endl;
        return;
    }    

    if (written == -1) {
        doError("writing: " + string(strerror(errno)));
        return;
    }
            
    done += written;
        
    if (done == len) {
        //cerr << "SEND FINISHED " << str << endl;

        WriteEntry entry = toWrite.front();
        if (entry.onWriteFinished)
            entry.onWriteFinished();

        toWrite.pop_front();
        done = 0;
        
        if (toWrite.empty())
            stopWriting();

        if (entry.next == NEXT_CONTINUE)
            return;

        if (!toWrite.empty())
            throw Exception("CLOSE or RECYCLE with data to write");

        if (entry.next == NEXT_CLOSE) {
            closeWhenHandlerFinished();
        }
        else if (entry.next == NEXT_RECYCLE) {
            recycleWhenHandlerFinished();
        }
        else throw Exception("invalid next action");
    }
}

void
PassiveConnectionHandler::
send(const std::string & str,
     NextAction next,
     OnWriteFinished onWriteFinished)
{
    // If we're not in the right thread, then set the send up to be
    // asynchronous.
    if (!transport().lockedByThisThread()) {
        doAsync([=] () { this->send(str, next, onWriteFinished); },
                "deferredSend");
        return;
    }

    //cerr << "message being sent<" << str << "> on handle" << transport().getHandle() <<  endl;
    transport().assertLockedByThisThread();

    WriteEntry entry;
    entry.date = Date::now();
    entry.data = str;
    entry.next = next;
    entry.onWriteFinished = onWriteFinished;

    //if (str.find("POST") != 0)
    //    cerr << "SEND " << str << endl;

    toWrite.push_back(entry);

    if (toWrite.size() == 1) {
        done = 0;

        if (transport().getHandle() == -1) {
            doError("send: connection closed by peer");
            return;
        }
        
        startWriting();
    }

    // Don't allow nested invocations of handle_output
    if (inSend) return;

    inSend = true;
    Call_Guard clearInSend([&] () { inSend = false; });

    // Try a preemptive (non-blocking) send to avoid going through the
    // reactor
    handleOutput();
}

void
PassiveConnectionHandler::
handleTimeout(Date time, size_t)
{
    cerr << status() << endl;
    throw Exception("timeout with no handler set");
}

} // namespace Datacratic

