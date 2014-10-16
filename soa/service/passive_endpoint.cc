/* passive_endpoint.cc
   Jeremy Barnes, 29 April 2011
   Copyright (c) 2011 Datacratic.  All rights reserved.

   Passive endpoint implementation.
*/

#include <unordered_map>

#include "jml/arch/futex.h"
#include "soa/service//passive_endpoint.h"
#include <poll.h>
#include <boost/date_time/gregorian/gregorian.hpp>

using namespace std;
using namespace ML;
using namespace boost::posix_time;

namespace Datacratic {


/*****************************************************************************/
/* PASSIVE ENDPOINT                                                          */
/*****************************************************************************/

PassiveEndpoint::
PassiveEndpoint(const std::string & name)
    : EndpointBase(name)
{
}

PassiveEndpoint::
~PassiveEndpoint()
{
    closePeer();
    shutdown();
}

int
PassiveEndpoint::
init(PortRange const & portRange, const std::string & hostname, int num_threads, bool synchronous,
     bool nameLookup, int backlog)
{
    //static const char *fName = "PassiveEndpoint::init:";
    //cerr << fName << this << ":was called for " << hostname << endl;
    spinup(num_threads, synchronous);

    int port = listen(portRange, hostname, nameLookup, backlog);
    cerr << "listening on hostname " << hostname << " port " << port << endl;
    return port;
}


/*****************************************************************************/
/* ACCEPTOR FOR SOCKETTRANSPORT                                              */
/*****************************************************************************/

AcceptorT<SocketTransport>::
AcceptorT()
    : fd(-1), endpoint(0), listening_(false)
{
}

AcceptorT<SocketTransport>::
~AcceptorT()
{
    closePeer();
}

int
AcceptorT<SocketTransport>::
listen(PortRange const & portRange,
       const std::string & hostname,
       PassiveEndpoint * endpoint,
       bool nameLookup,
       int backlog)
{
    closePeer();
    
    this->endpoint = endpoint;
    this->nameLookup = nameLookup;

    fd = socket(AF_INET, SOCK_STREAM, 0);

    // Avoid already bound messages for the minute after a server has exited
    int tr = 1;
    int res = setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &tr, sizeof(int));

    if (res == -1) {
        close(fd);
        fd = -1;
        throw Exception("error setsockopt SO_REUSEADDR: %s", strerror(errno));
    }

    const char * hostNameToUse
        = (hostname == "*" ? "0.0.0.0" : hostname.c_str());

    int port = portRange.bindPort
        ([&](int port)
         {
             addr = ACE_INET_Addr(port, hostNameToUse, AF_INET);

             //cerr << "port = " << port
             //     << " hostname = " << hostname
             //     << " addr = " << addr.get_host_name() << " "
             //     << addr.get_host_addr() << " "
             //     << addr.get_ip_address() << endl;

             int res = ::bind(fd,
                              reinterpret_cast<sockaddr *>(addr.get_addr()),
                              addr.get_addr_size());
             if (res == -1 && errno != EADDRINUSE)
                 throw Exception("listen: bind returned %s", strerror(errno));
             return res == 0;
         });
    
    if (port == -1) {
        throw Exception("couldn't bind to any port in range [%d,%d]", portRange.first,
                                                            portRange.last);
    }

    // Avoid already bound messages for the minute after a server has exited
    res = setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &tr, sizeof(int));

    if (res == -1) {
        close(fd);
        fd = -1;
        throw Exception("error setsockopt SO_REUSEADDR: %s", strerror(errno));
    }

    res = ::listen(fd, backlog);

    if (res == -1) {
        close(fd);
        fd = -1;
        throw Exception("error on listen: %s", strerror(errno));
    }

    if (port == 0) {
        sockaddr_in inAddr;
        socklen_t inAddrLen = sizeof(inAddr);
        res = ::getsockname(fd, (sockaddr *) &inAddr, &inAddrLen);
        port = ntohs(inAddr.sin_port);
        addr.set(&inAddr, inAddrLen);
    }

    listening_ = true;
    ML::futex_wake(listening_);

    shutdown = false;

    acceptThread.reset(new boost::thread([=] () { this->runAcceptThread(); }));
    return port;
}

void
AcceptorT<SocketTransport>::
closePeer()
{
    if (!acceptThread) return;
    shutdown = true;

    ML::memory_barrier();

    wakeup.signal();

    close(fd);
    fd = -1;

    acceptThread->join();
    acceptThread.reset();
}

std::string
AcceptorT<SocketTransport>::
hostname() const
{
    char buf[1024];
    int res = addr.get_host_name(buf, 1024);
    if (res == -1)
        throw ML::Exception("invalid hostname");
    return buf;
}

int
AcceptorT<SocketTransport>::
port() const
{
    return addr.get_port_number();
}

struct NameEntry {
    NameEntry(const string & name)
        : name_(name), date_(Date::now())
        {}

    string name_;
    Date date_;
};

void
AcceptorT<SocketTransport>::
runAcceptThread()
{
    //static const char *fName = "AcceptorT<SocketTransport>::runAcceptThread:";
    unordered_map<string,NameEntry> addr2Name;

    int res = fcntl(fd, F_SETFL, O_NONBLOCK);
    if (res != 0) {
        if (shutdown)
            return;  // deal with race between start up and shut down
        throw ML::Exception(errno, "fcntl");
    }

    while (!shutdown) {

        sockaddr_in addr;
        socklen_t addr_len = sizeof(addr);
        //cerr << "accept on fd " << fd << endl;

        pollfd fds[2] = {
            { fd, POLLIN, 0 },
            { wakeup.fd(), POLLIN, 0 }
        };

        int res = ::poll(fds, 2, -1);

        //cerr << "accept poll returned " << res << endl;

        if (shutdown)
            return;
        
        if (res == -1 && errno == EINTR)
            continue;
        if (res == 0)
            throw ML::Exception("should not be no fds ready to accept");

        if (fds[1].revents) {
            wakeup.read();
        }

        if (!fds[0].revents)
            continue;

        res = accept(fd, (sockaddr *)&addr, &addr_len);

        //cerr << "accept returned " << res << endl;

        if (res == -1 && errno == EWOULDBLOCK)
            continue;

        if (res == -1 && errno == EINTR) continue;

        if (res == -1)
            endpoint->acceptError(format("accept: %s", strerror(errno)));

#if 0
        union {
            char octets[4];
            uint32_t addr;
        } a;
        a.addr = addr.sin_addr;
#endif

        ACE_INET_Addr addr2(&addr, addr_len);

#if 0
        ptime now = second_clock::universal_time();

        cerr << boost::this_thread::get_id() << ":"<<to_iso_extended_string(now) << ":accept succeeded from "
             << addr2.get_host_addr() << ":" << addr2.get_port_number()
             << " (" << addr2.get_host_name() << ")"
             << " for endpoint " << endpoint->name() << " res = " << res
             << " pointer " << endpoint << endl;
#endif
        std::shared_ptr<SocketTransport> newTransport
            (new SocketTransport(this->endpoint));

        newTransport->peer_ = ACE_SOCK_Stream(res);
        string peerName = addr2.get_host_addr();
        if (nameLookup) {
            auto it = addr2Name.find(peerName);
            if (it == addr2Name.end()) {
                string addr = peerName;
                peerName = addr2.get_host_name();
                addr2Name.insert({addr, NameEntry(peerName)});
            }
            else {
                peerName = it->second.name_;
            }
        }

        if (peerName == "<unknown>")
            peerName = addr2.get_host_addr();
        newTransport->peerName_ = peerName;
        endpoint->associateHandler(newTransport);

        /* cleanup name entries older than 5 seconds */
        Date now = Date::now();
        auto it = addr2Name.begin();
        while (it != addr2Name.end()) {
            const NameEntry & entry = it->second;
            if (entry.date_.plusSeconds(5) < now) {
                it = addr2Name.erase(it);
            }
            else {
                it++;
            }
        }
    }
}

void
AcceptorT<SocketTransport>::
waitListening()
    const
{
    while (!listening_) {
        int oldListening = listening_;
        ML::futex_wait(listening_, oldListening);
    }
}

} // namespace Datacratic
