/** fd_exhauster.h                                                 -*- C++ -*-
    Jeremy Barnes, 16 May 2011
    Copyright (c) 2011 Datacratic.  All rights reserved.

    Class to exhaust file descriptors for testing purposes.
*/

#ifndef __jml_testing__fd_exhauster_h__
#define __jml_testing__fd_exhauster_h__

#include <boost/test/unit_test.hpp>
#include <sys/types.h>
#include <sys/socket.h>

namespace ML {

// Create sockets until no FDs left to exhaust FDs
struct FDExhauster {
    std::vector<int> sockets;

    FDExhauster()
    {
        int sock;
        do {
            sock = socket(AF_INET, SOCK_STREAM, 0);
            if (sock != -1)
                sockets.push_back(sock);
        } while (sockets.size() < 65536 && sock != -1);

        // If this fails, we can't extinguish the sockets
        BOOST_REQUIRE_EQUAL(sock, -1);
    }

    ~FDExhauster()
    {
        for (unsigned i = 0;  i < sockets.size();  ++i)
            close(sockets[i]);
    }
};


} // namespace ML

#endif /* __jml_testing__fd_exhauster_h__ */
