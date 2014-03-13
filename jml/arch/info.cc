/* info.cc
   Jeremy Barnes, 21 February 2007
   Copyright (c) 2007 Jeremy Barnes.  All rights reserved.

*/

#include "info.h"
#include <unistd.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netdb.h>
#include <pwd.h>
#include <errno.h>
#include "jml/arch/exception.h"
#include "jml/arch/cpuid.h"
#include <fstream>
#include <iostream>
#include <sys/time.h>
#include <time.h>
#include "jml/arch/cpu_info.h"
#include "jml/utils/guard.h"
#include <boost/bind.hpp>
#include <dirent.h>
#include "jml/arch/format.h"


using namespace std;

namespace ML {

int userid()
{
    return getuid();
}

std::string userid_to_username(int userid)
{
    struct passwd pwbuf;
    size_t buflen = sysconf(_SC_GETPW_R_SIZE_MAX);
    char buf[buflen];
    struct passwd * result = 0;

    int res = getpwuid_r(userid, &pwbuf, buf, buflen, &result);

    if (res != 0)
        throw Exception(errno, "userid_to_username()", "getpwuid_r");

    if (result == 0)
        throw Exception("usedid_to_username(): userid unknown");
    
    return result->pw_name;
}

std::string username()
{
    return userid_to_username(userid());
}

std::string hostname()
{
    char buf[128];
    int res = gethostname(buf, 128);
    if (res != 0)
        throw Exception(errno, "hostname", "hostname");
    buf[127] = 0;
    return buf;
}

std::string fqdn_hostname(std::string const & port)
{
    std::string host = hostname();

    struct addrinfo hints;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_flags = AI_CANONNAME;

    struct addrinfo * info = 0;
    int res = getaddrinfo(host.c_str(), port.c_str(), &hints, &info);
    if (res != 0)
        throw Exception(errno, "getaddrinfo: %s", gai_strerror(res));

    if(!info)
        throw Exception("fqdn_hostname(): no info");

    std::string fqdn(info->ai_canonname);
    freeaddrinfo(info);
    return fqdn;
}

std::string now()
{
    struct timeval tv;

    int res = gettimeofday(&tv, NULL);
    if (res != 0)
        throw Exception(errno, "now", "gettimeofday");
    
    char buf[128];
    ctime_r(&tv.tv_sec, buf);

    if (res != 0)
        throw Exception(errno, "now", "ctime");
    
    return buf;
}

std::string all_info()
{
    return now() + " " + username() + " " + hostname();
}

size_t num_open_files()
{
    DIR * dfd = opendir("/proc/self/fd");
    if (dfd == 0)
        throw Exception("num_open_files(): opendir(): "
                        + string(strerror(errno)));

    Call_Guard closedir_dfd(boost::bind(closedir, dfd));

    size_t result = 0;
    
    dirent entry;
    for (dirent * current = &entry;  current;  ++result) {
        int res = readdir_r(dfd, &entry, &current);
        if (res != 0)
            throw Exception("num_open_files(): readdir_r: "
                            + string(strerror(errno)));
    }

    return result;
}

std::string fd_to_filename(int fd)
{
    if (fd == -1)
        throw Exception("fd_to_filename(): invalid filename");

    size_t buffer_size = 256;

    string fn = format("/proc/self/fd/%d", fd);

    for (;;) {
        char buf[buffer_size];
        ssize_t ret = readlink(fn.c_str(), buf, buffer_size);
        if (ret == -1 && errno == -ENOENT)
            throw Exception("fd_to_filename(): fd is unknown");
        if (ret == -1)
            throw Exception(errno, "fd_to_filename", "readlink()");

        if (ret == buffer_size) {
            if (buffer_size > 1024)
                throw Exception("filename too long");
            buffer_size *= 2;
            continue;
        }

        buf[ret] = 0;
        return buf;
    }
}

} // namespace ML
