/* info.cc
   Jeremy Barnes, 21 February 2007
   Copyright (c) 2007 Jeremy Barnes.  All rights reserved.

*/

#include "info.h"
#include <unistd.h>
#include <stdio.h>
#include <sys/types.h>
#include <pwd.h>
#include <errno.h>
#include "arch/exception.h"
#include "arch/cpuid.h"
#include <fstream>
#include <iostream>
#include <sys/time.h>
#include <time.h>

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

int num_cpus_result = 0;

void init_num_cpus()
{
    ifstream stream("/proc/cpuinfo");

    int ncpus = 0;

    while (stream) {
        string line;
        getline(stream, line);

        //cerr << "line = " << line << endl;

        string::size_type found = line.find("processor");
        //cerr << "found = " << found << endl;
        if (found == 0) ++ncpus;
    }

    //cerr << "ncpus = " << ncpus << endl;

    if (ncpus == 0)
        ncpus = 1;
    
    num_cpus_result = ncpus;

    //cerr << "num_cpus_result = " << num_cpus_result << endl;
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

} // namespace ML
