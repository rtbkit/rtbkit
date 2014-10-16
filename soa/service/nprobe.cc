/** nprobe.cc                                 -*- C++ -*-
    Jan Sulmont, 17 Sep 2013
    Copyright (c) 2013 Datacratic.  All rights reserved.

    Stuff...

*/

#include "nprobe.h"
#include "jml/arch/info.h"

#include <sstream>
#include <thread>
#include <syslog.h>
#include <chrono>
#include <iostream>

namespace Datacratic
{

namespace
{

struct syslog_init {
    syslog_init() {
        ::openlog("RTBkit", LOG_PID, LOG_USER);
    }

} syslog_init_;

} // anonymous namespace

void syslog_probe_sink(const ProbeCtx& ctx, const std::vector<Span>& vs)
{
    using namespace std::chrono;
    using std::get;
    static auto hostname = ML::hostname();
    static auto pid = ::getpid();
    auto format = [&] (Span const& s) {
        std::ostringstream oss;
        oss << "{"
        << "\"tid\":" << std::this_thread::get_id()
        << ",\"host\":\"" << hostname << "\""
        << ",\"kpid\":" << pid
        << ",\"kind\":\"" << get<0>(ctx) << "\""
        << ",\"uniq\":\"" << get<1>(ctx) << "\""
        << ",\"freq\":" << get<2>(ctx)
        << ",\"pid\":" << s.pid_
        << ",\"id\":" << s.id_
        << ",\"tag\":\"" << s.tag_ << "\""
        << ",\"t1\":" << duration_cast<nanoseconds>(s.start_.time_since_epoch()).count()
        << ",\"t2\":" << duration_cast<nanoseconds>(s.end_.time_since_epoch()).count()
        << "}";
        return oss.str();
    };
    std::cerr << vs.size() << " spans" << std::endl;
    for (const auto& s: vs)
    {
        const auto str = format(s);
        std::cerr << str << std::endl ;
        syslog (LOG_INFO, "%s", str.c_str());
    }
}


} // Datacratic namespace
