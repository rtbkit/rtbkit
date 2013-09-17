#include <sstream>
#include <thread>
#include <syslog.h>
#include <chrono>
#include <iostream>
#include <boost/asio/ip/host_name.hpp>
#include "soa/service/nprobe.h"

namespace RTBKIT
{
namespace detail
{
struct syslog_init {
    syslog_init() {
        ::openlog("RTBkit", LOG_PID, LOG_LOCAL7);
    }
};
syslog_init syslog_init_ ;

void default_probe_sink(const RTBKIT::ProbeCtx& ctx, const std::vector<RTBKIT::Span>& vs)
{
    using namespace std::chrono;
    using std::get;
    static auto hostname = boost::asio::ip::host_name();
    static auto pid = ::getpid();
    auto format = [&] (RTBKIT::Span const& s) {
        std::ostringstream oss;
        oss << "{"
        << "\"tid\":\"" << std::this_thread::get_id() << "\""
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
    for (const auto& s: vs)
    {
        const auto str = format(s);
        std::cerr << str << std::endl ;
        syslog (LOG_INFO, "%s", str.c_str());
    }
}
}

template <typename T>
SinkCb
Trace<T>::S_sink_ = detail::default_probe_sink ;
}
