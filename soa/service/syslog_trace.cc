/* syslog_trace.cc
   Mathieu Stefani, 23 September 2013

   Utility to collect RTBKit traces from syslog
*/

#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <array>

#include <iostream>
#include <string>
#include <sstream>
#include <chrono>

#include "soa/jsoncpp/json.h"
#include "soa/service/nprobe.h"

namespace {

constexpr size_t MaxEntries = 1 << 3;

}

struct App {

    struct TraceEntry {
        int64_t tid;
        std::string hostname;
        int64_t id;
        int64_t parent_id;
        std::string tag;
        std::string uniq;
        int64_t freq;
        int64_t pid;

        std::chrono::nanoseconds t1;
        std::chrono::nanoseconds t2;

        static TraceEntry fromJson(const Json::Value &root) {
            try {
                const auto tid = root["tid"].asInt();
                const auto hostname = root["host"].asString();

                const auto id = root["id"].asInt();
                const auto parent_id = root["pid"].asInt();
                const auto tag = root["tag"].asString();
                const auto uniq = root["uniq"].asString();
                const int freq = root["freq"].asInt();
                const auto pid = root["kpid"].asInt();

                const auto t1 = std::chrono::nanoseconds { root["t1"].asInt() };
                const auto t2 = std::chrono::nanoseconds { root["t2"].asInt() };

                return TraceEntry { tid, hostname, id, parent_id, tag, 
                                    uniq, freq, pid, t1, t2 };
            } catch (const std::runtime_error &e) {
            }

            return TraceEntry { };
        }

        std::string print() const {
            std::ostringstream oss;
            oss << "TraceEntry { ";
            oss << "tid=" << tid << ", hostname=" << hostname
                << ", id=" << id << ", parent_id=" << parent_id
                << ", tag=" << tag << ", uniq=" << uniq
                << ", freq=" << freq << ", pid=" << pid
                << ", t1=" << t1.count() << ", t2=" << t2.count()
                << " }";
            return oss.str();
        }
    };

    int exec(const std::string &fifoPath) {
        int fd = open(fifoPath.c_str(), O_RDONLY);

        if (fd == -1) {
            ::perror("open");
            return 1;
        }

        char c;
        ssize_t bytes { 0 };
        std::string message;
        bool inMessage { false };
        for (;;) {
            if ((bytes = read(fd, &c, 1)) == -1) {
                ::perror("read");
                return 1;
            }
            
            if (c == '}') {
                message += c;
                if (!handleMessage(message)) 
                    std::cerr << "Failed to handle message: " << message << std::endl;
                message.clear();
                inMessage = false;
            } else if (c == '{') {
                inMessage = true;
            }

            if (inMessage) {
                message += c;
            }

        }
    }

private:
    std::array<TraceEntry, MaxEntries> entries;
    uint64_t index;

    bool handleMessage(const std::string &message) {

        Json::Value root;
        Json::Reader reader;
        bool ok = reader.parse(message, root);
        if (!ok) {
            return false;
        }

        auto entry = TraceEntry::fromJson(root);
#if 1
        std::cout << entry.print() << std::endl;
#endif

        entries[index & (MaxEntries - 1)] = std::move(entry);
        ++index;

        return true;
    }
};

void usage() {
    std::cout << "usage: syslog_aggregator fifo-path" << std::endl;
}

int main(int argc, const char *argv[]) {
    if (argc == 1) {
        usage();
        return 0;
    }

    App app;
    return app.exec(argv[1]);
}
