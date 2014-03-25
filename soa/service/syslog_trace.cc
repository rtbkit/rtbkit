/* syslog_trace.cc
   Mathieu Stefani, 23 September 2013

   Utility to collect RTBKit traces from syslog
*/

#include <array>

#include <iostream>
#include <string>
#include <sstream>
#include <chrono>
#include <mutex>

#include "soa/jsoncpp/json.h"
#include "soa/service/nprobe.h"
#include "soa/service/service_base.h"
#include "soa/service/rest_service_endpoint.h"
#include "soa/service/rest_request_router.h"
#include "soa/service/rest_request_binding.h"

#include <boost/program_options/cmdline.hpp>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/positional_options.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>

#include "soa/service/service_utils.h"
#include "jml/arch/spinlock.h"

namespace {

constexpr size_t MaxEntries = 1 << 16;

}

// See Ring::add below for the reason of this assertion
static_assert(!(MaxEntries & (MaxEntries - 1)), "MaxEntries must be 2^M");

using namespace Datacratic;

struct TracingRestEndpoint : public ServiceBase, public RestServiceEndpoint {

    struct TraceEntry {
        struct Context {
            std::string kind;
            int64_t freq;
            std::string uniq;
        } context;

        struct Span {
            int64_t tid;
            int64_t id;
            int64_t parent_id;
            std::string tag;
            int64_t pid;

            std::chrono::nanoseconds start;
            std::chrono::nanoseconds end;
        } span;

        std::string hostname;

        static TraceEntry fromJson(const Json::Value &root) {
            TraceEntry entry;
            try {
                const auto tid = root["tid"].asInt();
                const auto hostname = root["host"].asString();

                const auto id = root["id"].asInt();
                const auto parent_id = root["pid"].asInt();
                const auto kind = root["kind"].asString();
                const auto tag = root["tag"].asString();
                const auto uniq = root["uniq"].asString();
                const int freq = root["freq"].asInt();
                const auto pid = root["kpid"].asInt();

                const auto start = std::chrono::nanoseconds { root["t1"].asInt() };
                const auto end = std::chrono::nanoseconds { root["t2"].asInt() };

                entry.context = { kind, freq, uniq };
                entry.span = { tid, id, parent_id, tag, pid, start, end };
                entry.hostname = hostname;

            } catch (const std::runtime_error &e) {
            }

            return entry;
        }

        std::string print() const {
            std::ostringstream oss;
            oss << "TraceEntry { ";
            oss << " span { " 
                    << "tid = " << span.tid
                    << ", id = " << span.id << ", parent_id = " << span.parent_id
                    << ", tag = " << span.tag << ", pid = " << span.pid << " }"
                << " context { "
                    << "kind = " << context.kind << ", uniq = " << context.uniq
                    << ", freq = " << context.freq << " }"
                << " }";
            return oss.str();
        }
    };

    TracingRestEndpoint(
            const std::shared_ptr<ServiceProxies> &proxies,
            const std::string &name = "tracing-service"
            )
        : ServiceBase(name, proxies), RestServiceEndpoint(getServices()->zmqContext)
        {
            registerServiceProvider(serviceName(), { "tracing" });
            init(getServices()->config, serviceName());
            installRoutes();
        }


    int run(const std::string &fifoPath) {
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
    struct Ring {
        Ring()
           : index { 0 }
           , full { false }
        { }

        typedef std::array<TraceEntry, MaxEntries> value_type;

        void add(TraceEntry entry) {
            entries[index] = std::move(entry);

            /* Little optimization trick here. If we know that MaxEntries is a power
             * of 2, we can replace the modulo operation to compute the ring buffer index
             * with a bitwise mask.
             *
             * This is why MaxEntries must be a power of 2, hence the static assertion
             * above
             */
            index = (index + 1) & (MaxEntries - 1);

            if (!index)
                full = true;
            }

        bool isFull() const {
            return full;
        }

        value_type::const_iterator begin() const {
            return std::begin(entries);
        }

        value_type::const_iterator end() const {
            auto it = begin();
            if (JML_LIKELY(full))
                it = std::end(entries);
            else
                std::advance(it, index);

            return it;
        }

    private:
        uint64_t index;
        bool full;
        value_type entries;
    } ring;

    RestRequestRouter restRouter;
    typedef ML::Spinlock Lock;
    typedef std::lock_guard<Lock> Guard;

    mutable Lock ringLock;

    struct StatsEntry {
        StatsEntry(const std::string &tag, const std::vector<TraceEntry> &serie)
           : tag { tag }
           , centile_99 { 0.0 }
           , mean { 0.0 }
           , median { 0.0 }
           , serie_ { serie }
          {  
             compute();
          }

        StatsEntry(StatsEntry &&other) = default;
        StatsEntry(const StatsEntry &other) = default;

        void compute() 
        {
            if (serie_.empty())
                return;

            sort(begin(serie_), end(serie_),
                [this](const TraceEntry &lhs, const TraceEntry &rhs) {
                    const auto duration_lhs = duration(lhs);
                    const auto duration_rhs = duration(rhs);

                    return duration_lhs < duration_rhs;
            });

            const auto size = serie_.size();
            const auto rank_99 = int { round(0.99 * (size - 1)) };
            const auto &entry_99 = serie_[rank_99];
            centile_99 =  duration(entry_99);

            auto acc = [this](double current, const TraceEntry &entry)
                      {
                          return current + duration(entry);
                      };

            mean = accumulate(begin(serie_), end(serie_), 0.0, acc) / serie_.size();
            
            const auto medianIndex = size / 2;
            if (medianIndex % 2 == 0)
                median = duration(serie_[medianIndex]);
            else
                median = (duration(serie_[medianIndex]) + 
                          duration(serie_[medianIndex - 1])) / 2.0;

        }    

        std::string tag;
        double centile_99;
        double mean;
        double median;

        Json::Value toJson() const {
            Json::Value value;
            value["tag"] = tag;
            value["centile_99"] = centile_99;
            value["mean"] = mean;
            value["median"] = median;
            return value;
        }

        private:
            double duration(const TraceEntry &entry) {
                return std::chrono::duration_cast<std::chrono::milliseconds>(
                        entry.span.end - entry.span.start).count();
            }
            std::vector<TraceEntry> serie_;
    };

    struct ObjectStats {
        ObjectStats(const std::string &kind)
            : kind { kind }
        { }

        void addEntry(StatsEntry entry) {
            values.push_back(std::move(entry));
        }

        Json::Value toJson() const {
            Json::Value value;
            Json::Value data;
            std::for_each(begin(values), end(values), [&](const StatsEntry &entry) {
                value["kind"] = kind;
                data.append(entry.toJson());
            });

            value["data"] = data;

            return value;
        }


        std::string kind;
    private:
        std::vector<StatsEntry> values;
    };

    struct GlobalStats : public std::vector<ObjectStats> {
        Json::Value toJson() const {
            Json::Value root;
            std::for_each(begin(), end(), [&](const ObjectStats &stats) {
                root.append(stats.toJson());
            });

            return root;
        }
    };

    void installRoutes() {
        auto &v0Router = restRouter.addSubRouter(
                "/v0", "Simple tracing REST interface");

        addRouteSyncReturn(
                v0Router, "/stats", { "GET" },
                "Returns raw statistics",
                "Accumulated statistics from collected traces",
                [](const GlobalStats &stats) { return stats.toJson(); },
                &TracingRestEndpoint::computeStats,
                this
                );
    }


    bool handleMessage(const std::string &message) {

        Json::Value root;
        Json::Reader reader;
        bool ok = reader.parse(message, root);
        if (!ok) {
            return false;
        }

        auto entry = TraceEntry::fromJson(root);
        /* We need a lock here because the ring might also be read by the 
         * REST thread
         *
         * TODO: lock-free on write-side ?
         */
        {
            Guard guard(ringLock);
            ring.add(std::move(entry));
        }

        return true;
    }

    GlobalStats computeStats() const {
        typedef std::map<std::string, std::vector<TraceEntry>> TracingData;
        /* Maps object type (kind) to tracing data */
        std::map<std::string, TracingData> data;

        {
            Guard guard(ringLock);
            std::for_each(std::begin(ring), std::end(ring), [&](const TraceEntry &entry) {
                auto &tracingData = data[entry.context.kind];
                auto &vec = tracingData[entry.span.tag];
                vec.push_back(entry);
            });
        }

        GlobalStats stats;

        for (const auto &kind: data) {
            auto tags = kind.second;

            ObjectStats objStats(kind.first);

            for (const auto &tag: tags) {
                objStats.addEntry(StatsEntry(tag.first, tag.second));
            }

            stats.push_back(objStats);

        }

        return stats;

    }

    void handleRequest(const ConnectionId &conn, const RestRequest &request) const
    {
        restRouter.handleRequest(conn, request);
    }
};

int main(int argc, const char *argv[]) {

    ServiceProxyArguments serviceArgs;

    std::string fifoPath;
    namespace po = boost::program_options;

    po::options_description options;
    options
        .add(serviceArgs.makeProgramOptions());

    options.add_options() 
        ("path,p", po::value<std::string>(&fifoPath),
              "path of the fifo where logs are stored")
        ("help,h", "Display this help message");

    po::variables_map vm;
    try {
        po::store(po::command_line_parser(argc, argv).options(options).run(), vm);
        po::notify(vm);
    } catch (const std::exception &e) {
        std::cout << options << std::endl;
        return 1;
    }

    if (vm.count("help")) {
        std::cout << options << std::endl;
        return 0;
    }

    auto proxies = serviceArgs.makeServiceProxies();
    auto serviceName = serviceArgs.serviceName("tracing-service");
    TracingRestEndpoint endpoint(proxies, serviceName);
    endpoint.bindTcp(PortRange { 3481 }, PortRange { 3482 });
    endpoint.start();
    endpoint.run(fifoPath);

}
