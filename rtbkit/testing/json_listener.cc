/* json_feeder.cc
   Wolfgang Sourdeau, March 2013
   Copyright (c) 2013 Datacratic.  All rights reserved.

   A utility that listens to JSON requests on a specific port and logs them to
   disk
 */


#include <stdio.h>

#include <boost/program_options/cmdline.hpp>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/positional_options.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>

#include <jml/utils/ring_buffer.h>

#include <soa/service/rest_service_endpoint.h>


using namespace std;
using namespace boost::program_options;
using namespace Datacratic;

const int BufferSizePower = 4;

const int BufferSize = (1 << BufferSizePower);
const int BufferMask = BufferSize - 1;

struct JsonListener : public RestServiceEndpoint {
    JsonListener(const std::shared_ptr<zmq::context_t> & context)
        : RestServiceEndpoint(context),
          logFile_(NULL),
          history_(10), position_(0)
    {
        history_.resize(BufferSize);
    }

    ~JsonListener()
    {
        shutdown();
    }

    void init(std::shared_ptr<ConfigurationService> config,
              const std::string & endpointName,
              const std::string & filename)
    {
        logRequest = bind(&JsonListener::doLogRequest,
                          this, placeholders::_1, placeholders::_2);

        if (!logFile_) {
            logFile_ = fopen(filename.c_str(), "w");
            if (!logFile_)
                throw ML::Exception(errno,
                                    "could not open log file '" + filename
                                    + "'", "init");
        }
        RestServiceEndpoint::init(config, endpointName);

        addPeriodic("JsonListener::updatePosition", 1.0,
                    bind(&JsonListener::updatePosition, this),
                    true /* single threaded */);
    }

    void startSync()
    {
        startTime_ = Date::now();
        MessageLoop::startSync();
    }

    bool isStatsRequest(const RestRequest & req)
        const
    {
        return (req.verb == "GET" && req.resource == "/stats");
    }

    void updatePosition()
    {
        uint32_t newPosition((position_ + 1) & BufferMask);
        uint32_t & counter_ = history_[newPosition];
        counter_ = 0;
        position_ = newPosition;
        fflush (logFile_);
    }

    void doLogRequest(const ConnectionId & conn, const RestRequest & req)
    {
        if (!isStatsRequest(req)) {
            Guard lock(historyLock);

            /* request accounting */
            uint32_t & counter_ = history_[position_];
            counter_++;

            Date now = Date::now();
            fprintf (logFile_, "%f\n%s\n",
                     now.secondsSinceEpoch(), req.payload.c_str());
        }
    }

    void handleRequest(const ConnectionId & conn, const RestRequest & req)
        const
    {
        Json::Value nothing;

        if (isStatsRequest(req)) {
            uint32_t totalReq(0);
            float meanReqPerSec;

            for (const uint32_t & reqPerSec: history_) {
                totalReq += reqPerSec;
            }
            meanReqPerSec = float(totalReq) / BufferSize;

            string response("total (" + to_string(BufferSize) + " secs):"
                            + to_string(totalReq) + "\n"
                            + "mean: " + to_string(meanReqPerSec) + " req./sec\n");

            conn.sendResponse(200, response.c_str(), "text/plain");
        }
        else {
            conn.sendResponse(204, "");
        }
    }

    void shutdown()
    {
        if (logFile_) {
            fflush(logFile_);
            fclose(logFile_);
            logFile_ = NULL;
        }
    }

    Date startTime_;

    std::FILE *logFile_;

    /* request accounting */
    typedef unique_lock<mutex> Guard;
    mutable mutex historyLock;

    vector<uint32_t> history_;
    int position_;
};


int main(int argc, char *argv[])
{
    int port(0);
    string filename;

    {
        using namespace boost::program_options;

        options_description configuration_options("Configuration options");

        configuration_options.add_options()
            ("port,p", value(&port),
             "port to listen on")
            ("filename,f", value(&filename),
             "filename");
 
        options_description all_opt;
        all_opt.add(configuration_options);
        all_opt.add_options()
            ("help,h", "print this message");

        variables_map vm;
        store(command_line_parser(argc, argv)
              .options(all_opt)
              .run(),
              vm);
        notify(vm);

        if (vm.count("help")) {
            cerr << all_opt << endl;
            exit(1);
        }

        if (!port) {
            cerr << "'port' parameter is required and must be > 0" << endl;
            exit(1);
        }

        if (filename.empty()) {
            cerr << "'filename' parameter is required" << endl;
            exit(1);
        }
    }

    auto proxies = make_shared<ServiceProxies>();

    JsonListener listener(proxies->zmqContext);
    listener.init(proxies->config, "listener", filename);
    listener.bindFixedHttpAddress("*", port);

    listener.startSync();

    return 0;
}
