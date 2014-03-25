/* syslog_trace.cc
   Mathieu Stefani, 16 October 2013

   Simple REST client to display tracing results
*/

#include <iostream>
#include "soa/jsoncpp/json.h"
#include "soa/service/service_base.h"
#include "soa/service/service_utils.h"
#include "soa/service/rest_proxy.h"
#include "soa/service/http_rest_proxy.h"

#include <boost/program_options/cmdline.hpp>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/positional_options.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>

using namespace Datacratic;
using namespace std;

void displayResults(const Json::Value &data) {
    for (const auto &value: data) {
        const auto &kind = value["kind"].asString();
        cout << kind << endl;
        cout << string(kind.size(), '-') << endl;
        const auto &data = value["data"];
        for (const auto &span: data) {
            cout << string(4, ' ') << span["tag"].asString() << ":" << endl;
            cout << string(8, ' ') << "99th centile: " << span["centile_99"].asDouble() 
                                   << "ms" << endl;
            cout << string(8, ' ') << "median      : " << span["median"].asDouble() 
                                   << "ms" << endl;
        }
        cout << endl;
    }
}

int main(int argc, const char *argv[]) {
    ServiceProxyArguments serviceArgs;

    namespace po = boost::program_options;
    po::options_description options;

    options.
        add(serviceArgs.makeProgramOptions());
    options.add_options()
        ("help,h", "Show this help message");


    po::variables_map vm;
    try {
        po::store(po::command_line_parser(argc, argv).options(options).run(), vm);
        po::notify(vm);
    } catch (const std::exception &e) {
        cerr << options << endl;
        return 1;
    }

    if (vm.count("help"))
        cout << options << endl;

    RestProxy proxy;
    auto proxies = serviceArgs.makeServiceProxies();

    proxy.initServiceClass(proxies->config, "tracing", "zeromq"); 

    proxy.start();

    for (;;) {
        proxy.push([&](std::exception_ptr, int responseCode,
                                          const std::string &json) {
            Json::Value root;
            Json::Reader reader;
            bool ok = reader.parse(json, root);
            if (!ok)
                cerr << "Failed to parse json response" << endl;

            if (!root.isNull())
                displayResults(root);
            else
                cerr << "Received null response" << endl;

        }, "GET", "/v0/stats"); 

        ML::sleep(5);
    }
}
