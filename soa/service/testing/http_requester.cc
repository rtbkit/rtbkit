#include <iostream>
#include <memory>
#include <string>
#include <tuple>

#include <boost/program_options/cmdline.hpp>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>

#include "jml/utils/testing/watchdog.h"
#include "soa/service/message_loop.h"
#include "soa/service/rest_proxy.h"
#include "soa/service/http_client.h"


using namespace std;
using namespace Datacratic;
using namespace boost::program_options;

int main(int argc, char* argv[]){

    int num_reqs(0), num_resps(0), concurrency(1), max_reqs(1), 
        timeout(1000), connect_timeout(1000);
    string base_url = "http://127.0.0.1:8000";

    options_description all_opt;
    all_opt.add_options()
        ("url,u", value(&base_url),
         "URL for the requests")
        ("concurrency,c", value(&concurrency),
         "Number of concurrent requests")
        ("requests,r", value(&max_reqs),
         "total of number of requests to perform")
        ("request-timeout-ms,t", value(&timeout),
         "request timeout in milliseconds")
        ("connect-timeout-ms,i", value(&connect_timeout),
         "connect timeout in milliseconds")
        ("help,h", "show help");

    variables_map vm;
    store(command_line_parser(argc, argv)
          .options(all_opt)
          .run(),
          vm);
    notify(vm);

    if (vm.count("help")) {
        cerr << all_opt << endl;
        return 1;
    }

    MessageLoop loop;
    loop.start();

    //create the http client
    auto client = std::make_shared<HttpClient>(
                        base_url, concurrency);
    
    loop.addSource("httpClient", client);

    for(auto i = 0; i < max_reqs; i++){

        // create the callback    
        auto onResponse = [&,i] ( const HttpRequest & rq, 
                                HttpClientError errorCode_,
                                int status, string && headers, string && body) {
            cout << "received response " << i << endl;
            cout << "body : " << body << endl;
            cout << "errorCode_ : " << errorCode_ << endl;
            cout << "--------------" << endl;
            ML::futex_wake(num_reqs);
            ++num_resps;
        };
        auto cbs = make_shared<HttpClientSimpleCallbacks>(onResponse);

        client->get("", cbs, RestParams(),
                 RestParams({
                    {"Content-Type", "application/json"},
                    {"Connection", "keep-alive"}
                }), timeout, connect_timeout);
        
    }
    while(!num_reqs){
        int old(num_reqs);
        ML::futex_wait(num_reqs, old);
        if(num_resps == max_reqs)
            num_reqs = true;
    }
    loop.removeSource(client.get());
    return 0;
}
