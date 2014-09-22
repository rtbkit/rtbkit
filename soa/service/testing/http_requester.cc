#include <iostream>
#include <memory>
#include <string>
#include <tuple>

#include "jml/utils/testing/watchdog.h"
#include "soa/service/message_loop.h"
#include "soa/service/rest_proxy.h"
#include "soa/service/http_client.h"

#define BASE_URL        "http://api-east1-aws.appaudience.com"
#define CONCURRENT_REQS 1
#define NUM_REQS        1

using namespace std;
using namespace Datacratic;

int main(int argc, char* argv[]){

    MessageLoop loop;
    loop.start();

    int num_reqs(0), num_resps(0);

    //create the http client
    auto client = std::make_shared<HttpClient>(
                        BASE_URL, CONCURRENT_REQS);
    
    loop.addSource("httpClient", client);

    for(auto i = 0; i < NUM_REQS; i++){

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

        client->get("/api", cbs, RestParams({{"zipcode", "10001"}}),
                 RestParams({
                    {"Content-Type", "application/json"},
                    {"Connection", "keep-alive"}
                }), 1000);
        
    }
    while(!num_reqs){
        int old(num_reqs);
        cout << "waiting ... " << endl;
        ML::futex_wait(num_reqs, old);
        if(num_resps == NUM_REQS)
            num_reqs = true;
    }
    loop.removeSource(client.get());
    return 0;
}
