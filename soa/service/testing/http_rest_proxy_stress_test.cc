/* http_rest_proxy_stress_test.cc
   Wolfgang Sourdeau, August 2015
   Copyright (c) 2015 Datacratic.  All rights reserved.

   Online stress test that ensures that the OpenSSL locking callbacks works
   as expected.

   The url and the number of threads needs to be tweaked depending on the
   site configuration. The latter cannot be too high nor too low in order to
   cause the relevant race condition. This can be adjusted by "if-0-ing" the
   init function from the openssl_threading module.
*/

#include <thread>
#include <vector>

#include "soa/service/http_rest_proxy.h"


using namespace std;
using namespace Datacratic;


const string TestUrl("https://jigsaw.w3.org");

int main()
{
    int nThreads(150);

    auto threadFn = [&] () {
        HttpRestProxy proxy(TestUrl);
        try {
            auto resp = proxy.get("/");
            ExcAssert(resp.code() > 0);
        }
        catch (...) {
        }
    };

    vector<thread> threads;
    for (int i = 0; i < nThreads; i++) {
        threads.emplace_back(threadFn);
    }

    for (auto & th: threads) {
        th.join();
    }
}
