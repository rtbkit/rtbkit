#include "soa/service/http_client.h"

using namespace Datacratic;

struct AtInit {
    AtInit()
    {
        HttpClient::setHttpClientImplVersion(2);
    }
} atInit;

#include "http_client_test.cc"
