 /* mock_exchange_connector.cc
   Eric Robert, 9 April 2013
   Copyright (c) 2012 Datacratic Inc.  All rights reserved.

   Simple mock exchange connector
*/

#include "mock_exchange_connector.h"

namespace {
    using namespace RTBKIT;

    struct Init {
        Init() {
            ExchangeConnector::registerFactory<MockExchangeConnector>();
        }
    } init;
}

