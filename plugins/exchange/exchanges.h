 /* exchanges.h
   Eric Robert, 9 May 2013
   Copyright (c) 2012 Datacratic Inc.  All rights reserved.

   Exchange registration
*/

#include "rtbkit/examples/mock_exchange_connector.h"
#include "rtbkit/plugins/exchange/rubicon_exchange_connector.h"
#include "rtbkit/plugins/exchange/openrtb_exchange_connector.h"
#include "rtbkit/plugins/exchange/fbx_exchange_connector.h"

namespace {
    using namespace Datacratic;
    using namespace RTBKIT;

    struct Init {
        Init() {
            ExchangeConnector::registerFactory<MockExchangeConnector>();
            ExchangeConnector::registerFactory<RubiconExchangeConnector>();
            ExchangeConnector::registerFactory<OpenRTBExchangeConnector>();
            ExchangeConnector::registerFactory<FBXExchangeConnector>();
        }
    } init;
}

