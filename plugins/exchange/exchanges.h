 /* exchanges.h
   Eric Robert, 9 May 2013
   Copyright (c) 2012 Datacratic Inc.  All rights reserved.

   Exchange registration
*/

#include "rtbkit/examples/mock_exchange_connector.h"
#include "rtbkit/plugins/exchange/rubicon_exchange_connector.h"
#include "rtbkit/plugins/exchange/openrtb_exchange_connector.h"

namespace {
    using namespace Datacratic;
    using namespace RTBKIT;

    struct Init {
        static ExchangeConnector * createMockExchange(ServiceBase * owner,
                                                      std::string const & name) {
            return new MockExchangeConnector(*owner, name);
        }

        static ExchangeConnector * createRubiconExchange(ServiceBase * owner,
                                                         std::string const & name) {
            return new RubiconExchangeConnector(*owner, name);
        }

        static ExchangeConnector * createOpenRTBExchange(ServiceBase * owner,
                                                         std::string const & name) {
            return new OpenRTBExchangeConnector(*owner, name);
        }

        Init() {
            ExchangeConnector::registerFactory("mock", createMockExchange);
            ExchangeConnector::registerFactory("rubicon", createRubiconExchange);
            ExchangeConnector::registerFactory("openrtb", createOpenRTBExchange);
        }
    } init;
}

