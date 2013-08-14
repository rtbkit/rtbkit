#include "mock_exchange_connector.h"

namespace {
    using namespace RTBKIT;

    struct Init {
        Init() {
            ExchangeConnector::registerFactory<MockExchangeConnector>();
        }
    } init;
}
