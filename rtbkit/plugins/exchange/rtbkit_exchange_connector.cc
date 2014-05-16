/* rtbkit_exchange_connector.cc
   Mathieu Stefani, 15 May 2013
   
   Implementation of the RTBKit exchange connector.
*/

#include "rtbkit_exchange_connector.h"
#include "rtbkit/plugins/exchange/http_auction_handler.h"

using namespace Datacratic;

namespace RTBKIT {

/*****************************************************************************/
/* RTBKIT EXCHANGE CONNECTOR                                                 */
/*****************************************************************************/

RTBKitExchangeConnector::
RTBKitExchangeConnector(ServiceBase &owner, const std::string &name)
    : OpenRTBExchangeConnector(owner, name)
{
}

RTBKitExchangeConnector::
RTBKitExchangeConnector(const std::string &name,
                        std::shared_ptr<ServiceProxies> proxies)
    : OpenRTBExchangeConnector(name, proxies)
{
}

std::shared_ptr<BidRequest>
RTBKitExchangeConnector::
parseBidRequest(HttpAuctionHandler &connection,
                const HttpHeader &header,
                const std::string &payload)
{
    auto request = 
        OpenRTBExchangeConnector::parseBidRequest(connection, header, payload);


    if (request != nullptr) {
        for (const auto &imp: request->imp) {
            if (!imp.ext.isMember("allowed_ids")) {
                connection.sendErrorResponse("MISSING_EXTENSION_FIELD",
                    ML::format("The impression '%s' requires the 'allowed_ids' extension field",
                               imp.id.toString()));  
                request.reset();
                break;
            }
        }
    }

    return request;
}

} // namespace RTBKIT

namespace {

struct Init
{ 
    Init()
    {
        RTBKIT::FilterRegistry::registerFilter<RTBKIT::AllowedIdsCreativeExchangeFilter>();
        RTBKIT::ExchangeConnector::registerFactory<RTBKIT::RTBKitExchangeConnector>();
    }
} init;

}

