/* rubicon_exchange_connector.cc
   Jeremy Barnes, 15 March 2013
   
   Implementation of the Rubicon exchange connector.
*/

#include "rubicon_exchange_connector.h"
#include "rtbkit/core/agent_configuration/agent_config.h"
#include "rtbkit/openrtb/openrtb_parsing.h"
#include "soa/types/json_printing.h"
#include <boost/any.hpp>
#include <boost/lexical_cast.hpp>
#include "jml/utils/file_functions.h"

#include "crypto++/blowfish.h"
#include "crypto++/modes.h"
#include "crypto++/filters.h"

using namespace std;
using namespace Datacratic;

namespace RTBKIT {

/*****************************************************************************/
/* RUBICON EXCHANGE CONNECTOR                                                */
/*****************************************************************************/

RubiconExchangeConnector::
RubiconExchangeConnector(ServiceBase & owner, const std::string & name)
    : OpenRTBExchangeConnector(owner, name)
    , configuration_("rubicon")
{
    this->auctionResource = "/auctions";
    this->auctionVerb = "POST";

    init();
}

RubiconExchangeConnector::
RubiconExchangeConnector(const std::string & name,
                         std::shared_ptr<ServiceProxies> proxies)
    : OpenRTBExchangeConnector(name, proxies)
    , configuration_("rubicon")
{
    this->auctionResource = "/auctions";
    this->auctionVerb = "POST";

    init();
}

void
RubiconExchangeConnector::init()
{

    // 1.  Must have rubicon.attr containing creative attributes.  These
    //     turn into RubiconCreativeAttribute filters.
    configuration_.addField(
        "attr",
        [](const Json::Value & value, CreativeInfo & data) {
            Datacratic::jsonDecode(value, data.attr);
            return true;
    });
    // TODO: create filter from these...

    // 2.  Must have rubicon.adm that includes Rubicon's macro
    configuration_.addField(
        "adm",
        [](const Json::Value & value, CreativeInfo & data) {
            Datacratic::jsonDecode(value, data.adm);
            if (data.adm.find("${AUCTION_PRICE:BF}") == std::string::npos) {
                throw std::invalid_argument("${AUCTION_PRICE:BF} is expected in adm");
            }
            return true;
    }).snippet();

    // 3.  Must have creative ID in rubicon.crid
    configuration_.addField(
        "crid",
        [](const Json::Value & value, CreativeInfo & data) {
            Datacratic::jsonDecode(value, data.crid);
            return true;
    });

    // 4.  Must have advertiser names array in rubicon.adomain
    configuration_.addField(
        "adomain",
        [](const Json::Value & value, CreativeInfo & data) {
            Datacratic::jsonDecode(value, data.adomain);
            return true;
    });

    // 5. Optional campaign id
    configuration_.addField(
        "cid",
        [](const Json::Value & value, CreativeInfo & data) {
            Datacratic::jsonDecode(value, data.cid);
            return true;
    }).optional();

}

ExchangeConnector::ExchangeCompatibility
RubiconExchangeConnector::
getCampaignCompatibility(const AgentConfig & config,
                         bool includeReasons) const
{
    ExchangeCompatibility result;
    result.setCompatible();

    auto cpinfo = std::make_shared<CampaignInfo>();

    const Json::Value & pconf = config.providerConfig["rubicon"];

    try {
        cpinfo->seat = Id(pconf["seat"].asString());
        if (!cpinfo->seat)
            result.setIncompatible("providerConfig.rubicon.seat is null",
                                   includeReasons);
    } catch (const std::exception & exc) {
        result.setIncompatible
            (string("providerConfig.rubicon.seat parsing error: ")
             + exc.what(), includeReasons);
        return result;
    }

    result.info = cpinfo;

    return result;
}

ExchangeConnector::ExchangeCompatibility
RubiconExchangeConnector::
getCreativeCompatibility(const Creative & creative,
                         bool includeReasons) const
{
    return configuration_.handleCreativeCompatibility(creative, includeReasons);
}

float
RubiconExchangeConnector::
decodeWinPrice(const std::string & sharedSecret,
               const std::string & winPriceStr)
{
    ExcAssertEqual(winPriceStr.length(), 16);
        
    auto tox = [] (char c)
        {
            if (c >= '0' && c <= '9')
                return c - '0';
            else if (c >= 'A' && c <= 'F')
                return 10 + c - 'A';
            else if (c >= 'a' && c <= 'f')
                return 10 + c - 'a';
            throw ML::Exception("invalid hex digit");
        };

    unsigned char input[8];
    for (unsigned i = 0;  i < 8;  ++i)
        input[i]
            = tox(winPriceStr[i * 2]) * 16
            + tox(winPriceStr[i * 2 + 1]);
        
    CryptoPP::ECB_Mode<CryptoPP::Blowfish>::Decryption d;
    d.SetKey((byte *)sharedSecret.c_str(), sharedSecret.size());
    CryptoPP::StreamTransformationFilter
        filt(d, nullptr,
             CryptoPP::StreamTransformationFilter::NO_PADDING);
    filt.Put(input, 8);
    filt.MessageEnd();
    char recovered[9];
    size_t nrecovered = filt.Get((byte *)recovered, 8);

    ExcAssertEqual(nrecovered, 8);
    recovered[nrecovered] = 0;

    float res = boost::lexical_cast<float>(recovered);

    return res;
}

void
RubiconExchangeConnector::
setSeatBid(Auction const & auction,
           int spotNum,
           OpenRTB::BidResponse & response) const
{
    const Auction::Data * current = auction.getCurrentData();
    
    // Get the winning bid
    auto & resp = current->winningResponse(spotNum);

    // Find how the agent is configured.  We need to copy some of the
    // fields into the bid.
    const AgentConfig * config =
        std::static_pointer_cast<const AgentConfig>(resp.agentConfig).get();

    std::string en = exchangeName();

    // Get the exchange specific data for this campaign
    auto cpinfo = config->getProviderData<CampaignInfo>(en);

    // Put in the fixed parts from the creative
    int creativeIndex = resp.agentCreativeIndex;

    auto & creative = config->creatives.at(creativeIndex);

    // Get the exchange specific data for this creative
    auto crinfo = creative.getProviderData<CreativeInfo>(en);

    // Find the index in the seats array
    int seatIndex = 0;
    while(response.seatbid.size() != seatIndex) {
        if(response.seatbid[seatIndex].seat == cpinfo->seat) break;
        ++seatIndex;
    }

    // Create if required
    if(seatIndex == response.seatbid.size()) {
        response.seatbid.emplace_back();
        response.seatbid.back().seat = cpinfo->seat;
    }

    // Get the seatBid object
    OpenRTB::SeatBid & seatBid = response.seatbid.at(seatIndex);
    
    // Add a new bid to the array
    seatBid.bid.emplace_back();
    auto & b = seatBid.bid.back();

    // Put in the variable parts
    if (crinfo->cid.notNull()) {
        // either we use the configured campaign id...
        b.cid = crinfo->cid;
    } else {
        // ...or we use the agent name
        b.cid = Id(resp.agent);
    }

    b.id = Id(auction.id, auction.request->imp[0].id);
    b.impid = auction.request->imp[spotNum].id;
    b.price.val = getAmountIn<CPM>(resp.price.maxPrice);

    RubiconCreativeConfiguration::Context ctx = {
        creative,
        resp,
        *auction.request,
        spotNum
    };

    b.adm = configuration_.expand(crinfo->adm, ctx);
    b.adomain = crinfo->adomain;
    b.crid = crinfo->crid;
}

} // namespace RTBKIT

namespace {
    using namespace RTBKIT;

    struct AtInit {
        AtInit() {
            ExchangeConnector::registerFactory<RubiconExchangeConnector>();
        }
    } atInit;
}

