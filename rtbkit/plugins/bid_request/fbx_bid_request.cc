/* fbx_bid_request.cc
   Jean-Sebastien Bejeau, 19 June 2013

   Bid request parser for FBX.
*/

#include "fbx_bid_request.h"
#include "fbx.h"
#include "fbx_parsing.h"
#include "jml/utils/json_parsing.h"

using namespace std;

namespace RTBKIT {


/*****************************************************************************/
/* FBX BID REQUEST PARSER                                                */
/*****************************************************************************/

BidRequest *
fromFbx(FBX::BidRequest && req,
            const std::string & provider,
            const std::string & exchange)
{
    std::unique_ptr<BidRequest> result(new BidRequest());

    result->auctionId = std::move(req.requestId);
    result->auctionType = AuctionType::SECOND_PRICE;
    result->timestamp = Date::now();
    result->provider = provider;
    result->exchange = (exchange.empty() ? provider : exchange);

    result->ipAddress = req.userContext.ipAddressMasked;
    result->userAgent = req.userContext.userAgent;
    result->location.countryCode = req.userContext.country;
    result->isTest = false;
    result->unparseable = std::move(req.unparseable);


    /* Undefined in default bid request

    string partnerMatchId;		///< Partner’s user ID
    TaggedBool allowViewTag;        ///< Indicates if view tags are accepted.
    PageTypeCode pageTypeId;        ///< Page type
    TaggedInt   numSlots;           ///< Estimated number of ad slots in the placement
    */

    return result.release();
}

namespace {

static DefaultDescription<FBX::BidRequest> desc;

} // file scope

BidRequest *
FbxBidRequestParser::
parseBidRequest(const std::string & jsonValue,
                const std::string & provider,
                const std::string & exchange)
{
    StructuredJsonParsingContext jsonContext(jsonValue);

    FBX::BidRequest req;
    desc.parseJson(&req, jsonContext);

    return fromFbx(std::move(req), provider, exchange);
}

BidRequest *
FbxBidRequestParser::
parseBidRequest(ML::Parse_Context & context,
                const std::string & provider,
                const std::string & exchange)
{
    StreamingJsonParsingContext jsonContext(context);

    FBX::BidRequest req;
    desc.parseJson(&req, jsonContext);

    return fromFbx(std::move(req), provider, exchange);
}

} // namespace RTBKIT
