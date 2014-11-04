/* auction.cc
   Jeremy Barnes, 6 April 2011
   Copyright (c) 2011 Datacratic.  All rights reserved.

   Auction implementation.
*/

#include "rtbkit/common/auction.h"
#include "jml/arch/exception.h"
#include "jml/arch/format.h"
#include "jml/arch/backtrace.h"
#include "jml/utils/parse_context.h"
#include "jml/utils/less.h"
#include "jml/utils/string_functions.h"
#include "jml/utils/json_parsing.h"
#include "jml/db/persistent.h"

#include "ace/Acceptor.h"
#include <ace/Timer_Heap_T.h>
#include <ace/Synch.h>
#include <ace/Timer_Queue_Adapters.h>
#include "ace/SOCK_Acceptor.h"
#include <ace/High_Res_Timer.h>
#include <ace/Dev_Poll_Reactor.h>
#include <set>

using namespace std;
using namespace ML;

namespace RTBKIT {

/*****************************************************************************/
/* AUCTION                                                                   */
/*****************************************************************************/

Json::Value
Auction::Price::
toJson() const
{
    Json::Value result;
    result["maxPrice"] = maxPrice.toJson();
    result["priority"] = priority;
    return result;
}

std::string
Auction::Price::
toJsonStr() const
{
    return format("{\"maxPrice\":%d,\"priority\":%f}", maxPrice, priority);
}

Auction::Price
Auction::Price::
fromJson(const Json::Value& json)
{
    Price result;

    result.maxPrice = Amount::fromJson(json["maxPrice"]);
    if (json.isMember("priority"))
        result.priority = json["priority"].asDouble();

    return result;
}

void
Auction::Price::
createDescription(AuctionPriceDescription & d) {
    d.addField("maxPrice", &Price::maxPrice, "");
    d.addField("priority", &Price::priority, "");
}

std::string
Auction::Response::
print(WinLoss wl)
{
    switch (wl.val) {
    case WinLoss::PENDING: return "PENDING";
    case WinLoss::WIN:     return "WIN";
    case WinLoss::LOSS:    return "LOSS";
    case WinLoss::TOOLATE: return "TOOLATE";
    case WinLoss::INVALID: return "INVALID";
    default:
        throw Exception("invalid WinLoss value");
    }
}

Json::Value
Auction::Response::
toJson() const
{
    Json::Value result;

    result["price"] = price.toJson();
    result["test"] = test;
    result["bidData"] = bidData.toJson();

    result["agent"] = agent;
    result["account"] = account.toJson();
    result["meta"] = meta;
    result["wcm"] = wcm.toJson();

    result["creativeId"] = creativeId;
    result["creativeName"] = creativeName;

    result["localStatus"] = print(localStatus);

    return result;
}

bool
Auction::Response::
valid() const
{
    if (price.maxPrice.value <= 0
        || account.empty()
        || agent == ""
        || creativeId == -1)
        return false;
    return true;
}

void
Auction::Response::
serialize(DB::Store_Writer & store) const
{
    int version = 7;

    store << version << price.maxPrice << price.priority << account
          << test << agent << bidData.toJsonStr() << meta << creativeId
          << creativeName << localStatus.val << visitChannels << wcm;
}

void
Auction::Response::
reconstitute(DB::Store_Reader & store)
{
    int version, localStatusi;
    string bidDataStr;
    store >> version;
    if (version == 1) {
        string campaign, strategy;
        string tag, click_url, tagId;
        store >> price.maxPrice >> price.priority
              >> tag >> click_url >> tagId >> strategy >> campaign
              >> test >> agent >> bidDataStr >> meta >> creativeId
              >> creativeName >> localStatusi;
        account = { campaign, strategy };
    }
    else if (version == 2) {
        string campaign, strategy;
        int maxPriceUSDMicrosCPM, tagId;
        store >> maxPriceUSDMicrosCPM >> price.priority
              >> tagId >> strategy >> campaign
              >> test >> agent >> bidDataStr >> meta >> creativeId
              >> creativeName >> localStatusi;
        price.maxPrice = MicroUSD_CPM(maxPriceUSDMicrosCPM);
        account = { campaign, strategy };
    }
    else if (version == 3) {
        string campaign, strategy;
        int tagId;
        store >> price.maxPrice >> price.priority
              >> tagId >> strategy >> campaign
              >> test >> agent >> bidDataStr >> meta >> creativeId
              >> creativeName >> localStatusi;
        account = { campaign, strategy };
    }
    else if (version == 4) {
        int tagId;
        store >> price.maxPrice >> price.priority
              >> tagId >> account
              >> test >> agent >> bidDataStr >> meta >> creativeId
              >> creativeName >> localStatusi;
    }
    else if (version == 5) {
        int tagId;
        store >> price.maxPrice >> price.priority
              >> tagId >> account
              >> test >> agent >> bidDataStr >> meta >> creativeId
              >> creativeName >> localStatusi >> visitChannels;
    }
    else if (version == 6) {
        int tagId;
        store >> price.maxPrice >> price.priority
              >> tagId >> account
              >> test >> agent >> bidDataStr >> meta >> creativeId
              >> creativeName >> localStatusi >> visitChannels >> wcm;
    }
    else if (version == 7) {
        store >> price.maxPrice >> price.priority >> account
              >> test >> agent >> bidDataStr >> meta >> creativeId
              >> creativeName >> localStatusi >> visitChannels >> wcm;
    }
    else throw ML::Exception("reconstituting wrong version");

    //convert string to bids object
    bidData = Bids::fromJson(bidDataStr);

    localStatus = (WinLoss)localStatusi;
}

void
Auction::Response::
createDescription(AuctionResponseDescription & d) {
    d.addField("price", &Response::price, "");
    d.addField("account", &Response::account, "");
    d.addField("test", &Response::test, "");
    d.addField("agent", &Response::agent, "");
    d.addField("bidData", &Response::bidData, "");
    d.addField("meta", &Response::meta, "");
    d.addField("creativeId", &Response::creativeId, "");
    d.addField("creativeName", &Response::creativeName, "");
    d.addField("localStatus", &Response::localStatus, "");
    d.addField("visitChannels", &Response::visitChannels, "");
    d.addField("wcm", &Response::wcm, "");
}

Auction::
Auction()
    : isZombie(false), exchangeConnector(nullptr), data(new Data())
{
}

Auction::
Auction(ExchangeConnector * exchangeConnector,
        HandleAuction handleAuction,
        std::shared_ptr<BidRequest> request,
        const std::string & requestStr,
        const std::string & requestStrFormat,
        Date start,
        Date expiry)
    : isZombie(false), start(start), expiry(expiry),
      request(request),
      requestStr(requestStr),
      requestStrFormat(requestStrFormat),
      exchangeConnector(exchangeConnector),
      handleAuction(handleAuction),
      data(new Data(numSpots()))
{
    ML::atomic_add(created, 1);

    this->id = request->auctionId;
    this->requestSerialized = request->serializeToString();
}

Auction::
~Auction()
{
    // Clean up the chain of data pointers
    Data * d = data;
    while (d) {
        Data * d2 = d->oldData;
        delete d;
        d = d2;
    }

    ML::atomic_add(destroyed, 1);
}

long long Auction::created = 0;
long long Auction::destroyed = 0;

double
Auction::
timeAvailable(Date now) const
{
    return now.secondsUntil(expiry);
}

double
Auction::
timeUsed(Date now) const
{
    return start.secondsUntil(now);
}

Auction::WinLoss
Auction::
setResponse(int spotNum, Response newResponse)
{
    Data * current = this->data;

    if (spotNum < 0 || spotNum >= current->responses.size())
        throw ML::Exception("invalid spot number in response");

    if (newResponse.price.maxPrice.isNegative()
        || newResponse.agent == ""
        || newResponse.creativeId == -1)
        return WinLoss::INVALID;

    if (current->tooLate)
        return WinLoss::TOOLATE;


    WinLoss result;

    auto_ptr<Data> newData(new Data());

    for (;;) {
        if (current->tooLate)
            return WinLoss::TOOLATE;
        
        *newData = *current;

        bool hasExisting = current->hasValidResponse(spotNum);

        result = newResponse.localStatus = WinLoss::PENDING;
        newData->responses[spotNum].push_back(newResponse);

        if (hasExisting) {
            auto & spot = newData->responses[spotNum];
            
            // Filter on priority first.
            if (newResponse.price.priority >
                current->winningResponse(spotNum).price.priority) {
                std::swap(spot.front(), spot.back());
            }
            // If not filter on price
            else if(newResponse.price.priority ==
                    current->winningResponse(spotNum).price.priority &&
                    newResponse.price.maxPrice >
                    current->winningResponse(spotNum).price.maxPrice) {
                 std::swap(spot.front(), spot.back());
            }
            else {
                // Do nothing, whichever bid came first wins.
            }

            spot.back().localStatus = WinLoss::LOSS;
        }

        newData->oldData = current;

        if (!ML::cmp_xchg(this->data, current, newData.get())) continue;
        newData.release();
        return result;
    }
}

const std::vector<std::vector<Auction::Response> > & 
Auction::
getResponses() const
{
    const Data * current = this->data;
    return current->responses;
}

void
Auction::
addDataSources(const std::set<std::string> & sources)
{
    if (sources.empty()) return;

    Data * current = this->data;
    unique_ptr<Data> newData(new Data);

    for (;;) {

        std::set<std::string> newSources = current->dataSources;
        newSources.insert(sources.begin(), sources.end());

        // Nothing new was added, just bail.
        if (newSources.size() == current->dataSources.size()) return;

        if (!newData) newData.reset(new Data);
        *newData = *current;
        std::swap(newData->dataSources, newSources);

        if (!ML::cmp_xchg(this->data, current, newData.get())) continue;
        newData.release();
        return;
    }
}

const std::set<std::string> &
Auction::
getDataSources() const
{
    const Data * current = this->data;
    return current->dataSources;
}

bool
Auction::
finish()
{
    Data * current = this->data;

    for (;;) {
        if (current->tooLate)
            return false;

        auto_ptr<Data> newData(new Data(*current));

        for (unsigned spotNum = 0;  spotNum < numSpots(); ++spotNum) {
            if (newData->hasValidResponse(spotNum))
                newData->responses[spotNum][0].localStatus = WinLoss::WIN;
        }
        
        newData->oldData = current;
        newData->tooLate = true;

        if (!ML::cmp_xchg(this->data, current, newData.get())) continue;
        newData.release();
        break;
    }

    handleAuction(shared_from_this());

    return true;
}

bool
Auction::
setError(const std::string & error, const std::string & details)
{
    Data * current = this->data;

    for (;;) {
        if (current->tooLate)
            return false;

        auto_ptr<Data> newData(new Data(*current));
        
        newData->error = error;
        newData->details = details;

        for (unsigned spotNum = 0;  spotNum < numSpots();  ++spotNum) {
            if (newData->hasValidResponse(spotNum)) {
                newData->responses[spotNum][0].localStatus = WinLoss::LOSS;
            }
        }
        newData->oldData = current;
        newData->tooLate = true;

        if (!ML::cmp_xchg(this->data, current, newData.get())) continue;
        newData.release();
        break;
    }

    handleAuction(shared_from_this());
    
    return true;
}

bool
Auction::
tooLate()
{
    return data->tooLate;
}

std::string
Auction::
status() const
{
    Data * current = this->data;

    string result = ML::format("Auction: %d imp", (int)numSpots());
    if (current->tooLate) result += " tooLate";
    if (!current->error.empty()) result += " error: " + current->error;

    result += " [";
    for (int spotNum = 0;  spotNum < numSpots();  ++spotNum) {
        if (spotNum != 0) result += "; ";
        if (current->hasValidResponse(spotNum))
            result += " winner: "
                + current->winningResponse(spotNum).toJson().toString();
    }
    result += "]";

    return result;
}

Json::Value
Auction::
getResponseJson(int spotNum) const
{
    Json::Value result;

    Data * current = this->data;

    if (!current->error.empty()) {
        result["error"] = current->error;
        result["details"] = current->details;
        return result;
    }
    
    for (unsigned spotNum = 0;  spotNum < numSpots();  ++spotNum) {
        if (current->hasValidResponse(spotNum))
            result[spotNum] = current->winningResponse(spotNum).toJson();
    }
    return result;
}

const Auction::Price Auction::NONE;

} // namespace RTBKIT

