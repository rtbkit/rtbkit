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


std::string
Auction::Response::
print(WinLoss wl)
{
    switch (wl) {
    case PENDING: return "PENDING";
    case WIN:     return "WIN";
    case LOSS:    return "LOSS";
    case TOOLATE: return "TOOLATE";
    case INVALID: return "INVALID";
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
    result["tagId"] = tagId;
    result["bidData"] = bidData;

    result["agent"] = agent;
    result["account"] = account.toJson();
    result["meta"] = meta;

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
        || tagId == -1)
        return false;
    return true;
}

void
Auction::Response::
serialize(DB::Store_Writer & store) const
{
    int version = 5;
    store << version << price.maxPrice << price.priority
          << tagId << account
          << test << agent << bidData << meta << creativeId
          << creativeName << (int)localStatus << visitChannels;
}

void
Auction::Response::
reconstitute(DB::Store_Reader & store)
{
    int version, localStatusi;
    store >> version;
    if (version == 1) {
        string campaign, strategy;
        string tag, click_url;
        store >> price.maxPrice >> price.priority
              >> tag >> click_url >> tagId >> strategy >> campaign
              >> test >> agent >> bidData >> meta >> creativeId
              >> creativeName >> localStatusi;
        account = { campaign, strategy };
    }
    else if (version == 2) {
        string campaign, strategy;
        int maxPriceUSDMicrosCPM;
        store >> maxPriceUSDMicrosCPM >> price.priority
              >> tagId >> strategy >> campaign
              >> test >> agent >> bidData >> meta >> creativeId
              >> creativeName >> localStatusi;
        price.maxPrice = MicroUSD_CPM(maxPriceUSDMicrosCPM);
        account = { campaign, strategy };
    }
    else if (version == 3) {
        string campaign, strategy;
        store >> price.maxPrice >> price.priority
              >> tagId >> strategy >> campaign
              >> test >> agent >> bidData >> meta >> creativeId
              >> creativeName >> localStatusi;
        account = { campaign, strategy };
    }
    else if (version == 4) {
        store >> price.maxPrice >> price.priority
              >> tagId >> account
              >> test >> agent >> bidData >> meta >> creativeId
              >> creativeName >> localStatusi;
    }
    else if (version == 5) {
        store >> price.maxPrice >> price.priority
              >> tagId >> account
              >> test >> agent >> bidData >> meta >> creativeId
              >> creativeName >> localStatusi >> visitChannels;
    }
    else throw ML::Exception("reconstituting wrong version");
    localStatus = (WinLoss)localStatusi;
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
        || newResponse.tagId == -1)
        return INVALID;

    if (current->tooLate)
        return TOOLATE;


    WinLoss result;

    auto_ptr<Data> newData(new Data());

    for (;;) {
        if (current->tooLate)
            return TOOLATE;
        
        if (current->hasValidResponse(spotNum)
            && newResponse.price.priority
                <= current->winningResponse(spotNum).price.priority)
            return LOSS;
        
        *newData = *current;

        bool hasExisting = current->hasValidResponse(spotNum);

        result = newResponse.localStatus = PENDING;
        newData->responses[spotNum].push_back(newResponse);

        if (hasExisting) {
            newData->responses[spotNum][0].localStatus = LOSS;
            std::swap(newData->responses[spotNum].front(),
                      newData->responses[spotNum].back());
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
                newData->responses[spotNum][0].localStatus = WIN;
            
            for (unsigned i = 1;  i < newData->responses[spotNum].size();  ++i)
                newData->responses[spotNum][i].localStatus = LOSS;
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
            for (unsigned i = 0;  i < newData->responses[spotNum].size();  ++i) {
                newData->responses[spotNum][i].localStatus = LOSS;
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

    string result = ML::format("Auction: %d spots", (int)numSpots());
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

