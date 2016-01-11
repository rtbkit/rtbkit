/** bids.cc                                 -*- C++ -*-
    Rémi Attab, 27 Feb 2013
    Copyright (c) 2013 Datacratic.  All rights reserved.

    Implementation of the representation of a bid response.

*/

#include "bids.h"

#include "jml/utils/exc_check.h"
#include "jml/utils/json_parsing.h"

using namespace std;
using namespace ML;

namespace RTBKIT {


/******************************************************************************/
/* BID STATUS                                                                 */
/******************************************************************************/

const char* bidStatusToChar(BidStatus status)
{
    switch (status) {
    case BS_WIN:        return "WIN";  break;
    case BS_LOSS:       return "LOSS";  break;
    case BS_TOOLATE:    return "TOOLATE";  break;
    case BS_INVALID:    return "INVALID";  break;
    case BS_LOSTBID:    return "LOSTBID";  break;
    case BS_DROPPEDBID: return "DROPPEDBID";  break;
    case BS_NOBUDGET:   return "NOBUDGET";  break;
    default:
        throw ML::Exception("unknown bid status");
    }
}

BidStatus bidStatusFromString(const std::string& str)
{
    switch (str[0]) {
    case 'D':
        if (str == "DROPPEDBID") return BS_DROPPEDBID;
        break;

    case 'I':
        if (str == "INVALID") return BS_INVALID;
        break;

    case 'N':
        if (str == "NOBUDGET") return BS_NOBUDGET;
        break;

    case 'L':
        if (str == "LOSS") return BS_LOSS;
        if (str == "LOSTBID") return BS_LOSTBID;
        if (str == "LATEWIN") return BS_WIN;
        break;

    case 'T':
        if (str == "TOOLATE") return BS_TOOLATE;
        break;

    case 'W':
        if (str == "WIN") return BS_WIN;
        break;
    };

    throw ML::Exception("unknown bid status");
}


/******************************************************************************/
/* BID                                                                        */
/******************************************************************************/

void
Bid::
bid(int creativeIndex, Amount price, double priority)
{
    if (this->price > price) return;
    if (this->price == price && this->priority > priority) return;

    ExcCheckGreaterEqual(creativeIndex, 0, "Invalid creative index");

    auto it = find(
            availableCreatives.begin(),
            availableCreatives.end(),
            creativeIndex);
    ExcCheck(it != availableCreatives.end(),
            "Creative index is not available for bidding: " + std::to_string(creativeIndex));

    this->creativeIndex = creativeIndex;
    this->price = price;
    this->priority = priority;
}


Json::Value
Bid::
toJson() const
{
    if (isNullBid()) return Json::Value();

    Json::Value json(Json::objectValue);

    json["creative"] = creativeIndex;
    json["price"] = price.toString();
    json["priority"] = priority;
    json["spotIndex"] = spotIndex;
    json["ext"] = ext;
    if (!account.empty()) json["account"] = account.toString();

    return json;
}

Bid
Bid::
fromJson(ML::Parse_Context& context)
{
    Bid bid;

    if (context.match_literal("null") || context.match_literal("{}"))
        return bid;  // null bid

    auto onBidField = [&] (
            const std::string& fieldName, ML::Parse_Context& context)
        {
            ExcCheck(!fieldName.empty(), "invalid empty field name");

            bool foundField = true;
            switch(fieldName[0]) {

            case 'a':
                if (fieldName == "account")
                    bid.account = AccountKey(expectJsonStringAscii(context));

                else foundField = false;
                break;

            case 'c':
                if (fieldName == "creative")
                    bid.creativeIndex = context.expect_int();

                else foundField = false;
                break;

	    case 'e':
	      if (fieldName == "ext"){
		  bid.ext = expectJson(context);
	      }
		else
		  foundField = false;
		break;

            case 'p':
                if (fieldName == "price")
                    bid.price = Amount::parse(expectJsonStringAscii(context));

                else if (fieldName == "priority")
                    bid.priority = context.expect_double();

                else foundField = false;
                break;

            case 's':
                if (fieldName == "spotIndex")
                    bid.spotIndex = context.expect_int();

                // Legacy name for priority
                else if (fieldName == "surplus")
                    bid.priority = context.expect_double();

                else foundField = false;
                break;

            default: foundField = false;
            }

            ExcCheck(foundField, "unknown bid field " + fieldName);
        };

    expectJsonObject(context, onBidField);

    return bid;
}


/******************************************************************************/
/* BIDS                                                                       */
/******************************************************************************/

Bid&
Bids::
bidForSpot(int spotIndex)
{
    for (Bid& bid : *this) {
        if (bid.spotIndex == spotIndex) return bid;
    }

    char error[32];
    snprintf(error, sizeof error, "Unknown spot index '%d'", spotIndex);

    ExcCheck(false, error);
}


Json::Value
Bids::
toJson() const
{
    Json::Value json(Json::objectValue);

    auto& bids = json["bids"];
    for (const Bid& bid : *this)
        bids.append(bid.toJson());

    if (!dataSources.empty()) {
        auto& sources = json["sources"];
        for (const string& dataSource : dataSources)
            sources.append(dataSource);
    }

    return json;
}

std::string
Bids::
toJsonStr() const{
    return toJson().toStringNoNewLine();
}

Bids
Bids::
fromJson(const std::string& raw)
{
    Bids result;

    auto onDataSourceEntry = [&] (int, ML::Parse_Context& context)
        {
            result.emplace_back(Bid::fromJson(context));
        };

    auto onBidEntry = [&] (int, ML::Parse_Context& context)
        {
            result.dataSources.insert(expectJsonStringAscii(context));
        };

    auto onBidsEntry = [&] (const string& fieldName, ML::Parse_Context& context)
        {
            ExcCheck(!fieldName.empty(), "invalid empty field name");

            bool foundField = true;
            switch (fieldName[0]) {
            case 'b':
                if (fieldName == "bids")
                    expectJsonArray(context, onDataSourceEntry);

                else foundField = false;
                break;

            case 's':
                if (fieldName == "sources")
                    expectJsonArray(context, onBidEntry);

                else foundField = false;
                break;

            default: foundField = false;
            }

            ExcCheck(foundField, "unknown bids field " + fieldName);
        };

    ML::Parse_Context context(raw, raw.c_str(), raw.c_str() + raw.length());
    expectJsonObject(context, onBidsEntry);

    return result;
}

/******************************************************************************/
/* BID RESULT                                                                 */
/******************************************************************************/

BidResult
BidResult::
parse(const std::vector<std::string>& msg)
{
    BidResult result;
    ExcCheckGreaterEqual(msg.size(), 6, "Invalid bid result message size");

    result.result = bidStatusFromString(msg[0]);
    result.timestamp = boost::lexical_cast<double>(msg[1]);

    result.confidence = msg[2];
    result.auctionId = Id(msg[3]);
    result.spotNum = boost::lexical_cast<int>(msg[4]);
    result.secondPrice = Amount::parse(msg[5]);

    // Lightweight messages stop here
    if (msg.size() <= 6) return result;
    ExcCheckGreaterEqual(msg.size(), 11, "Invalid long bid result message size");

    string bidRequestSource = msg[6];
    result.request.reset(BidRequest::parse(bidRequestSource, msg[7]));

    auto jsonParse = [] (const std::string& str)
        {
            if (str.empty()) return Json::Value();
            return Json::parse(str);
        };

    result.ourBid = Bids::fromJson(msg[8]);
    result.metadata = jsonParse(msg[9]);
    result.augmentations = jsonParse(msg[10]);

    return result;
}

} // namepsace RTBKIT

namespace Datacratic{

void
DefaultDescription<Bids>::
parseJsonTyped(Bids *val, JsonParsingContext &context) const{
  Json::Value v = context.expectJson();
  //could be optimized by defining a new function which parses JSON::Value directly
  *val = std::move(Bids::fromJson(v.toStringNoNewLine()));
}

void
DefaultDescription<Bids>::
printJsonTyped(const Bids *val, JsonPrintingContext &context) const{
  context.writeJson(val->toJson());
}

bool
DefaultDescription<Bids>::
isDefaultTyped(const Bids *val) const{
  return val->empty();
}

}

