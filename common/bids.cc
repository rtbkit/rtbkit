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
/* BIDS                                                                       */
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
            "Creative index is not available for bidding: " + creativeIndex);

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

    return json;
}

Json::Value
Bids::
toJson() const
{
    Json::Value json(Json::arrayValue);

    for (const Bid& bid : *this)
        json.append(bid.toJson());

    return json;
}

Bids
Bids::
fromJson(const std::string& raw)
{
    Bids result;

    ML::Parse_Context context(raw, raw.c_str(), raw.c_str() + raw.length());

    auto onBidEntry = [&] (int, ML::Parse_Context& context)
        {
            if (context.match_literal("null") || context.match_literal("{}")) {
                result.emplace_back();
                return;  // null bid
            }

            Bid bid;

            auto onBidField = [&] (
                    const std::string& fieldName, ML::Parse_Context& context)
            {
                ExcCheck(!fieldName.empty(), "invalid empty field name");

                if (fieldName[0] == 'c' && fieldName == "creative")
                    bid.creativeIndex = context.expect_int();

                else if (fieldName[0] == 'p' && fieldName == "price")
                    bid.price = Amount::parse(expectJsonStringAscii(context));

                else if ((fieldName[0] == 'p' && fieldName == "priority")
                        || (fieldName[0] == 's' && fieldName == "surplus"))
                {
                    bid.priority = context.expect_double();
                }

                else throw ML::Exception("unknown bid field " + fieldName);
            };

            expectJsonObject(context, onBidField);
            result.push_back(bid);
        };

    expectJsonArray(context, onBidEntry);
    return result;
}


} // namepsace RTBKIT
