/* fees.cc                                               -*- C++ -*-
   JS Bejeau, 13 June 2015
   Copyright (c) 2012 Datacratic.  All rights reserved.

*/

#include "fees.h"

using namespace std;

namespace RTBKIT {
namespace {
    
    unordered_map<string, Fees::FeesFactory> FeesRegistery;

} // namespace anonymous

/***************************/
// FEES
/***************************/
Fees::
Fees() {
}

Fees::
~Fees() {
}

void
Fees::
RegisterFees(string name, FeesFactory factory) {
    FeesRegistery[name] = factory;

}

shared_ptr<Fees>
Fees::
createFees(Json::Value const &json) {
    auto name = json.get("type", "null").asString();
    if(FeesRegistery.find(name) == FeesRegistery.end()) {
        throw ML::Exception("Unknown type '%s'", name.c_str());
    }else {
        return FeesRegistery[name](json);
    }

    throw ML::Exception("Unknown type '%s'", name.c_str());
}

/***************************/
// NULL FEES
/***************************/

NullFees::
NullFees(Json::Value const &) {
}

NullFees::
~NullFees(){
}

shared_ptr<Fees>
NullFees::
createNullFees() {
    Json::Value json;
    shared_ptr<Fees> ret(new NullFees(json));
    return ret;
}


Amount
NullFees::
applyFees(Amount bidPrice) const {
    return bidPrice;
}

Json::Value
NullFees::
toJson() const {

    Json::Value result;
    result["type"] = "none";
    return result;
}


/***************************/
// FLAT FEES
/***************************/

FlatFees::
FlatFees(Json::Value const & json) {
    
    if (json.isMember("params") && json["params"].isObject()) {
        auto params = json["params"];
        if (params.isMember("a") && params["a"].isString()) {
            a = Amount::parse(params["a"].asString());
            if (a.isNegative()) {
                throw ML::Exception("Flat fees should have one parameter :"
                        " a (Positive or Null)");
            }
        }else {
            throw ML::Exception("Flat fees should have one parameter : a (Amount string)");
        }

    } else {
        throw ML::Exception("Flat fees should have parameters object : params");
    }
}

FlatFees::
~FlatFees(){
}

Amount
FlatFees::
applyFees(Amount bidPrice) const {
    auto bidPriceFinal = bidPrice - a;
    if (bidPriceFinal.isNonNegative()) {
        if (bidPriceFinal <= bidPrice) {
            return bidPriceFinal;
        }else {
            return bidPrice;
        }
    } else {
        return MicroUSD(0);
    }
}

Json::Value
FlatFees::
toJson() const {

    Json::Value result;
    Json::Value params;
    result["type"] = "flat";
    params["a"] = a.toString();
    result["params"] = params;
    return result;
}

/***************************/
// LINEAR FEES
/***************************/

LinearFees::
LinearFees(Json::Value const & json) {

    if (json.isMember("params") && json["params"].isObject()) {
        auto params = json["params"];
        if (params.isMember("a") && params["a"].isNumeric() &&
             params.isMember("b") && params["b"].isNumeric()) {
            a = params["a"].asDouble();
            b = params["b"].asDouble();

            // "a" and "b" validation
            // Bid Price Final must be always inferior than Original Bid Price
            // on the entire biddable range : 0-40000 $

            // "b" is the origin ordinate and Original Bid Price minimum is 0
            // so "b" must be inferior or equal to 0
            if (b > 0) {
                throw ML::Exception("Linear fees should be able to bid"
                        " on all biddable price range");
            }

            // Known "b", if we want to bid on the total range of price
            // "a", the maximum slope should give a Final Bid Price at 40000
            if (a > (1.0 - (b/40000))) {
                throw ML::Exception("Linear fees should be able to bid"
                        " on all biddable price range");
            }

        }else {
            throw ML::Exception("Linear fees should have 2 parameters : a & b (double)");
        }

    } else {
        throw ML::Exception("Linear fees should have parameters object : params");
    }
}

LinearFees::
~LinearFees(){
}

Amount
LinearFees::
applyFees(Amount bidPrice) const {

    auto bidPriceFinal = bidPrice * a + MicroUSD(b);
    if (bidPriceFinal.isNonNegative()) {
        if (bidPriceFinal <= bidPrice) {
            return bidPriceFinal;
        }else {
            return bidPrice;
        }
    } else {
        return MicroUSD(0);
    }
}

Json::Value
LinearFees::
toJson() const {

    Json::Value result;
    Json::Value params;
    result["type"] = "linear";
    params["a"] = a;
    params["b"] = b;
    result["params"] = params;
    return result;
}

/***************************/
// REGISTER FEES
/***************************/
namespace {

struct InitFees {
    InitFees(){
        RTBKIT::Fees::RegisterFees("none", NullFees::RegisterSpecificFees);
        RTBKIT::Fees::RegisterFees("flat", FlatFees::RegisterSpecificFees);
        RTBKIT::Fees::RegisterFees("linear", LinearFees::RegisterSpecificFees);
    }
}initfees;

} // anonymous namespace


} // namespace RTBKIT
