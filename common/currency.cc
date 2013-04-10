/* currency.cc
   Jeremy Barnes, 5 November 2012
   Copyright (c) 2012 Datacratic Inc.  All rights reserved.

*/

#include "rtbkit/common/currency.h"

using namespace std;
using namespace ML;
using namespace Datacratic;


namespace RTBKIT {


/*****************************************************************************/
/* AMOUNT                                                                    */
/*****************************************************************************/


Json::Value
Amount::
toJson() const
{
    Json::Value result(Json::arrayValue);
    result[0] = value;
    result[1] = getCurrencyStr();
    return result;
}

Amount
Amount::
fromJson(const Json::Value & json)
{
    return Amount(json[1].asString(), json[0].asInt());
}

std::string
Amount::
getCurrencyStr(CurrencyCode currencyCode)
{
    switch (currencyCode) {
    case CurrencyCode::CC_NONE: return "NONE";
    case CurrencyCode::CC_USD:  return "USD/1M";
    default:
        throw ML::Exception("unknown currency code %d", (uint32_t)currencyCode);
    }
}

std::string
Amount::
getCurrencyStr() const
{
    return getCurrencyStr(currencyCode);
}

std::string
Amount::
toString() const
{
    if (!*this)
        return "0";
    return to_string(value) + getCurrencyStr();
}

Amount
Amount::
parse(const std::string & value)
{
    if (value == "0")
        return Amount();

    const char * p = value.c_str();
    char * ep = 0;
    errno = 0;
    long long val = strtoll(p, &ep, 10);
    if (errno != 0)
        throw ML::Exception("invalid amount parsed");
    string currency(ep);

    if (currency == "") {
        throw ML::Exception("no currency on Amount " + value);
    }

    return Amount(currency, val);
}

CurrencyCode
Amount::
parseCurrency(const std::string & currency)
{
    if (currency == "NONE")
        return CurrencyCode::CC_NONE;
    else if (currency == "USD/1M")
        return CurrencyCode::CC_USD;
    else throw ML::Exception("unknown currency code " + currency);
}

std::ostream &
operator << (std::ostream & stream, Amount amount)
{
    return stream << amount.toString();
}

void
Amount::
assertCurrencyIsCompatible(const Amount & other) const
{
    if (!currencyIsCompatible(other))
        throw ML::Exception("currencies are not compatible: "
                            + toString() + " vs " + other.toString());
}


bool
Amount::
operator < (const Amount & other) const
{
    assertCurrencyIsCompatible(other);
    return value < other.value;
}

bool
Amount::
operator <= (const Amount & other) const
{
    assertCurrencyIsCompatible(other);
    return value <= other.value;
}

bool
Amount::
operator > (const Amount & other) const
{
    assertCurrencyIsCompatible(other);
    return value > other.value;
}

bool
Amount::
operator >= (const Amount & other) const
{
    assertCurrencyIsCompatible(other);
    return value >= other.value;
}

void
Amount::
serialize(ML::DB::Store_Writer & store) const
{
    if (!*this) {
        store << (unsigned char)0;
        return;
    }
    
    store << (unsigned char)(1 + (value < 0))
          << (uint32_t)currencyCode
          << ML::DB::compact_size_t(value >= 0 ? value : -value); // need compact_int_t as it is signed
}

void
Amount::
reconstitute(ML::DB::Store_Reader & store)
{
    unsigned char version;
    store >> version;
    if (version == 0) {
        *this = Amount();
        return;
    }

    if (version > 2)
        throw ML::Exception("invalid version reconstituting Amount");

    uint32_t ccode;
    store >> ccode;
    currencyCode = (CurrencyCode)ccode;
    value = ML::DB::compact_size_t(store);
    if (version == 2)
        value = -value;

    //cerr << "reconstituted = " << *this << endl;
}


/*****************************************************************************/
/* CURRENCY POOL                                                             */
/*****************************************************************************/

bool
CurrencyPool::
operator == (const CurrencyPool & other) const
{
    auto checkContains = [] (const CurrencyPool & pool1,
                             const CurrencyPool & pool2)
        {
            for (auto amt: pool1.currencyAmounts) {
                if (pool2.getAvailable(amt.currencyCode) != amt)
                    return false;
            }

            return true;
        };

    return checkContains(*this, other) && checkContains(other, *this);
}

Amount
CurrencyPool::
getAvailable(const CurrencyCode & currency) const
{
    for (auto & am: currencyAmounts) {
        if (am.currencyCode == currency) {
            return am;
        }
    }

    return Amount(currency, 0);
}

bool
CurrencyPool::
hasAvailable(const Amount & amount) const
{
    return getAvailable(amount.currencyCode).value >= amount.value;
}

Json::Value
CurrencyPool::
toJson() const
{
    Json::Value result(Json::objectValue);
    for (auto a: currencyAmounts)
        result[a.getCurrencyStr()] = a.value;
    return result;
}

CurrencyPool
CurrencyPool::
fromJson(const Json::Value & json)
{
    CurrencyPool result;

    for (auto it = json.begin(), end = json.end();  it != end;  ++it)
        result += Amount(it.memberName(), (*it).asInt());

    //cerr << "getting currencyPool from JSON " << json
    //     << " returned " << result << endl;

    return result;
}

std::string
CurrencyPool::
toString() const
{
    if (isZero())
        return "0";

    string result;
    for (unsigned i = 0;  i < currencyAmounts.size();  ++i) {
        if (i > 0)
            result += ", ";
        result += currencyAmounts[i].toString();
    }
    return result;
}

std::ostream & operator << (std::ostream & stream, CurrencyPool pool)
{
    return stream << pool.toString();
}

void
CurrencyPool::
serialize(ML::DB::Store_Writer & store) const
{
    store << (unsigned char)0
          << currencyAmounts;
}

void
CurrencyPool::
reconstitute(ML::DB::Store_Reader & store)
{
    unsigned char version;
    store >> version;
    if (version != 0)
        throw ML::Exception("error reconstituting currency pool");
    store >> currencyAmounts;
}

bool
CurrencyPool::
isSameOrPastVersion(const CurrencyPool & otherPool)
    const
{
    for (const Amount & amount: currencyAmounts) {
        const Amount otherAmount
            = otherPool.getAvailable(amount.currencyCode);
        if (amount < otherAmount)
            return false;
    }

    return true;
}

/*****************************************************************************/
/* LINE ITEMS                                                                */
/*****************************************************************************/


LineItems
LineItems::
fromJson(const Json::Value & json)
{
    LineItems result;

    for (auto it = json.begin(), end = json.end();  it != end;  ++it)
        result[it.memberName()] = CurrencyPool::fromJson(*it);

    return result;
}

Json::Value
LineItems::
toJson() const
{
    Json::Value result(Json::objectValue);
    for (auto & e: entries)
        result[e.first] = e.second.toJson();
    return result;
}

bool
LineItems::
operator == (const LineItems & other) const
{
    auto compareLineItems = [] (const LineItems & li1,
                                const LineItems & li2)
        {
            for (auto & e: li1.entries) {
                if (e.second != li2[e.first])
                    return false;
            }
            return true;
        };

    return compareLineItems(*this, other) && compareLineItems(other, *this);
}

LineItems &
LineItems::
operator += (const LineItems & other)
{
    for (auto & e: other.entries)
        (*this)[e.first] += e.second;
    return *this;
}

void
LineItems::
serialize(ML::DB::Store_Writer & store) const
{
    store << (unsigned char)0 // version
          << entries;
}

void
LineItems::
reconstitute(ML::DB::Store_Reader & store)
{
    unsigned char version;
    store >> version;
    if (version != 0)
        throw ML::Exception("error reconstituting currency pool");
    store >> entries;
}

} // namespace RTBKIT

