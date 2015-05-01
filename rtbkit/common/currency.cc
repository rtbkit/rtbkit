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
/* CURRENCY CODE                                                             */
/*****************************************************************************/

std::string toString(CurrencyCode code)
{
    if (code == CurrencyCode::CC_NONE)
        return "NONE";
    if (code == CurrencyCode::CC_USD)
        return "USD";
    if (code == CurrencyCode::CC_EUR)
        return "EUR";
    if (code == CurrencyCode::CC_IMP)
        return "IMP";
    if (code == CurrencyCode::CC_CLK)
        return "CLK";
    throw ML::Exception("unknown currency code");
}

CurrencyCode parseCurrencyCode(const std::string & code)
{
    if (code == "NONE")
        return CurrencyCode::CC_NONE;
    if (code == "USD")
        return CurrencyCode::CC_USD;
    if (code == "EUR")
        return CurrencyCode::CC_EUR;
    if (code == "IMP")
        return CurrencyCode::CC_IMP;
    if (code == "CLK")
        return CurrencyCode::CC_CLK;
    throw ML::Exception("unknown currency code");
}

CurrencyCode jsonDecode(const Json::Value & j, CurrencyCode *)
{
    if (j.isNull())
        return CurrencyCode::CC_NONE;
    string s = j.asString();
    if (s.size() > 4)
        throw ML::Exception("unknown currency code " + s);
    return parseCurrencyCode(s);
}

Json::Value jsonEncode(CurrencyCode code)
{
    return Json::Value(toString(code));
}

/*****************************************************************************/
/* AMOUNT                                                                    */
/*****************************************************************************/


Json::Value
Amount::
toJson() const
{
    if (isZero())
        return 0;

    Json::Value result(Json::arrayValue);
    result[0] = value;
    result[1] = getCurrencyStr();

    //cerr << "Amount toJson() gave " << result.toString() << endl;

    return result;
}

std::string
Amount::
getCurrencyStr(CurrencyCode currencyCode)
{
    switch (currencyCode) {
    case CurrencyCode::CC_NONE:
        return "NONE";
    case CurrencyCode::CC_EUR:
        return "EUR/1M";
    case CurrencyCode::CC_USD:
        return "USD/1M";
    case CurrencyCode::CC_IMP:
        return "IMP";
    case CurrencyCode::CC_CLK:
        return "CLK";
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
fromJson(const Json::Value & val)
{
    //cerr << "Amount fromJson " << val.toString() << endl;

    if (val.isArray()) {
        if (val.size() != 2)
            throw ML::Exception("invalid amount size");
        return Amount(val[1].asString(), val[0].asInt());
    }
    else if (val.isString())
        return parse(val.asString());
    else if (val.isNumeric() && val.asDouble() == 0.0)
        return Amount();
    else if (val.isNull())
        return Amount();
    else if (val.isObject()) {
        string currencyCode = "NONE";
        int64_t value = 0;
        for (auto it = val.begin(), end = val.end();  it != end;  ++it) {
            if (it.memberName() == "value")
                value = it->asInt();
            else if (it.memberName() == "currencyCode")
                currencyCode = it->asString();
            else throw ML::Exception("unknown Amount field " + it.memberName());
        }
        return Amount(currencyCode, value);
    }
    else throw ML::Exception("unknown amount " + val.toString());
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
    if (currency == "EUR/1M")
        return CurrencyCode::CC_EUR;
    if (currency == "USD/1M")
        return CurrencyCode::CC_USD;
    if (currency == "IMP")
        return CurrencyCode::CC_IMP;
    if (currency == "CLK")
        return CurrencyCode::CC_CLK;
    throw ML::Exception("unknown currency code " + currency);
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


/*****************************************************************************/
/* VALUE DESCRIPTIONS                                                        */
/*****************************************************************************/

struct LineItemsDescription
    : public Datacratic::ValueDescriptionT<LineItems> {

    LineItemsDescription()
    {
    }

    virtual void parseJsonTyped(LineItems * val,
                                JsonParsingContext & context) const
    {
        *val = std::move(LineItems::fromJson(context.expectJson()));
    }

    virtual void printJsonTyped(const LineItems * val,
                                JsonPrintingContext & context) const
    {
        context.writeJson(val->toJson());
    }

    virtual bool isDefaultTyped(const LineItems * val) const
    {
        return val->empty();
    }
};

struct AmountDescription
    : public Datacratic::ValueDescriptionT<Amount> {

    AmountDescription()
    {
    }

    virtual void parseJsonTyped(Amount * val,
                                JsonParsingContext & context) const
    {
        *val = std::move(Amount::fromJson(context.expectJson()));
        //cerr << "parsing amount JSON" << *val << endl;
    }

    virtual void printJsonTyped(const Amount * val,
                                JsonPrintingContext & context) const
    {
        context.writeJson(val->toJson());
    }

    virtual bool isDefaultTyped(const Amount * val) const
    {
        return val->isZero();
    }
};

struct CurrencyPoolDescription
    : public Datacratic::ValueDescriptionT<CurrencyPool> {

    CurrencyPoolDescription()
    {
    }

    virtual void parseJsonTyped(CurrencyPool * val,
                                JsonParsingContext & context) const
    {
        *val = std::move(CurrencyPool::fromJson(context.expectJson()));
    }

    virtual void printJsonTyped(const CurrencyPool * val,
                                JsonPrintingContext & context) const
    {
        context.writeJson(val->toJson());
    }

    virtual bool isDefaultTyped(const CurrencyPool * val) const
    {
        return val->empty();
    }
};

struct CurrencyCodeDescription
    : public Datacratic::EnumDescription<CurrencyCode> {

    CurrencyCodeDescription()
    {
    }

    virtual void parseJsonTyped(CurrencyCode * val,
                                JsonParsingContext & context) const
    {
        std::string s = context.expectStringAscii();
        *val = parseCurrencyCode(s);
    }

    virtual void printJsonTyped(const CurrencyCode * val,
                                JsonPrintingContext & context) const
    {
        context.writeString(toString(*val));
    }

    virtual bool isDefaultTyped(const CurrencyCode * val) const
    {
        return false;
    }
    
};

ValueDescriptionT<LineItems> * getDefaultDescription(LineItems *)
{
    return new LineItemsDescription();
}

ValueDescriptionT<CurrencyPool> * getDefaultDescription(CurrencyPool *)
{
    return new CurrencyPoolDescription();
}

ValueDescriptionT<Amount> * getDefaultDescription(Amount *)
{
    return new AmountDescription();
}

ValueDescriptionT<CurrencyCode> * getDefaultDescription(CurrencyCode *)
{
    return new CurrencyCodeDescription();
}

} // namespace RTBKIT

