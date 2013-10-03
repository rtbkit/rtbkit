/* currency.h                                                     -*- C++ -*-
   Jeremy Barnes, 10 October 2012
   Copyright (c) 2012 Datacratic Inc.  All rights reserved.

*/

#ifndef __types__currency_h__
#define __types__currency_h__


#include "jml/utils/exc_assert.h"
#include "jml/utils/compact_vector.h"
#include "jml/utils/unnamed_bool.h"
#include "jml/db/persistent.h"
#include "soa/jsoncpp/json.h"
#include "soa/types/value_description.h"

namespace RTBKIT {


enum class CurrencyCode : std::uint32_t {
    CC_NONE = 'N' << 24 | 'O' << 16 | 'N' << 8 | 'E',
    CC_USD  = 'U' << 24 | 'S' << 16 | 'D' << 8,       // micro dollars
    CC_IMP  = 'I' << 24 | 'M' << 16 | 'P' << 8
};

std::string toString(CurrencyCode code);
CurrencyCode parseCurrencyCode(const std::string & code);

CurrencyCode jsonDecode(const Json::Value & j, CurrencyCode * = 0);
Json::Value jsonEncode(CurrencyCode code);


/*****************************************************************************/
/* AMOUNT                                                                    */
/*****************************************************************************/

struct Amount {
    Amount(CurrencyCode currencyCode = CurrencyCode::CC_NONE, int64_t value = 0)
        : value(value), currencyCode(currencyCode)
    {
        if (currencyCode == CurrencyCode::CC_NONE)
            ExcAssertEqual(value, 0);
    }

    Amount(const std::string & currencyStr, int64_t value = 0)
        : value(value), currencyCode(parseCurrency(currencyStr))
    {
        if (currencyCode == CurrencyCode::CC_NONE)
            ExcAssertEqual(value, 0);
    }

    bool isZero() const { return value == 0; }
    bool isNonNegative() const { return value >= 0; }
    bool isNegative() const { return value < 0; }

    /** Returns the minimum of the two amounts. */
    Amount limit(const Amount & other) const
    {
        assertCurrencyIsCompatible(other);
        return Amount(currencyCode, std::min(value, other.value));
    }

    JML_IMPLEMENT_OPERATOR_BOOL(!isZero());

    Amount & operator += (const Amount & other)
    {
        if (!other)
            return *this;
        else if (!*this)
            *this = other;
        else {
            ExcAssertEqual(currencyCode, other.currencyCode);
            value += other.value;
        }
        return *this;
    }
    
    Amount operator + (const Amount & other) const
    {
        Amount result = *this;
        result += other;
        return result;
    }

    Amount & operator -= (const Amount & other)
    {
        if (!other)
            return *this;
        else if (!*this)
            *this = -other;
        else {
            if (currencyCode != other.currencyCode) {
                using namespace std;
                cerr << this->toString() << " - " << other.toString() << endl;
            }
            ExcAssertEqual(currencyCode, other.currencyCode);
            value -= other.value;
        }
        return *this;
    }
    
    Amount operator - (const Amount & other) const
    {
        Amount result = *this;
        result -= other;
        return result;
    }

    Amount operator * (double factor) const
    {
        double result = static_cast<double>(value) * factor;
        return Amount(currencyCode, static_cast<int64_t>(result));
    }

    bool currencyIsCompatible(const Amount & other) const
    {
        if (currencyCode == other.currencyCode) return true;
        if (currencyCode == CurrencyCode::CC_NONE && value == 0) return true;
        if (other.currencyCode == CurrencyCode::CC_NONE && other.value == 0) return true;
        return false;
    }

    void assertCurrencyIsCompatible(const Amount & other) const;

    /** They compare equal if:
        - Both currency code and amount are equal, OR
        - Both is zero (no currency, no amount), OR
        - One is zero with a currency code and one is zero with no code
    */
    bool operator == (const Amount & other) const
    {
        return value == other.value
            && (currencyCode == other.currencyCode
                || (value == 0
                    && (currencyCode == CurrencyCode::CC_NONE
                        || other.currencyCode == CurrencyCode::CC_NONE)));
    }

    bool operator != (const Amount & other) const
    {
        return ! operator == (other);
    }

    bool operator < (const Amount & other) const;
    bool operator <= (const Amount & other) const;
    bool operator > (const Amount & other) const;
    bool operator >= (const Amount & other) const;

    Amount operator - () const
    {
        Amount result = *this;
        result.value = -result.value;
        return result;
    }

    static std::string getCurrencyStr(CurrencyCode currencyCode);
    std::string getCurrencyStr() const;
    static CurrencyCode parseCurrency(const std::string & currency);

    std::string toString() const;

    Json::Value toJson() const;
    static Amount fromJson(const Json::Value & json);
    static Amount parse(const std::string & value);

    void serialize(ML::DB::Store_Writer & store) const;
    void reconstitute(ML::DB::Store_Reader & store);

    int64_t value;
    CurrencyCode currencyCode;
};

std::ostream & operator << (std::ostream & stream, Amount amount);

IMPL_SERIALIZE_RECONSTITUTE(Amount);

struct MicroUSD : public Amount {
    MicroUSD(int64_t value = 0)
        : Amount(CurrencyCode::CC_USD, value)
    {
    }

    MicroUSD(Amount amount)
        : Amount(amount)
    {
        if (amount)
            ExcAssertEqual(currencyCode, CurrencyCode::CC_USD);
    }

    operator int64_t () const { return value; }
};

struct USD : public Amount {
    USD(double value = 0.0)
        : Amount(CurrencyCode::CC_USD, value * 1000000)
    {
    }

    USD(Amount amount)
        : Amount(amount)
    {
        if (amount)
            ExcAssertEqual(currencyCode, CurrencyCode::CC_USD);
    }
    
    operator double () const { return value * 0.000001; }
};

struct USD_CPM : public Amount {
    USD_CPM(double amountInDollarsCPM = 0.0)
        : Amount(CurrencyCode::CC_USD, amountInDollarsCPM * 1000)
    {
    }

    USD_CPM(Amount amount)
        : Amount(amount)
    {
        if (amount)
            ExcAssertEqual(currencyCode, CurrencyCode::CC_USD);
    }

    operator double () const { return value * 0.001; }
};

struct MicroUSD_CPM : public Amount {
    MicroUSD_CPM(int64_t amountInMicroDollarsCPM = 0.0)
        : Amount(CurrencyCode::CC_USD, amountInMicroDollarsCPM / 1000)
    {
    }

    MicroUSD_CPM(Amount amount)
        : Amount(amount)
    {
        if (amount)
            ExcAssertEqual(currencyCode, CurrencyCode::CC_USD);
    }

    operator int64_t () const { return value * 1000; }
};


/*****************************************************************************/
/* CURRENCY POOL                                                             */
/*****************************************************************************/

/** This aggregates amounts over multiple currencies.  The values are kept
    separately so that they can be combined according to application
    specific logic.
*/
struct CurrencyPool {

    void clear()
    {
        currencyAmounts.clear();
    }

    bool empty() const
    {
        return currencyAmounts.empty();
    }

    CurrencyPool()
    {
    }

    CurrencyPool(const Amount & amount)
        : currencyAmounts(1, amount)
    {
        if (!amount)
            currencyAmounts.clear();
    }

    CurrencyPool & operator += (const Amount & amount)
    {
        if (!amount) return *this;

        for (auto & am: currencyAmounts) {
            if (am.currencyCode == amount.currencyCode) {
                am += amount;
                return *this;
            }
        }

        currencyAmounts.push_back(amount);
        std::sort(currencyAmounts.begin(), currencyAmounts.end(),
                  [] (Amount am1, Amount am2)
                  { return am1.currencyCode < am2.currencyCode; });

        return *this;
    }

    CurrencyPool & operator -= (const Amount & amount)
    {
        return operator += (-amount);
    }

    CurrencyPool operator + (const Amount & amount) const
    {
        CurrencyPool result = *this;
        result += amount;
        return result;
    }

    CurrencyPool operator - (const Amount & amount) const
    {
        CurrencyPool result = *this;
        result -= amount;
        return result;
    }

    CurrencyPool & operator += (const CurrencyPool & other)
    {
        for (auto & am: other.currencyAmounts)
            operator += (am);
        return *this;
    }

    CurrencyPool & operator -= (const CurrencyPool & other)
    {
        for (auto & am: other.currencyAmounts)
            operator -= (am);
        return *this;
    }

    CurrencyPool operator + (const CurrencyPool & spend) const
    {
        CurrencyPool result = *this;
        result += spend;
        return result;
    }

    CurrencyPool operator - (const CurrencyPool & spend) const
    {
        CurrencyPool result = *this;
        result -= spend;
        return result;
    }

    /** Limits the other amount so that none of its entries are higher
        than the current pool.
    */
    CurrencyPool limit(const CurrencyPool & other) const
    {
        CurrencyPool result;

        for (auto & am: other.currencyAmounts) {
            Amount a = getAvailable(am.currencyCode).limit(am);
            if (a)
                result.currencyAmounts.push_back(a);
        }

        return result;
    }

    /** Return the maximum of 0 or the amount for each entry. */
    CurrencyPool nonNegative() const
    {
        CurrencyPool result;

        for (auto & am: currencyAmounts) {
            if (am.isNonNegative())
                result.currencyAmounts.push_back(am);
        }

        return result;
    }

    bool operator == (const Amount & other) const
    {
        return operator == (CurrencyPool(other));
    }
    bool operator != (const Amount & other) const
    {
        return ! operator == (other);
    }

    bool operator == (const CurrencyPool & other) const;
    bool operator != (const CurrencyPool & other) const
    {
        return ! operator == (other);
    }

    /** Return if there is enough available to cover the given
        request.
    */
    bool hasAvailable(const Amount & amount) const;

    /** Return the amount available in the given currency code. */
    Amount getAvailable(const CurrencyCode & currency) const;

    bool isNonNegative() const
    {
        for (auto & am: currencyAmounts)
            if (!am.isNonNegative())
                return false;
        return true;
    }

    bool isZero() const
    {
        for (auto & am: currencyAmounts)
            if (!am.isZero())
                return false;
        return true;
    }

    void serialize(ML::DB::Store_Writer & store) const;
    void reconstitute(ML::DB::Store_Reader & store);

    ML::compact_vector<Amount, 1> currencyAmounts;  /// Amounts per currency

    Json::Value toJson() const;
    std::string toString() const;

    static CurrencyPool fromJson(const Json::Value & json);

    bool isSameOrPastVersion(const CurrencyPool & otherPool) const;
};

std::ostream & operator << (std::ostream & stream, CurrencyPool pool);

IMPL_SERIALIZE_RECONSTITUTE(CurrencyPool);


/*****************************************************************************/
/* LINE ITEMS                                                                */
/*****************************************************************************/

/** Maintains a spend per "line item". */

struct LineItems {

    bool isZero() const
    {
        for (const auto & e: entries)
            if (!e.second.isZero())
                return false;
        return true;
    }

    void clear()
    {
        entries.clear();
    }

    bool empty() const
    {
        for (auto & e: entries) {
            if (!e.second.empty())
                return false;
        }
        return true;
    }

    /** Aggregate total spend over line items. */
    CurrencyPool total() const;
    
    CurrencyPool & operator [] (std::string item)
    {
        return entries[item];
    }

    CurrencyPool operator [] (std::string item) const
    {
        auto it = entries.find(item);
        if (it == entries.end())
            return CurrencyPool();
        return it->second;
    }

    static LineItems fromJson(const Json::Value & json);
    Json::Value toJson() const;

    bool operator == (const LineItems & other) const;
    bool operator != (const LineItems & other) const
    {
        return ! operator == (other);
    }

    LineItems & operator += (const LineItems & other);

    void serialize(ML::DB::Store_Writer & store) const;
    void reconstitute(ML::DB::Store_Reader & store);

    std::map<std::string, CurrencyPool> entries;
};

inline std::ostream & operator << (std::ostream & stream, const LineItems & li)
{
    return stream << li.toJson();
}

IMPL_SERIALIZE_RECONSTITUTE(LineItems);

Datacratic::ValueDescriptionT<LineItems> *
getDefaultDescription(LineItems * = 0);

Datacratic::ValueDescriptionT<CurrencyCode> *
getDefaultDescription(CurrencyCode * = 0);

Datacratic::ValueDescriptionT<CurrencyPool> *
getDefaultDescription(CurrencyPool * = 0);

Datacratic::ValueDescriptionT<Amount> *
getDefaultDescription(Amount * = 0);


} // namespace RTBKIT

namespace std {

template<>
struct hash<RTBKIT::CurrencyCode>
{
    size_t operator() (const RTBKIT::CurrencyCode & code) const
    { return static_cast<int>(code); }
};

}

#endif /* __types__currency_h__ */
