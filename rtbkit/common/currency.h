/* currency.h                                                     -*- C++ -*-
   Jeremy Barnes, 10 October 2012
   Copyright (c) 2012 Datacratic Inc.  All rights reserved.

*/

#ifndef __types__currency_h__
#define __types__currency_h__

#include <cstddef>
#include <ratio>
#include <type_traits>

#include <boost/preprocessor/cat.hpp>


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
    CC_EUR  = 'E' << 24 | 'U' << 16 | 'R' << 8,       // micro euros
    CC_IMP  = 'I' << 24 | 'M' << 16 | 'P' << 8,
    CC_CLK  = 'C' << 24 | 'L' << 16 | 'K' << 8
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

typedef std::ratio<1, 1> Micro;
typedef std::ratio<std::micro::den, std::micro::num> NaturalCurrency;
typedef std::ratio<std::micro::den / 1000, std::micro::num> CPM;
typedef std::ratio_multiply<std::micro, std::ratio<1000, 1>> MicroCPM;

template <typename Ratio>
struct PriceIntegerType
{
    typedef typename std::conditional<(Ratio::den > Ratio::num), double, int64_t>::type type;
};

#define CHECK_PRICE_INTEGER_TYPE(price_ratio, result_type)                          \
    static_assert(                                                                  \
        std::is_same<                                                               \
            /* ratio reverted because we want make sure that the way back has */    \
            /*   the proper type */                                                 \
            typename PriceIntegerType<std::ratio<price_ratio::den, price_ratio::num>>::type, \
            result_type>::value,                                                    \
        "Something's wrong")

CHECK_PRICE_INTEGER_TYPE(Micro, int64_t);
CHECK_PRICE_INTEGER_TYPE(NaturalCurrency, double);
CHECK_PRICE_INTEGER_TYPE(CPM, double);
CHECK_PRICE_INTEGER_TYPE(MicroCPM, int64_t);

#undef CHECK_PRICE_INTEGER_TYPE

template <CurrencyCode CURRENCY, typename Ratio>
struct CurrencyTemplate : public Amount
{
    template <typename T>
    static inline T currentRatioToBaseRatio(T value)
    {
        return (value * Ratio::num) / Ratio::den;
    }

    CurrencyTemplate()
    : Amount(CURRENCY, 0)
    {
    }

    template <typename T,
              typename = typename std::enable_if<std::is_arithmetic<T>::value>::type>
    CurrencyTemplate(T value)
    : Amount(CURRENCY, currentRatioToBaseRatio(value))
    {
    }

    CurrencyTemplate(const Amount& amount)
    : Amount(amount)
    {
        if (amount)
            ExcAssertEqual(currencyCode, CURRENCY);
    }

    template <typename R>
    bool operator==(const CurrencyTemplate<CURRENCY, R>& rhs) const
    {
        return Amount::operator==(rhs);
    }

    template <typename T>
    operator T() const;
};

namespace detail {

template <typename Ratio>
struct CurrencyConverter
{
    typedef std::ratio<Ratio::den, Ratio::num> RevertedRatio;
    typedef decltype(std::declval<Amount>().value) ValueType;

    template <CurrencyCode code>
    CurrencyConverter(const CurrencyTemplate<code, Ratio>& currency)
    : value(currency.value)
    , code(currency.currencyCode)
    {
    }

    template <typename T,
              typename ReturnType = typename PriceIntegerType<RevertedRatio>::type,
              typename = typename std::enable_if<std::is_arithmetic<T>::value>::type>
    inline ReturnType baseRatioToCurrentRatio(T value) const
    {
        static_assert(std::is_same<ReturnType, T>::value,
                      "Invalid type cast required");
        return (ReturnType(value) * Ratio::den) / Ratio::num;
    }

    template <typename T, typename ReturnType = typename std::decay<T>::type>
    inline ReturnType baseRatioToCurrentRatio(...) const
    {
        static_assert(std::is_convertible<Amount, ReturnType>::value,
                      "Cast is impossible");
        return Amount{code, value};
    }

    template <typename T>
    operator T() const
    {
        return baseRatioToCurrentRatio<T>(value);
    }

    ValueType value;
    CurrencyCode code;
};

} // namespace detail

template <CurrencyCode CURRENCY, typename Ratio>
template <typename T>
CurrencyTemplate<CURRENCY, Ratio>::operator T() const
{
    return detail::CurrencyConverter<Ratio>(*this);
}


template <CurrencyCode, typename Ratio>
inline detail::CurrencyConverter<Ratio> amountToCurrencyRatio(const Amount& amount);

#define CURRENCY EUR
#include "currency.h.in"
#undef CURRENCY

#define CURRENCY USD
#include "currency.h.in"
#undef CURRENCY

template<typename Ratio = Micro>
inline detail::CurrencyConverter<Ratio> getAmountIn(CurrencyCode currency, const Amount& amount)
{
    switch (currency)
    {
        case CurrencyCode::CC_EUR:
            return amountToCurrencyRatio<CurrencyCode::CC_EUR, Ratio>(amount);
        case CurrencyCode::CC_USD:
            return amountToCurrencyRatio<CurrencyCode::CC_USD, Ratio>(amount);

        default:
            throw std::runtime_error("Cannot convert amount to currency: " +
                                     toString(currency));
    };
}

template<typename Ratio = Micro>
inline detail::CurrencyConverter<Ratio> getAmountIn(const Amount& amount)
{
    return getAmountIn<Ratio>(amount.currencyCode, amount);
}

template <CurrencyCode CURRENCY, typename Ratio, typename Integer>
inline Amount createAmount(Integer amount)
{ return CurrencyTemplate<CURRENCY, Ratio>(amount); }

template <typename Ratio, typename Integer>
Amount createAmount(Integer amount, CurrencyCode currency)
{
    switch (currency)
    {
        case CurrencyCode::CC_EUR:
            return createAmount<CurrencyCode::CC_EUR, Ratio>(amount);
        case CurrencyCode::CC_USD:
            return createAmount<CurrencyCode::CC_USD, Ratio>(amount);

        default:
            throw std::runtime_error("Cannot convert create amount of: " +
                                     toString(currency));
    };
}

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

    CurrencyPool operator *= (double factor)
    {
        for (auto & am: currencyAmounts) am = am * factor;
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

    CurrencyPool operator * (double factor) const
    {
        CurrencyPool result = *this;
        result *= factor;
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
