/* account_key.h                                                   -*- C++ -*-
   Jeremy Barnes, 24 November 2012
   Copyright (c) 2012 Datacratic.  All rights reserved.

   Key to identify an account.
*/

#pragma once

#include <string>
#include <vector>
#include "jml/utils/string_functions.h"
#include "jml/arch/exception.h"
#include "jml/db/persistent_fwd.h"
#include "soa/jsoncpp/json.h"
#include "soa/service/json_codec.h"
#include "soa/types/basic_value_descriptions.h"
#include <city.h>


namespace RTBKIT {

/** Validate the the given name (campaign or strategy) is valid for our
    system, that is composed of the following:
    
    - Uppercase or lowercase letters
    - numerals
    - underscores
    - maximum length of 64 characters
    - not null
*/
void validateSlug(const std::string & slug);


typedef std::vector<std::string> AccountKeyBase;


/*****************************************************************************/
/* ACCOUNT KEY                                                               */
/*****************************************************************************/

/** Key within the banker to identify an account.  These are always
    hierarchical.
*/
struct AccountKey : public AccountKeyBase {
    AccountKey()
    {
    }

    AccountKey(const std::string & str, char delimiter = ':')
        : AccountKeyBase(ML::split(str, delimiter))
    {
      validate();
    }

    AccountKey(const std::vector<std::string> & vals)
        : AccountKeyBase(vals)
    {
        validate();
    }

    AccountKey(const std::initializer_list<std::string> & vals)
        : AccountKeyBase(vals)
    {
        validate();
    }

    void validate() const
    {
        for (const std::string & slug: *this)
            validateSlug(slug);
    }

    std::string toString(char delimiter = ':') const
    {
        std::string result;
        for (unsigned i = 0;  i < size();  ++i) {
            if (i != 0)
                result += delimiter;
            result += (*this)[i];
        }
        return result;
    }

    AccountKey parent() const
    {
        if (empty())
            throw ML::Exception("no parent");
        AccountKey result = *this;
        result.pop_back();
        return result;
    }

    AccountKey childKey(const std::string & childName) const
    {
        AccountKey result = *this;
        result.push_back(childName);
        return result;
    }

    bool hasPrefix(const AccountKey & otherKey) const
    {
        return size() >= otherKey.size()
            && std::equal(otherKey.begin(), otherKey.end(), begin());
    }

    Json::Value toJson() const
    {
        return Datacratic::jsonEncode(static_cast<AccountKeyBase>(*this));
    }

    static AccountKey fromJson(const Json::Value & json)
    {
        return Datacratic::jsonDecode(json, (AccountKeyBase *)0);
    }

    uint64_t hash() const
    {
        uint64_t res = 1232134;
        for (auto s: *this)
            res = CityHash64WithSeed(s.c_str(), s.size(), res);
        return res;
    }

    bool operator< (const AccountKey& other)
    {
        for (size_t i = 0, n = std::min(size(), other.size()); i < n; ++i) {
            int comp = at(i).compare(other[i]);
            if (comp < 0) return true;
            if (comp > 0) return false;
        }
        return size() < other.size();
    }

    using AccountKeyBase::at;

    std::string at(unsigned int index, const std::string& fallback) const {
        return index < size() ? (*this)[index] : fallback;
    }

    void serialize(ML::DB::Store_Writer & store) const;
    void reconstitute(ML::DB::Store_Reader & store);
};

inline std::ostream &
operator << (std::ostream & stream, const AccountKey & key)
{
    return stream << key.toString();
}

} // namespace RTBKIT

namespace std {

template<>
struct hash<RTBKIT::AccountKey> {
    uint64_t operator () (const RTBKIT::AccountKey & key) const
    {
        return key.hash();
    }
};

} // namespace std

namespace Datacratic {
    template<>
    struct DefaultDescription<RTBKIT::AccountKey>
        : public ValueDescriptionI<RTBKIT::AccountKey, ValueKind::STRING> {

        virtual void parseJsonTyped(RTBKIT::AccountKey * val,
                                    JsonParsingContext & context) const
        {
            *val = RTBKIT::AccountKey(context.expectStringAscii());
        }

        virtual void printJsonTyped(const RTBKIT::AccountKey * val,
                                    JsonPrintingContext & context) const
        {
            context.writeString(val->toString());
        }

        virtual bool isDefaultTyped(const RTBKIT::AccountKey * val) const
        {
            return val->empty();
        }
    };
}

