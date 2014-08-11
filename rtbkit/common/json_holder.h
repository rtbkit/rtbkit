/* json_holder.h                                                   -*- C++ -*-
   Jeremy Barnes, 1 June 2012
   Copyright (c) 2012 Datacratic.  All rights reserved.

   Container to allow JSON to be accessed, both as a string and as a
   structured object if required.
*/

#pragma once

#include "soa/jsoncpp/json.h"
#include "soa/types/basic_value_descriptions.h"
#include "jml/db/persistent_fwd.h"
#include "jml/utils/unnamed_bool.h"
#include <memory>


namespace RTBKIT {

struct JsonHolderDescription;

/*****************************************************************************/
/* JSON HOLDER                                                               */
/*****************************************************************************/

/** Read-only access to JSON data.  Much more efficient than using a
    Json::Value.
*/

struct JsonHolder {
    JsonHolder()
    {
    }

    JsonHolder(Json::Value && val)
        : parsed(new Json::Value(val))
    {
    }

    JsonHolder(const Json::Value & val)
        : parsed(new Json::Value(val))
    {
    }

    JsonHolder(const std::string & val)
        : str(val)
    {
    }

    ~JsonHolder()
    {
    }

    JsonHolder & operator = (Json::Value && val)
    {
        clear();
        parsed.reset(new Json::Value(val));
        return *this;
    }

    JsonHolder & operator = (const Json::Value & val)
    {
        clear();
        parsed.reset(new Json::Value(val));
        return *this;
    }

    JsonHolder & operator = (const std::string & val)
    {
        clear();
        str = val;
        return *this;
    }

    void clear()
    {
        str.clear();
        parsed.reset();
    }

    static const std::string nullStr;
    
    const std::string & toString() const
    {
        if (!parsed) {
            if (str.empty()) return nullStr;
            return str;
        }
        if (parsed->isNull()) return nullStr;
        makeString();
        return str;
    }

    const Json::Value & toJson() const
    {
        if (parsed) return *parsed;
        makeJson();
        return *parsed;
    }

    void serialize(ML::DB::Store_Writer & store) const;
    void reconstitute(ML::DB::Store_Reader & store);
    
    bool isNonNull() const
    {
        if (parsed) return !parsed->isNull();
        return !str.empty() && str != "null";
    }

#if 0
    bool valid() const
    {
        return parsed || !str.empty();
    }

    JML_IMPLEMENT_OPERATOR_BOOL(valid());
#endif

    mutable std::string str;

private:    
    void makeString() const;
    void makeJson() const;

    mutable std::shared_ptr<const Json::Value> parsed;
};

IMPL_SERIALIZE_RECONSTITUTE(JsonHolder);

std::ostream & operator << (std::ostream & stream, const JsonHolder & json);

} // namespace RTBKIT


namespace Datacratic {

template<>
struct DefaultDescription<RTBKIT::JsonHolder>
  : public StructureDescription<RTBKIT::JsonHolder> {
    DefaultDescription();
};

} // namespace Datacratic
