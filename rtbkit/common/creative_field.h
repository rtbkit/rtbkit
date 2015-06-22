#pragma once

#include <string>
#include <functional>
#include <soa/jsoncpp/value.h>

namespace RTBKIT
{

template <typename CreativeData>
class CreativeField
{
public:
    typedef std::function<bool(const Json::Value &, CreativeData &)> Handler;

    static const char Delimiter = '.';

    CreativeField()
    : name_()
    , required_(true)
    , snippet_(false)
    , handler_()
    , defaultValue_()
    {
    }

    CreativeField(const std::string & name, Handler handler)
    : name_(name)
    , required_(true)
    , snippet_(false)
    , handler_(handler)
    , defaultValue_()
    {
    }

    CreativeField & required()
    {
        required_ = true;
        return *this;
    }

    CreativeField & optional()
    {
        required_ = false;
        return *this;
    }

    CreativeField & snippet(bool value = true)
    {
        snippet_ = value;
        return *this;
    }

    template <typename T>
    CreativeField & defaultTo(T && value)
    {
        defaultValue_ = value;
        required_ = false;
        return *this;
    }

    bool operator()(const Json::Value & value, CreativeData & data) const
    {
        return handler_(value, data);
    }

    const std::string getName() const
    {
        return name_;
    }

    const Json::Value & getDefaultValue() const
    {
        return defaultValue_;
    }

    bool isRequired() const
    {
        return required_;
    }

    bool isSnippet() const
    {
        return snippet_;
    }

    Json::Value extractJsonValue(const Json::Value& value) const {
        if (JML_UNLIKELY(value.isNull())) {
            return Json::Value::null;
        }

        auto split = [](const std::string& str, char delim) {
            std::istringstream iss(str);
            std::vector<std::string> res;

            std::string elem;
            while (std::getline(iss, elem, delim)) {
                res.push_back(elem);
            }
            return res;
        };

        const auto parts = split(name_, Delimiter);
        ExcAssertGreater(parts.size(), 0);
        Json::Value currentVal(value);
        for (const auto& part: parts) {
            currentVal = currentVal[part];

            if (currentVal.isNull() && required_) {
                return Json::Value::null;
            }

        }

        return currentVal.isNull() ? defaultValue_ : currentVal;

    }

private:
    std::string name_;
    bool required_;
    bool snippet_;
    Handler handler_;
    Json::Value defaultValue_;
};

} // namespace RTBKIT
