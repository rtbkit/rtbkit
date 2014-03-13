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

private:
    std::string name_;
    bool required_;
    bool snippet_;
    Handler handler_;
    Json::Value defaultValue_;
};

} // namespace RTBKIT
