/* creative_configuration.cc
   Mathieu Stefani, 07 mars 2016
   Copyright (c) 2015 Datacratic.  All rights reserved.
   
*/

#include "creative_configuration.h"
#include "creative_expander.h"

namespace RTBKIT {

CreativeConfiguration::CreativeConfiguration(
        const std::string& exchange)
    : exchange_(exchange)
{
    registerDefaultExpanders();
    registerDefaultFilters();
    registerCustomExpanders();
}

void
CreativeConfiguration::registerDefaultExpanders() {
    expanderDict_ = {
    {
            "exchange",
            std::bind(&CreativeConfiguration::exchange_, this)
    },

    {
        "creative.id",
        [](const Context& ctx)
        { return std::to_string(ctx.creative.id); }
    },

    {
        "creative.name",
        [](const Context& ctx)
        { return ctx.creative.name; }
    },

    {
        "creative.width",
        [](const Context& ctx)
        { return std::to_string(ctx.creative.format.width); }
    },

    {
        "creative.height",
        [](const Context& ctx)
        { return std::to_string(ctx.creative.format.height); }
    },

    {
        "bidrequest.id",
        [](const Context& ctx)
        { return ctx.bidrequest.auctionId.toString(); }
    },

    {
        "bidrequest.user.id",
        [](const Context& ctx) -> std::string
        {
            if ( ctx.bidrequest.user){
                return ctx.bidrequest.user->id.toString();
            }
            return "";
        }
    },

    {
        "bidrequest.publisher.id",
        /* [this](const Context& ctx)  this triggers a gcc bug:
            * http://gcc.gnu.org/bugzilla/show_bug.cgi?id=58824
            */
        [](const Context& ctx) -> std::string
        {
            auto const& br = ctx.bidrequest;
            if (br.site && br.site->publisher) {
                return br.site->publisher->id.toString();
            } else if (br.app && br.app->publisher) {
                return br.app->publisher->id.toString();
            } else {
                std::cerr << "In bid request: " << br.toJson().toString()
                            << " no publisher id found" << std::endl;

                throw std::runtime_error("No publisher id available");
            }
        }
    },

    {
        "bidrequest.timestamp",
        [](const Context& ctx)
        { return std::to_string( ctx.bidrequest.timestamp.secondsSinceEpoch() ); }
    },

    {
        "response.account",
        [](const Context& ctx)
        { return ctx.response.account.toString(); }
    },
    {
        "imp.id",
        [](const Context& context)
        { return context.bidrequest.imp[context.spotNum].id.toString(); }
    }

    };
}

void
CreativeConfiguration::registerDefaultFilters() {
    filters_ = {
        {
            "lower",
            [](std::string& value) -> std::string&
            {
                boost::algorithm::to_lower(value);
                return value;
            }
        },
        {
            "upper",
            [](std::string& value) -> std::string&
            {
                boost::algorithm::to_upper(value);
                return value;
            }
        },
        {
            "urlencode",
            [](std::string& value) -> std::string&
            {
                std::string result;
                for (auto c: value) {
                    if (isalnum(c) || c == '-' || c == '_' || c == '.' || c == '~')
                        result += c;
                    else result += ML::format("%%%02X", c);
                }

                value = std::move(result);
                return value;
            }
        },
    };
}

void
CreativeConfiguration::registerCustomExpanders() {
    auto plugins = PluginInterface<ExpanderBase>::getNames();

    for (const auto& plugin: plugins) {
        auto factory = PluginInterface<ExpanderBase>::getPlugin(plugin);
        auto expander = factory();
        expander->registerExpanders(*this);
    }
}

} // namespace RTBKIT

