/* creative_expander.h
   Mathieu Stefani, 07 mars 2016
   Copyright (c) 2015 Datacratic.  All rights reserved.
   
   Expander plugin system for the CreativeConfiguration class
*/

#include "creative_configuration.h"
#include "rtbkit/common/plugin_interface.h"

namespace RTBKIT {

class ExpanderBase {
public:
    virtual void registerExpanders(CreativeConfiguration& config) = 0;

    /*************************************************************************/
    /* FACTORY INTERFACE                                                     */
    /*************************************************************************/

    /** Type of a callback which is registered as an exchange factory */
    typedef std::function<ExpanderBase * ()> Factory;

    /** plugin interface needs to be able to request the root name of the plugin library */
    static const std::string libNameSufix() {return "expander";};

    /** Create a new filter from a factory */
    static std::unique_ptr<ExpanderBase>
    create(const std::string& expander) {
        auto factory = PluginInterface<ExpanderBase>::getPlugin(expander);
        return std::unique_ptr<ExpanderBase>(factory());
    }
};

namespace meta {
    template<typename... > struct void_t { typedef void type; };

    template<bool B> struct true_t;

    template<> struct true_t<true> {
        typedef std::true_type type;

        static constexpr bool value = false;
    };

}

namespace details {
    template<typename Exchange, typename = void> struct IsExchangeConnector : public std::false_type { };

    template<typename Exchange>
    struct IsExchangeConnector<Exchange,
        typename meta::void_t<
            typename meta::true_t<std::is_base_of<ExchangeConnector, Exchange>::value>::type,
            decltype(&Exchange::exchangeNameString)
        >::type
    > : public std::true_type { };

}

template<typename Exchange>
class ExchangeExpander : public ExpanderBase {
    static_assert(details::IsExchangeConnector<Exchange>::value,
            "The Exchange must inherit from ExchangeConnector and define a static exchangeNameString function");

    virtual void registerExchangeExpanders(CreativeConfiguration& config) = 0;

    void registerExpanders(CreativeConfiguration& config) {
        if (config.exchange() == Exchange::exchangeNameString()) {
            registerExchangeExpanders(config);
        }
    }
};

} // namespace RTBKIT
