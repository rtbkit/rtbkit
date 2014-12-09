#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <iostream>

#include <boost/test/unit_test.hpp>

#include "rtbkit/core/agent_configuration/agent_config.h"
#include "rtbkit/common/creative_configuration.h"

const std::string providerConfigEmpty = R"FIXTURE(
{
    "test":  { }
}
)FIXTURE";

const std::string providerConfigOnlyRequired = R"FIXTURE(
{
    "test":  { "name" : "thisisme" }
}
)FIXTURE";

const std::string providerConfigComplete = R"FIXTURE(
{
    "test":  {
        "name" : "thisisme",
        "optional" : "helloworld"
    }
}
)FIXTURE";

const std::string SNIPPET1 = "%{bidrequest.id}";
const std::string SNIPPET2 = "%{creative.name}";
const std::string SNIPPET3 = "%{meta.test.coucou}";
const std::string SNIPPET4 = "%{meta.filter#upper}";
const std::string SNIPPET5 = "%{meta.filter#lower}";
const std::string SNIPPET6 = "%{meta.filter#urlencode}";

#define APPLY_ON_SNIPPETS(SNIPPET_FN)   \
    SNIPPET_FN(1, SNIPPET1)             \
    SNIPPET_FN(2, SNIPPET2)             \
    SNIPPET_FN(3, SNIPPET3)             \
    SNIPPET_FN(4, SNIPPET4)             \
    SNIPPET_FN(5, SNIPPET5)             \
    SNIPPET_FN(6, SNIPPET6)

const auto providerConfigSnippet = [&](){
    Json::Value conf;
    auto& test = conf["test"];
# define APPLY_FN(NB, var)\
        test["snippet" #NB] = var;
    APPLY_ON_SNIPPETS(APPLY_FN);
#undef APPLY_FN
    return conf;
}();

namespace {
struct Dummy
{
# define APPLY_FN(NB, var)\
    std::string snippet ## NB;
    APPLY_ON_SNIPPETS(APPLY_FN);
#undef APPLY_FN
};
}

using RTBKIT::CreativeConfiguration;
typedef CreativeConfiguration<Dummy> TestCreativeConfiguration;

RTBKIT::Creative example1 = RTBKIT::Creative::sampleLB;
RTBKIT::Creative example2 = RTBKIT::Creative::sampleBB;
RTBKIT::Creative example3 = RTBKIT::Creative::sampleWS;

BOOST_AUTO_TEST_CASE(test_basic)
{
    TestCreativeConfiguration conf("test");

    example1.providerConfig = Json::parse(providerConfigEmpty);
    example2.providerConfig = Json::parse(providerConfigOnlyRequired);
    example3.providerConfig = Json::parse(providerConfigComplete);

    {
        conf.addField("name",
                      [](const Json::Value &, Dummy &) { return true; });
        conf.addField("optional",
                      [](const Json::Value &, Dummy &) { return true; }
        ).optional();

        auto result = conf.handleCreativeCompatibility(example1, true);
        BOOST_CHECK(!result.isCompatible);
    }

    {
        conf.addField("name",
                      [](const Json::Value & value, Dummy &) {
                          BOOST_CHECK_EQUAL(value.asString(), "thisisme");
                          return true;
                      }
        );

        conf.addField("optional",
                      [](const Json::Value & value, Dummy &) {
                        BOOST_CHECK_EQUAL(value.asString(), "thisisdefault");
                        return true;
                      }
        ).defaultTo("thisisdefault");

        auto result = conf.handleCreativeCompatibility(example2, true);
        BOOST_CHECK(result.isCompatible);
    }

    {
        conf.addField("name",
                      [](const Json::Value & value, Dummy &) {
                        BOOST_CHECK_EQUAL(value.asString(), "thisisme");
                        return true;
                       }
        );

        conf.addField("optional",
                      [](const Json::Value & value, Dummy &) {
                        BOOST_CHECK_EQUAL(value.asString(), "helloworld");
                        return true;
                      }
        ).defaultTo("thisisdefault");

        auto result = conf.handleCreativeCompatibility(example3, true);
        BOOST_CHECK(result.isCompatible);
    }
}

BOOST_AUTO_TEST_CASE(test_snippet)
{
    TestCreativeConfiguration conf("test");

#define APPLY_FN(NB, var)                                        \
    conf.addField("snippet" #NB,                                 \
                  [](const Json::Value & value, Dummy & dummy) { \
                      dummy.snippet##NB = value.asString();      \
                      BOOST_CHECK_EQUAL(dummy.snippet##NB, var); \
                      return true;                               \
                  }).snippet();

    APPLY_ON_SNIPPETS(APPLY_FN)

#undef APPLY_FN

    {
        example1.providerConfig = providerConfigSnippet;
        auto result = conf.handleCreativeCompatibility(example1, true);
        BOOST_CHECK(result.isCompatible);
    }

    RTBKIT::BidRequest bidrequest;
    bidrequest.auctionId = Datacratic::Id("helloworld");
    RTBKIT::Auction::Response response;
    {
        Json::Value value;
        auto& test = value["test"];
        test["coucou"] = "Test:Meta";
        value["filter"] = "Test:Meta";

        response.meta = value.toString();
    }

    TestCreativeConfiguration::Context context{example1, response, bidrequest, 0};
    {

        const auto RESULT1 = "helloworld";
        const auto RESULT2 = example1.name;
        const auto RESULT3 = "Test:Meta";
        const auto RESULT4 = "TEST:META";
        const auto RESULT5 = "test:meta";
        const auto RESULT6 = "Test%3AMeta";

#define APPLY_FN(NB, var) \
        BOOST_CHECK_EQUAL(RESULT ## NB, conf.expand(var, context));
        APPLY_ON_SNIPPETS(APPLY_FN);
#undef APPLY_FN
    }
}

namespace {
struct MyNiceStruct{};

const std::string providerConfigAlternativeMarker = R"FIXTURE(
{
    "test":  { "snippet" : "{{{bidrequest.id}}}" }
}

)FIXTURE";

const std::string EXPECTED = "testid";

}

typedef CreativeConfiguration<MyNiceStruct> CreativeConfigurationInst;


template <>
const std::string CreativeConfigurationInst::VARIABLE_MARKER_BEGIN = "{{{";

template <>
const std::string CreativeConfigurationInst::VARIABLE_MARKER_END = "}}}";


BOOST_AUTO_TEST_CASE(test_overwrite_var_marker)
{
    BOOST_CHECK_EQUAL("{{{", CreativeConfigurationInst::VARIABLE_MARKER_BEGIN);
    BOOST_CHECK_EQUAL("}}}", CreativeConfigurationInst::VARIABLE_MARKER_END);

    CreativeConfigurationInst conf("test");
    example1.providerConfig = Json::parse(providerConfigAlternativeMarker);
    std::string snippet;
    conf.addField("snippet",
                  [&](const Json::Value & value, MyNiceStruct &)
                  {
                      snippet = value.asString();
                      return true;
                  }).snippet();

    auto result = conf.handleCreativeCompatibility(example1, true);
    BOOST_CHECK(result.isCompatible);


    RTBKIT::BidRequest bidrequest;
    bidrequest.auctionId = Datacratic::Id(EXPECTED);
    RTBKIT::Auction::Response response;

    CreativeConfigurationInst::Context context{example1, response, bidrequest, 0};
    BOOST_CHECK_EQUAL(EXPECTED, conf.expand(snippet, context));
}

namespace {
const std::string providerConfigInvalid = R"FIXTURE(
{
    "test":  { "snippet" : "{{{bidrequest.id" }
}
)FIXTURE";

const std::string providerConfigInvalidFilter = R"FIXTURE(
{
    "test":  { "snippet" : "{{{bidrequest.id#filter2}}}" }
}
)FIXTURE";

const std::string providerConfigInvalidVar = R"FIXTURE(
{
    "test":  { "snippet" : "{{{toto.id}}}" }
}
)FIXTURE";

}

BOOST_AUTO_TEST_CASE(test_error)
{
    CreativeConfigurationInst conf("test");

    example1.providerConfig = Json::parse(providerConfigInvalid);
    std::string snippet;

    conf.addField("snippet",
                 [&](const Json::Value& value, MyNiceStruct&)
                 {
                     snippet = value.asString();
                     return true;
                 }).snippet();

    BOOST_CHECK_THROW(conf.handleCreativeCompatibility(example1, true),
                      std::invalid_argument);


    example1.providerConfig = Json::parse(providerConfigInvalidFilter);
    BOOST_CHECK_THROW(conf.handleCreativeCompatibility(example1, true),
                      std::runtime_error);

    example1.providerConfig = Json::parse(providerConfigInvalidVar);
    BOOST_CHECK_THROW(conf.handleCreativeCompatibility(example1, true),
                      std::runtime_error);
}
