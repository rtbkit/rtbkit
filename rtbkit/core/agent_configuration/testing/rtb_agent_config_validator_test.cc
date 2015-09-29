/* rtb_agent_config_validator.cc
*/

#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include "rtbkit/core/agent_configuration/agent_config.h"
#include <iostream>

using namespace std;
using namespace RTBKIT;

BOOST_AUTO_TEST_CASE( test_agent_config_fees )
{
    AgentConfig config;
    std::string payload;

    // Normal case : No fees / NO EXCEPTION expected
    payload = R"JSON( {
            "account" : ["hello", "worlds"],
            "creatives": [
            {
                "name": "MaCreative",
                "height": 250,
                "width": 300,
                "languageFilter": {
                "include": []
                },
                "id": 5
            }]}
        )JSON";


    config.parse(payload);

    // Normal case : NullFees : NO EXCEPTION expected
    payload = R"JSON( {
            "account" : ["hello", "worlds"],
            "creatives": [
            {
                "name": "MaCreative",
                "height": 250,
                "width": 300,
                "languageFilter": {
                "include": []
                },
                "id": 5,
                "fees": {
                    "type": "none"
                }
            }]}
        )JSON";


    config.parse(payload);

    // Normal case : Flat Fees : NO EXCEPTION expected
    payload = R"JSON( {
            "account" : ["hello", "worlds"],
            "creatives": [
            {
                "name": "MaCreative",
                "height": 250,
                "width": 300,
                "languageFilter": {
                "include": []
                },
                "id": 5,
                "fees": {
                    "type": "flat",
                    "params" : {
                        "a" : "10USD/1M"
                    }
                }
            }]}
        )JSON";


    config.parse(payload);

    // Normal case : NO EXCEPTION expected
    payload = R"JSON( {
            "account" : ["hello", "worlds"],
            "creatives": [
            {
                "name": "MaCreative",
                "height": 250,
                "width": 300,
                "languageFilter": {
                "include": []
                },
                "id": 5,
                "fees": {
                    "type": "linear",
                    "params": {
                        "a": 0.1,
                        "b": -0.2
                    }
                }
            }]}
        )JSON";


    config.parse(payload);
}

BOOST_AUTO_TEST_CASE( test_agent_config_language_filter )
{
    AgentConfig config;
    std::string payload;

    // Normal case : Empty array : NO EXCEPTION expected
    payload = R"JSON( {
            "account" : ["hello", "worlds"],
            "creatives": [
            {
                "name": "MaCreative",
                "height": 250,
                "width": 300,
                "languageFilter": {
                "include": []
                },
                "id": 5
            }]}
        )JSON";


    config.parse(payload);

    // Normal case : NO EXCEPTION expected
    payload = R"JSON( {
            "account" : ["hello", "worlds"],
            "creatives": [
            {
                "name": "MaCreative",
                "height": 250,
                "width": 300,
                "id": 5
            }]}
        )JSON";


    config.parse(payload);

    // Include is null
    payload = R"JSON( {
            "account" : ["hello", "worlds"],
            "creatives": [
            {
                "name": "MaCreative",
                "height": 250,
                "width": 300,
                "languageFilter": {
                "include": null
                },
                "id": 5
            }]}
        )JSON";

    BOOST_CHECK_THROW(config.parse(payload),ML::Exception);

    // Normal case : Include
    payload = R"JSON( {
            "account" : ["hello", "worlds"],
            "creatives": [
            {
                "name": "MaCreative",
                "height": 250,
                "width": 300,
                "languageFilter": {
                "include": ["zh"]
                },
                "id": 5
            }]}
        )JSON";


    config.parse(payload);
    BOOST_CHECK_EQUAL(config.creatives[0].languageFilter.include[0], "zh");

    // Normal case : Exclude 
    payload = R"JSON( {
            "account" : ["hello", "worlds"],
            "creatives": [
            {
                "name": "MaCreative",
                "height": 250,
                "width": 300,
                "languageFilter": {
                "exclude": ["de"]
                },
                "id": 5
            }]}
        )JSON";


    config.parse(payload);
    BOOST_CHECK_EQUAL(config.creatives[0].languageFilter.exclude[0], "de");

    // Include is not an array
    payload = R"JSON( {
            "account" : ["hello", "worlds"],
            "creatives": [
            {
                "name": "MaCreative",
                "height": 250,
                "width": 300,
                "languageFilter": {
                "include": "zh"
                },
                "id": 5
            }]}
        )JSON";


    BOOST_CHECK_THROW(config.parse(payload),ML::Exception);


    // Exclude is not an array
   payload = R"JSON( {
            "account" : ["hello", "worlds"],
            "creatives": [
            {
                "name": "MaCreative",
                "height": 250,
                "width": 300,
                "languageFilter": {
                "exclude": "zh"
                },
                "id": 5
            }]}
        )JSON";


    BOOST_CHECK_THROW(config.parse(payload),ML::Exception);
}
