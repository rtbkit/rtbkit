/* rtb_agent_config_validator.cc
*/

#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include "rtbkit/core/agent_configuration/agent_config.h"
#include <iostream>

using namespace std;
using namespace RTBKIT;

BOOST_AUTO_TEST_CASE( test_fees )
{
    Amount bidPrice = MicroUSD(5);

    std::string payload;
    Json::Value json;
    std::shared_ptr<Fees> fees;

    /***********************************************************/
    // Null Fees
    payload = R"JSON( {
                "type": "none"
              }
        )JSON";

    json = Json::parse(payload);
    fees = Fees::createFees(json);

    BOOST_CHECK_EQUAL(fees->applyFees(bidPrice), bidPrice);

    
    payload = R"JSON( {
              }
        )JSON";

    json = Json::parse(payload);
    BOOST_CHECK_THROW(Fees::createFees(json),ML::Exception);

    /***********************************************************/
    // Flat Fees
    payload = R"JSON( {
                "type": "flat",
                "params": {
                    "a": "1USD/1M"
                }
            }
        )JSON";

    json = Json::parse(payload);
    fees = Fees::createFees(json);

    BOOST_CHECK_EQUAL(fees->applyFees(bidPrice), bidPrice - MicroUSD(1));

    payload = R"JSON( {
                "type": "flat",
                "params": {
                    "a": "6USD/1M"
                }
            }
        )JSON";

    json = Json::parse(payload);
    fees = Fees::createFees(json);

    BOOST_CHECK_EQUAL(fees->applyFees(bidPrice), MicroUSD(0));

    // Check error handling
    payload = R"JSON( {
                "type": "flat",
                "params": 1 
            }
        )JSON";

    json = Json::parse(payload);
    BOOST_CHECK_THROW(Fees::createFees(json),ML::Exception);

    payload = R"JSON( {
                "type": "flat",
                "params": {
                    "b" : 1
                } 
            }
        )JSON";

    json = Json::parse(payload);
    BOOST_CHECK_THROW(Fees::createFees(json),ML::Exception);

    payload = R"JSON( {
                "type": "flat",
                "params": {
                    "a" : 1.2
                } 
            }
        )JSON";

    json = Json::parse(payload);
    BOOST_CHECK_THROW(Fees::createFees(json),ML::Exception);

    payload = R"JSON( {
                "type": "flat",
                "params": {
                    "a": "-6USD/1M"
                }
            }
        )JSON";

    json = Json::parse(payload);
    BOOST_CHECK_THROW(Fees::createFees(json),ML::Exception);

    /***********************************************************/
    // Linear Fees
    payload = R"JSON( {
                "type": "linear",
                "params": {
                    "a": 1,
                    "b" : -2
                }
            }
        )JSON";

    json = Json::parse(payload);
    fees = Fees::createFees(json);

    BOOST_CHECK_EQUAL(fees->applyFees(bidPrice), bidPrice * 1 + MicroUSD(-2));

    payload = R"JSON( {
                "type": "linear",
                "params": {
                    "a": 1.25,
                    "b" : -10000
                }
            }
        )JSON";

    json = Json::parse(payload);
    fees = Fees::createFees(json);

    BOOST_CHECK_EQUAL(fees->applyFees(bidPrice), MicroUSD(0));

    // Check error handling
    payload = R"JSON( {
                "type": "linear",
                "params": 1
            }
        )JSON";

    json = Json::parse(payload);
    BOOST_CHECK_THROW(Fees::createFees(json),ML::Exception);

    payload = R"JSON( {
                "type": "linear",
                "params": {
                    "b" : 1
                }
            }
        )JSON";

    json = Json::parse(payload);
    BOOST_CHECK_THROW(Fees::createFees(json),ML::Exception);

    payload = R"JSON( {
                "type": "linear",
                "params": {
                    "a" : 1
                }
            }
        )JSON";

    json = Json::parse(payload);
    BOOST_CHECK_THROW(Fees::createFees(json),ML::Exception);

    payload = R"JSON( {
                "type": "linear",
                "params": {
                    "a" : "1"
                }
            }
        )JSON";

    json = Json::parse(payload);
    BOOST_CHECK_THROW(Fees::createFees(json),ML::Exception);

    payload = R"JSON( {
                "type": "linear",
                "params": {
                    "a" : 1,
                    "b" : 10
                }
            }
        )JSON";

    json = Json::parse(payload);
    BOOST_CHECK_THROW(Fees::createFees(json),ML::Exception);

    payload = R"JSON( {
                "type": "linear",
                "params": {
                    "a" : 2,
                    "b" : -10
                }
            }
        )JSON";

    json = Json::parse(payload);
    BOOST_CHECK_THROW(Fees::createFees(json),ML::Exception);
}
