#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include "soa/service/sns.h"

using namespace std;
using namespace Datacratic;

BOOST_AUTO_TEST_CASE( test_mock_sns_api )
{
    auto sns = make_shared<MockSnsApiWrapper>(3);
    sns->publish("coco");
    sns->publish("caramba");
    sns->publish("caramel");

    BOOST_CHECK(sns->queue.front() == "coco");
    sns->queue.pop();
    BOOST_CHECK(sns->queue.front() == "caramba");
    sns->queue.pop();
    BOOST_CHECK(sns->queue.front() == "caramel");
    sns->queue.pop();

    sns->publish("coco");
    sns->publish("caramba");
    sns->publish("caramel");
    sns->publish("choucroute");
    sns->publish("chapelure");
    BOOST_CHECK(sns->queue.front() == "caramel");
}
