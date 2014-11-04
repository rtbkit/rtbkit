#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include "rtbkit/common/bids.h"
#include <boost/test/unit_test.hpp>

using namespace std;
using namespace RTBKIT;
using namespace Datacratic;


BOOST_AUTO_TEST_CASE(creativeMatrixTest)
{
  std::string testStr = "{\"bids\":[{\"spotIndex\":0,\"price\":\"2000USD/1M\",\"ext\":[\"test1\",\"test2\"]}]}";
  Bids bidObj = Bids::fromJson(testStr);

  std::string jsonStr = bidObj.toJsonStr();
  
  BOOST_CHECK_EQUAL(bidObj[0].spotIndex, 0);
  BOOST_CHECK_EQUAL(bidObj[0].ext.toStringNoNewLine(), "[\"test1\",\"test2\"]");
}
