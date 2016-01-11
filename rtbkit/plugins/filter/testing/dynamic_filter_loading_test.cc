/** loading_filters_dynamically_test
    
    Sirma Cagil Altay 19 Oct 2015

    Test dynamic filter loading using 
    custom_filter
*/

#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include "rtbkit/core/router/router.h"
#include "jml/arch/info.h"
#include "jml/utils/file_functions.h"
#include "rtbkit/common/filter.h"

using namespace std;
using namespace ML;
using namespace Datacratic;
using namespace RTBKIT;

BOOST_AUTO_TEST_CASE( test_filter_dynamic_loading_1 )
{
    string configFile = R"({"filterMask":["My"]})";
    Json::Value filtersConfig = Json::parse(configFile);
    
    Router router;
    BOOST_REQUIRE_THROW(router.initFilters(filtersConfig),ML::Exception);
}

BOOST_AUTO_TEST_CASE( test_filter_dynamic_loading_2 )
{
    string configFile = R"({})";
    Json::Value filtersConfig = Json::parse(configFile);
    
    Router router;
    router.initFilters(filtersConfig);

    std::vector<string> refined_filters = router.filters.getFilterNames();

    bool found_My = false;
    bool found_Your = false;

    for(int i=0; i<refined_filters.size(); i++){
        if (refined_filters[i] == "My")
           found_My = true;

        if (refined_filters[i] == "Your")
           found_Your = true;
    }

    BOOST_REQUIRE(refined_filters.size()>2);
    BOOST_REQUIRE(found_My==false);
    BOOST_REQUIRE(found_Your==false);
}

BOOST_AUTO_TEST_CASE( test_filter_dynamic_loading_3 )
{
    string configFile = R"({"extraFilterFiles":["custom_filter"]})";
    Json::Value filtersConfig = Json::parse(configFile);
    
    Router router;
    router.initFilters(filtersConfig);

    std::vector<string> refined_filters = router.filters.getFilterNames();
    
    bool found_My = false;
    bool found_Your = false;

    for(int i=0; i<refined_filters.size(); i++){
        if (refined_filters[i] == "My")
           found_My = true;

        if (refined_filters[i] == "Your")
           found_Your = true;
    }

    BOOST_REQUIRE(refined_filters.size()>2);
    BOOST_REQUIRE(found_My==true);
    BOOST_REQUIRE(found_Your==true);
}

BOOST_AUTO_TEST_CASE( test_filter_dynamic_loading_4 )
{
    string configFile = R"({"filterMask":["My"],"extraFilterFiles":["custom_filter"]})";
    Json::Value filtersConfig = Json::parse(configFile);
    
    Router router;
    router.initFilters(filtersConfig);

    vector<string> refined_filters = router.filters.getFilterNames();

    BOOST_REQUIRE(refined_filters.size()==1);
    BOOST_REQUIRE(refined_filters[0]=="My");
}

BOOST_AUTO_TEST_CASE( test_filter_dynamic_loading_5 )
{
    ML::File_Read_Buffer buf("rtbkit/plugins/filter/testing/test-filter-config.json");
    Json::Value filtersConfig = Json::parse(std::string(buf.start(), buf.end()));
    
    Router router;
    router.initFilters(filtersConfig);

    vector<string> refined_filters = router.filters.getFilterNames();

    BOOST_REQUIRE(refined_filters.size()==2);
    BOOST_REQUIRE(refined_filters[0]=="Your");
    BOOST_REQUIRE(refined_filters[1]=="My");
}
