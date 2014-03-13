/* parse_context_test.cc                                           -*- C++ -*-
   Jeremy Barnes, 16 February 2007
   Copyright (c) 2007 Jeremy Barnes.  All rights reserved.

   Test of tick counter functionality.
*/

#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include "jml/utils/configuration.h"
#include "jml/utils/enum_info.h"
#include <boost/test/unit_test.hpp>
#include "jml/utils/vector_utils.h"

using namespace ML;
using namespace std;

using boost::unit_test::test_suite;

static const char * config1 = 
"\n"
"# classifier-training.txt                                  -*- awk -*-\n"
"#\n"
"# Training parameters for the classifier training\n"
"\n"
"# classifier for small datasets.\n"
"bagged_boosted_trees {\n"
"    # main classifier makes 10 bags for stability\n"
"    type=bagging;\n"
"    num_bags=5;\n"
"    verbosity=3;\n"
"    \n"
"    # weak learner for bagging is boosted decision trees.  Boosting is good\n"
"    # for functions like this as has a smooth profile due to margin\n"
"    # maximization.\n"
"    weak_learner {\n"
"        type=boosting;\n"
"        \n"
"        min_iter=5;\n"
"        max_iter=20;\n"
"        \n"
"        verbosity=3;\n"
"        \n"
"        # weak learner for boosting is decision trees of depth 10.  A high\n"
"        # capacity classifier is needed to model the interraction between "
"        # the categorical variables.\n"
"        weak_learner {\n"
"            type=decision_tree;\n"
"            max_depth=6;\n"
"            \n"
"            # use an update algorithm suitable for boosting\n"
"            update_alg=gentle;\n"
"            \n"
"            # don't print out trees as we go\n"
"            verbosity=0;\n"
"        }\n"
"    }\n"
"}\n"
"\n"
"# classifier for when we have large datasets.\n"
"bagged_trees {\n"
"    # main classifier makes 10 bags for stability\n"
"    type=bagging;\n"
"    num_bags=10;\n"
"    verbosity=3;\n"
"    \n"
"    # weak learner for bagging is decision trees.\n"
"    weak_learner {\n"
"        type=decision_tree;\n"
"        max_depth=12;\n"
"        \n"
"        # don't print out trees as we go\n"
"        verbosity=0;\n"
"    }\n"
"}\n"
"\n"
"# Here we override any specific parameters.\n"
"case1 {\n"
"    sub1 : bagged_trees {\n"
"    }\n"
"\n"
"    sub2 : bagged_trees {\n"
"    }\n"
"\n"
"    sub3 : bagged_trees {\n"
"    }\n"
"\n"
"    sub4 : bagged_trees {\n"
"    }\n"
"\n"
"    sub5 : bagged_trees {\n"
"    }\n"
"}\n"
"\n"
"# Case 2 uses bagged boosted trees as we have less training data\n"
"case2 : case1 {\n"
"    sub1 : bagged_boosted_trees {\n"
"    }\n"
"\n"
"    sub2 : bagged_boosted_trees {\n"
"    }\n"
"\n"
"    sub3 : bagged_boosted_trees {\n"
"    }\n"
"\n"
"    sub4 : bagged_boosted_trees {\n"
"    }\n"
"\n"
"    sub5 : bagged_boosted_trees {\n"
"    }\n"
"}\n"
"\n"
"setting=0;\n"
"outer {\n"
"    setting=1;\n"
"    inner {\n"
"        setting=2;\n"
"    }\n"
"}\n"
"vector_values=1,2,3,4,5,6;\n"
"vector_values2=1,2,3,4,5,6\n"
"vector_values3 = 1,2,3,4,5,6\n"
"vector_values4 = \"w\",\"x\",\"y\";";
    
enum Update {
    NORMAL = 0,   ///< Normal update
    GENTLE = 1,   ///< Gentle update
    PROB   = 2    ///< Probabilistic update
};

DECLARE_ENUM_INFO(Update, 3);

ENUM_INFO_NAMESPACE

const Enum_Opt<Update>
Enum_Info<Update>::OPT[3] = {
    { "normal",      NORMAL   },
    { "gentle",      GENTLE   },
    { "prob",        PROB     } };

const char * Enum_Info<Update>::NAME = "Update";

END_ENUM_INFO_NAMESPACE

BOOST_AUTO_TEST_CASE( test1 )
{
    Configuration config;
    config.parse_string(config1, "config1 test configuration");

    BOOST_CHECK_EQUAL(config["bagged_boosted_trees.type"], "bagging");
    string t;
    BOOST_CHECK(config.find(t, "bagged_boosted_trees.type"));
    BOOST_CHECK_EQUAL(t, "bagging");

    Configuration config2(config, "bagged_boosted_trees",
                          Configuration::PREFIX_REPLACE);
    BOOST_CHECK_EQUAL(config2["type"], "bagging");
    t = "";
    BOOST_CHECK(config2.find(t, "type"));

    BOOST_CHECK_EQUAL(t, "bagging");
    BOOST_CHECK_THROW(config2["hello"] = "hello", ML::Exception);

    Configuration config3(config2, "weak_learner",
                          Configuration::PREFIX_APPEND);
    BOOST_CHECK_EQUAL(config3["type"], "boosting");
    t = "";
    BOOST_CHECK(config3.find(t, "type"));
    BOOST_CHECK_EQUAL(t, "boosting");
    BOOST_CHECK_THROW(config3["hello"] = "hello", ML::Exception);

    Configuration config4(config3, "weak_learner",
                          Configuration::PREFIX_APPEND);
    BOOST_CHECK_EQUAL(config4["type"], "decision_tree");
    t = "";
    BOOST_CHECK(config4.find(t, "type"));
    BOOST_CHECK_EQUAL(t, "decision_tree");
    BOOST_CHECK_THROW(config4["hello"] = "hello", ML::Exception);

    config["bagged_boosted_trees.weak_learner.weak_learner.type"]
        = "naive_bayes";

    BOOST_CHECK_EQUAL(config["bagged_boosted_trees.weak_learner.weak_learner.type"], "naive_bayes");
    BOOST_CHECK_EQUAL(config2["weak_learner.weak_learner.type"], "naive_bayes");
    BOOST_CHECK_EQUAL(config3["weak_learner.type"], "naive_bayes");
    BOOST_CHECK_EQUAL(config4["type"], "naive_bayes");

    unsigned u = 0;
    BOOST_CHECK(config.get(u, "bagged_boosted_trees.num_bags"));
    BOOST_CHECK_EQUAL(u, 5);
    u = 0;
    BOOST_CHECK(config.find(u, "bagged_boosted_trees.num_bags"));
    BOOST_CHECK_EQUAL(u, 5);
    u = 0;
    config.must_find(u, "bagged_boosted_trees.num_bags");
    BOOST_CHECK_EQUAL(u, 5);
    u = 0;
    config.require(u, "bagged_boosted_trees.num_bags");
    BOOST_CHECK_EQUAL(u, 5);
    u = 0;

    BOOST_CHECK(config2.get(u, "num_bags"));
    BOOST_CHECK_EQUAL(u, 5);
    u = 0;
    BOOST_CHECK(config2.find(u, "num_bags"));
    BOOST_CHECK_EQUAL(u, 5);
    u = 0;
    config2.must_find(u, "num_bags");
    BOOST_CHECK_EQUAL(u, 5);
    u = 0;
    config2.require(u, "num_bags");
    BOOST_CHECK_EQUAL(u, 5);

    /* Test enums */
    Update update = (Update)-1;
    BOOST_CHECK(config.get(update, "bagged_boosted_trees.weak_learner.weak_learner.update_alg"));
    BOOST_CHECK_EQUAL(update, GENTLE);

    BOOST_CHECK_THROW(config.get(update, "bagged_boosted_trees.weak_learner.type"), std::exception);

    vector<string> extra;
    extra.push_back("param1=value1");

    config.parse_command_line(extra);
    BOOST_CHECK_EQUAL(config["param1"], "value1");

    extra.clear();
    extra.push_back("param1.param2=value2");

    config.parse_command_line(extra);
    BOOST_CHECK_EQUAL(config["param1.param2"], "value2");

    extra.clear();
    extra.push_back("weak_learner.weak_learner.type=bonus");

    config.parse_command_line(extra);
    BOOST_CHECK_EQUAL(config["weak_learner.weak_learner.type"], "bonus");

    vector<int> v;
    config.get(v, "vector_values");
    BOOST_CHECK_EQUAL(v.size(), 6);
    BOOST_CHECK_EQUAL(v.at(0), 1);
    BOOST_CHECK_EQUAL(v.at(1), 2);
    BOOST_CHECK_EQUAL(v.at(2), 3);
    BOOST_CHECK_EQUAL(v.at(3), 4);
    BOOST_CHECK_EQUAL(v.at(4), 5);
    BOOST_CHECK_EQUAL(v.at(5), 6);

    v.clear();
    config.get(v, "vector_values2");
    BOOST_CHECK_EQUAL(v.size(), 6);
    BOOST_CHECK_EQUAL(v.at(0), 1);
    BOOST_CHECK_EQUAL(v.at(1), 2);
    BOOST_CHECK_EQUAL(v.at(2), 3);
    BOOST_CHECK_EQUAL(v.at(3), 4);
    BOOST_CHECK_EQUAL(v.at(4), 5);
    BOOST_CHECK_EQUAL(v.at(5), 6);

    v.clear();
    config.get(v, "vector_values3");
    BOOST_CHECK_EQUAL(v.size(), 6);
    BOOST_CHECK_EQUAL(v.at(0), 1);
    BOOST_CHECK_EQUAL(v.at(1), 2);
    BOOST_CHECK_EQUAL(v.at(2), 3);
    BOOST_CHECK_EQUAL(v.at(3), 4);
    BOOST_CHECK_EQUAL(v.at(4), 5);
    BOOST_CHECK_EQUAL(v.at(5), 6);

#if 0
    vector<string> v2;
    config.get(v2, "vector_values4");
    BOOST_CHECK_EQUAL(v2.size(), 3);
    BOOST_CHECK_EQUAL(v2.at(0), "w");
    BOOST_CHECK_EQUAL(v2.at(1), "x");
    BOOST_CHECK_EQUAL(v2.at(2), "y");
#endif
}
