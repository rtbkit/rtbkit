/** augmentation_test.cc                                 -*- C++ -*-
    Rémi Attab, 15 Jan 2013
    Copyright (c) 2013 Datacratic.  All rights reserved.

    tests for the augmentation list structure.

*/

#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include "rtbkit/common/augmentation.h"

#include <boost/test/unit_test.hpp>


using namespace std;
using namespace ML;
using namespace Datacratic;
using namespace RTBKIT;


struct AugmentationFixture
{
    AccountKey accA, accBB, accBC, accBBA;
    string tag0, tag1, tag2;
    Json::Value data0, data1, data2;

    AugmentationFixture()
    {
        accA = { "A" };
        accBB = { "B", "B" };
        accBC = { "B", "C" };
        accBBA = { "B", "B", "A" };

        tag0 = "a-tag";
        tag1 = "bob-the-tag";
        tag2 = "not-a-tag-tag";

        data0 = Json::Value(Json::objectValue);
        data0["aug0"] = 42;

        data1 = Json::Value(Json::objectValue);
        data1["bob-the-augmentor"] = "bob-the-data-for-the-augmentor";

        data2 = Json::Value(Json::objectValue);
        Json::Value& arr = data2["another-augmentor"] = Json::Value(Json::arrayValue);
        arr.append(10);
        arr.append("this is not a number");
    }

};

void checkTags(const set<string>& tags, const initializer_list<string>& exp)
{
    // cerr << "TAGS: { ";
    // for (const string& tag : tags) cerr << tag << ", ";
    // cerr << "} => { ";
    // for (const string& tag : exp) cerr << tag << ", ";
    // cerr << "}" << endl;

    BOOST_CHECK_EQUAL(tags.size(), exp.size());
    if (tags.size() != exp.size()) return;

    bool ret = equal(tags.begin(), tags.end(), exp.begin());
    BOOST_CHECK(ret);
}

void checkData(
        const Json::Value& data, const initializer_list<Json::Value>& exp)
{
    // cerr << "DATA: " << data.toString();
    // cerr << "EXPECTED: " << endl;
    // for (const auto& data : exp) cerr << "\t" << data.toString();

    int totalFound = 0;

    vector<string> members = data.getMemberNames();
    for (const string& member : members) {

        bool found = false;
        for (const Json::Value & val : exp) {
            if (!val.isMember(member)) continue;
            BOOST_CHECK_EQUAL(val[member], data[member]);
            found = true;
            totalFound++;
            break;
        }

        BOOST_CHECK(found);
    }

    BOOST_CHECK_EQUAL(totalFound, exp.size());
}

void check(
        const AugmentationList & list,
        const AccountKey & key,
        const initializer_list<string>& expTags,
        const initializer_list<Json::Value>& expData)
{
    auto aug = list.filterForAccount(key);
    checkData(aug.data, expData);
    checkTags(aug.tags, expTags);

    auto tags = list.tagsForAccount(key);
    BOOST_CHECK_EQUAL(tags.size(), aug.tags.size());
    BOOST_CHECK(equal(tags.begin(), tags.end(), aug.tags.begin()));
}


BOOST_FIXTURE_TEST_CASE( test_filters, AugmentationFixture )
{
    AugmentationList list;
    list[AccountKey()] = { {tag0}, data0 };
    list[accA] = { {tag1}, data1 };
    list[accBB] = { {tag2}, data2 };
    list[accBC];
    list[accBBA] = { {tag1}, data1 };

    check(list, {}, { tag0 }, { data0 });
    check(list, accA, { tag0, tag1 }, { data0, data1 });
    check(list, accBB, { tag0, tag2 }, { data0, data2 });
    check(list, accBC, { tag0 }, { data0 });
    check(list, accBBA, { tag0, tag1, tag2 }, { data0, data1, data2 });

    AccountKey acc0({ "A", "B" });
    check(list, acc0, { tag0, tag1 }, { data0, data1 });

    AccountKey acc1({ "C" });
    check(list, acc1, { tag0 }, { data0 });
}


BOOST_FIXTURE_TEST_CASE( test_json, AugmentationFixture )
{
    AugmentationList origin;
    origin[AccountKey()] = { {tag0}, data0 };
    origin[accA] = { {tag1}, data1 };
    origin[accBB] = { {tag2}, data2 };
    origin[accBC];
    origin[accBBA] = { {tag1}, data1 };

    Json::Value json = origin.toJson();
    cerr << json.toString();

    AugmentationList list = AugmentationList::fromJson(json);
    check(list, {},     { tag0 },             { data0 });
    check(list, accA,   { tag0, tag1 },       { data0, data1 });
    check(list, accBB,  { tag0, tag2 },       { data0, data2 });
    check(list, accBC,  { tag0 },             { data0 });
    check(list, accBBA, { tag0, tag1, tag2 }, { data0, data1, data2 });

    AccountKey acc0({ "A", "B" });
    check(list, acc0, { tag0, tag1 }, { data0, data1 });

    AccountKey acc1({ "C" });
    check(list, acc1, { tag0 }, { data0 });
}


BOOST_FIXTURE_TEST_CASE( test_merge, AugmentationFixture )
{
    AugmentationList list;
    list[AccountKey()] = { { tag0 }, data0 };
    list[accA] = { { tag0 }, data0 };

    AugmentationList toMerge;
    toMerge[AccountKey()] = { { tag0, tag2 }, data1 };
    toMerge[accBB] = { { tag1 }, data2 };

    list.mergeWith(toMerge);
    check(list, {}, { tag0, tag2 }, { data0, data1 });

    {
        const Augmentation& aug = list[accA];
        checkTags(aug.tags, { tag0 });
        checkData(aug.data, { data0 });
    }

    {
        const Augmentation& aug = list[accBB];
        checkTags(aug.tags, { tag1 });
        checkData(aug.data, { data2 });
    }

}

