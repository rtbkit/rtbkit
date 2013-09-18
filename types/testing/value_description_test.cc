/* value_description_test.h                                        -*- C++ -*-
   Wolfgang Sourdeau, June 2013
   Copyright (c) 2013 Datacratic Inc.  All rights reserved.

   Test value_description mechanisms
*/


#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <sstream>
#include <string>
#include <boost/test/unit_test.hpp>

#include "jml/utils/file_functions.h"
#include "soa/types/basic_value_descriptions.h"

#include "soa/types/id.h"


using namespace std;
using namespace ML;
using namespace Datacratic;

/* ensures that signed integers < (1 << 32 - 1) are serialized as integers */
BOOST_AUTO_TEST_CASE( test_default_description_print_id_32 )
{
    DefaultDescription<Datacratic::Id> desc;
    Id idBigDec;
    ostringstream outStr;
    StreamJsonPrintingContext jsonContext(outStr);
    string result;

    idBigDec.type = Id::Type::BIGDEC;
    idBigDec.val1 = 0x7fffffff;
    idBigDec.val2 = 0;

    desc.printJsonTyped(&idBigDec, jsonContext);
    result = outStr.str();

    string expected = "2147483647";
    BOOST_CHECK_EQUAL(expected, result);
}

/* ensures that integers >= 1 << 32 are serialized as strings */
BOOST_AUTO_TEST_CASE( test_default_description_print_id_non_32 )
{
    DefaultDescription<Datacratic::Id> desc;
    Id idBigDec;
    ostringstream outStr;
    StreamJsonPrintingContext jsonContext(outStr);
    string result;

    idBigDec.type = Id::Type::BIGDEC;
    idBigDec.val1 = 0x8fffffff;
    idBigDec.val2 = 0;

    desc.printJsonTyped(&idBigDec, jsonContext);
    result = outStr.str();

    string expected = "\"2415919103\"";
    BOOST_CHECK_EQUAL(expected, result);
}

/* ensures that 64 bit integers are properly parsed as such */
BOOST_AUTO_TEST_CASE( test_default_description_parse_id_64 )
{
    string input = "81985529216486895";
    File_Read_Buffer buffer(input.c_str(), input.size());
    StreamingJsonParsingContext jsonContext(buffer);

    Id expected;
    expected.type = Id::Type::BIGDEC;
    expected.val1 = 0x0123456789abcdef;
    expected.val2 = 0;

    DefaultDescription<Datacratic::Id> desc;
    Id result;
    desc.parseJsonTyped(&result, jsonContext);

    BOOST_CHECK_EQUAL(expected, result);
}

/* ensures that string-encoded 64 bit integers are properly parsed as 64 bit
 * integers */
BOOST_AUTO_TEST_CASE( test_default_description_parse_id_64_str )
{
    string input = "\"81985529216486895\"";
    File_Read_Buffer buffer(input.c_str(), input.size());
    StreamingJsonParsingContext jsonContext(buffer);

    Id expected;
    expected.type = Id::Type::BIGDEC;
    expected.val1 = 0x0123456789abcdef;
    expected.val2 = 0;

    DefaultDescription<Datacratic::Id> desc;
    Id result;
    desc.parseJsonTyped(&result, jsonContext);

    BOOST_CHECK_EQUAL(expected, result);
}

/* ensures that 128 bit integers are properly serialized as strings */
BOOST_AUTO_TEST_CASE( test_default_description_print_id_128 )
{
    DefaultDescription<Datacratic::Id> desc;
    Id idBigDec;
    ostringstream outStr;
    StreamJsonPrintingContext jsonContext(outStr);
    string result;

    idBigDec.type = Id::Type::BIGDEC;
    idBigDec.val1 = 0x0123456789abcdef;
    idBigDec.val2 = 0x0011223344556677;

    desc.printJsonTyped(&idBigDec, jsonContext);
    result = outStr.str();

    /* we do not support 128-bit int output */
    string expected = "\"88962710306127693105141072481996271\"";
    BOOST_CHECK_EQUAL(expected, result);
}

/* ensures that string-encoded 128 bit integers are properly parsed as 128
 * bit integers */
BOOST_AUTO_TEST_CASE( test_default_description_parse_id_128_str )
{
    string input = "\"88962710306127693105141072481996271\"";
    File_Read_Buffer buffer(input.c_str(), input.size());
    StreamingJsonParsingContext jsonContext(buffer);

    Id expected;
    expected.type = Id::Type::BIGDEC;
    expected.val1 = 0x0123456789abcdef;
    expected.val2 = 0x0011223344556677;

    DefaultDescription<Datacratic::Id> desc;
    Id result;
    desc.parseJsonTyped(&result, jsonContext);

    BOOST_CHECK_EQUAL(expected, result);
}

namespace Datacratic {

typedef map<string, string> StringDict;

struct SubClass : public std::string
{
    explicit SubClass(const std::string & other)
        : std::string(other)
    {}
};

typedef map<SubClass, string> SubClassDict;

template<>
struct KeyConverter<SubClass>
{
    static SubClass stringToKey(const string & str)
    { return SubClass(str); }

    static string keyToString(const SubClass & k)
    { return k; }
};

}

BOOST_AUTO_TEST_CASE( test_value_description_map )
{
    string data("{ \"key1\": \"value\", \"key2\": \"value2\" }");

    {
        StringDict dict;

        // auto d = getDefaultDescription(&dict);
        // cerr << d << endl;

        dict = jsonDecodeStr(data, &dict);
        BOOST_CHECK_EQUAL(dict["key1"], string("value"));
        BOOST_CHECK_EQUAL(dict["key2"], string("value2"));
    }

    {
        SubClassDict dict;

        dict = jsonDecodeStr(data, &dict);
        BOOST_CHECK_EQUAL(dict[SubClass("key1")], string("value"));
        BOOST_CHECK_EQUAL(dict[SubClass("key2")], string("value2"));
    }
}
