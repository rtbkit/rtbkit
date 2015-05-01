/* value_description_test.h                                        -*- C++ -*-
   Wolfgang Sourdeau, June 2013
   Copyright (c) 2013 Datacratic Inc.  All rights reserved.

   Test value_description mechanisms
*/


#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <sstream>
#include <string>
#include <boost/lexical_cast.hpp>
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

/* ensures that ids are always rendered as strings, notwithstanding their
 * internal type */
BOOST_AUTO_TEST_CASE( test_stringid_description )
{
    StringIdDescription desc;
    Id idBigDec;
    ostringstream outStr;
    StreamJsonPrintingContext jsonContext(outStr);
    string result;

    idBigDec.type = Id::Type::BIGDEC;
    idBigDec.val = 2;

    desc.printJsonTyped(&idBigDec, jsonContext);
    result = outStr.str();

    string expected = "\"2\"";
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

/* SubClass1, where function overloads are used */

struct SubClass1 : public std::string
{
    explicit SubClass1(const std::string & other)
        : std::string(other)
    {}
};

typedef map<SubClass1, string> SubClass1Dict;

inline SubClass1 stringToKey(const std::string & str, SubClass1 *) { return SubClass1(str); }
inline std::string keyToString(const SubClass1 & str) { return str; }


/* SubClass2, where template specialization is used */

struct SubClass2 : public std::string
{
    explicit SubClass2(const std::string & other)
        : std::string(other)
    {}
};

typedef map<SubClass2, string> SubClass2Dict;

template<typename T>
struct FreeFunctionKeyCodec<SubClass2, T>
{
    static SubClass2 decode(const std::string & s, SubClass2 *) { return SubClass2(s); }
    static std::string encode(const SubClass2 & t) { return t; }
};

/* CompatClass, a class convertible from/to std::string */
struct CompatClass
{
    CompatClass()
    {}

    CompatClass(const std::string & value)
        : value_(value)
    {}

    std::string value_;

    operator std::string() const
    { return value_; }

    bool operator < (const CompatClass & other)
        const
    { return value_ < other.value_; }
};

typedef map<CompatClass, string> CompatClassDict;

string to_string(const CompatClass & k)
{ return string(k); }

}

namespace boost {

template<>
CompatClass
lexical_cast<CompatClass>(const string & s)
{
    return CompatClass(s);
}

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
        SubClass1Dict dict;

        dict = jsonDecodeStr(data, &dict);
        BOOST_CHECK_EQUAL(dict[SubClass1("key1")], string("value"));
        BOOST_CHECK_EQUAL(dict[SubClass1("key2")], string("value2"));
    }

    {
        SubClass2Dict dict;

        dict = jsonDecodeStr(data, &dict);
        BOOST_CHECK_EQUAL(dict[SubClass2("key1")], string("value"));
        BOOST_CHECK_EQUAL(dict[SubClass2("key2")], string("value2"));
    }

    {
        CompatClassDict dict;

        dict = jsonDecodeStr(data, &dict);

        string value1 = dict[CompatClass("key1")];
        BOOST_CHECK_EQUAL(value1, string("value"));
        string value2 = dict[CompatClass("key2")];
        BOOST_CHECK_EQUAL(value2, string("value2"));
    }

}

enum SomeSize {
    SMALL,
    MEDIUM,
    LARGE
};
CREATE_ENUM_DESCRIPTION(SomeSize);
SomeSizeDescription::
SomeSizeDescription()
{
    addValue("SMALL", SomeSize::SMALL, "");
    addValue("MEDIUM", SomeSize::MEDIUM, "");
    addValue("LARGE", SomeSize::LARGE, "");
}

struct SomeTestStructure {
    Id someId;
    std::string someText;
    std::vector<std::string> someStringVector;
    SomeSize someSize;

    SomeTestStructure(Id id = Id(0), std::string text = "nothing") : someId(id), someText(text) {
    }

    bool operator==(SomeTestStructure const & other) const {
        return someId == other.someId && someText == other.someText;
    }

    friend std::ostream & operator<<(std::ostream & stream, SomeTestStructure const & data) {
        return stream << "id=" << data.someId << " text=" << data.someText;
    }
};

CREATE_STRUCTURE_DESCRIPTION(SomeTestStructure)

SomeTestStructureDescription::
SomeTestStructureDescription() {
    addField("someId", &SomeTestStructure::someId, "");
    addField("someText", &SomeTestStructure::someText, "");
    addField("someStringVector", &SomeTestStructure::someStringVector, "");
    addField("someSize", &SomeTestStructure::someSize, "");
}

BOOST_AUTO_TEST_CASE( test_structure_description )
{
    SomeTestStructure data(Id(42), "hello world");

    // write the thing
    using namespace Datacratic;
    ValueDescription * desc = getDefaultDescription(&data);
    std::stringstream stream;
    StreamJsonPrintingContext context(stream);
    desc->printJson(&data, context);

    // inline in some other thing
    std::string value = ML::format("{\"%s\":%s}", desc->typeName, stream.str());

    // parse it back
    SomeTestStructure result;
    ML::Parse_Context source("test", value.c_str(), value.size());
        expectJsonObject(source, [&](std::string key,
                                     ML::Parse_Context & context) {
            auto desc = ValueDescription::get(key);
            if(desc) {
                StreamingJsonParsingContext json(context);
                desc->parseJson(&result, json);
            }
        });

    BOOST_CHECK_EQUAL(result, data);

    std::shared_ptr<const ValueDescription> vd =
        ValueDescription::get("SomeTestStructure");
    BOOST_CHECK_EQUAL(vd->kind, ValueKind::STRUCTURE);

    ValueDescription::FieldDescription fd = vd->getField("someStringVector");
    BOOST_CHECK_EQUAL(fd.description->kind, ValueKind::ARRAY);

    const ValueDescription * subVdPtr = &(fd.description->contained());
    BOOST_CHECK_EQUAL(subVdPtr->kind, ValueKind::STRING);

    fd = vd->getField("someSize");
    BOOST_CHECK_EQUAL(fd.description->kind, ValueKind::ENUM);
    vector<string> keys = fd.description->getEnumKeys();
    BOOST_CHECK_EQUAL(keys.size(), 3);
    BOOST_CHECK_EQUAL(keys[0], "SMALL");
    BOOST_CHECK_EQUAL(keys[1], "MEDIUM");
    BOOST_CHECK_EQUAL(keys[2], "LARGE");
}

struct S1 {
    string val1;
};

struct S2 : S1 {
    string val2;
};

CREATE_STRUCTURE_DESCRIPTION(S1);
CREATE_STRUCTURE_DESCRIPTION(S2);

S1Description::S1Description()
{
    addField("val1", &S1::val1, "first value");
}

S2Description::S2Description()
{
    addParent<S1>(); // make sure we don't get "parent description is not a structure
    addField("val2", &S2::val2, "second value");
}

struct RecursiveStructure {
    std::map<std::string, std::shared_ptr<RecursiveStructure> > elements;
    std::vector<std::shared_ptr<RecursiveStructure> > vec;
    std::map<std::string, RecursiveStructure> directElements;
};

CREATE_STRUCTURE_DESCRIPTION(RecursiveStructure);

RecursiveStructureDescription::RecursiveStructureDescription()
{
    addField("elements", &RecursiveStructure::elements,
             "elements of structure");
    addField("vec", &RecursiveStructure::vec,
             "vector of elements of structure");
    addField("directElements", &RecursiveStructure::directElements,
             "direct map of elements");
}

BOOST_AUTO_TEST_CASE( test_recursive_description )
{
    RecursiveStructureDescription desc;

    RecursiveStructure s;
    s.elements["first"] = make_shared<RecursiveStructure>();
    s.elements["first"]->elements["first.element"] = make_shared<RecursiveStructure>();
    s.elements["first"]->elements["first.element2"];  // null

    s.elements["second"]; // null
    s.vec.push_back(make_shared<RecursiveStructure>());
    s.vec[0]->vec.push_back(make_shared<RecursiveStructure>());
    s.vec[0]->elements["third"] = nullptr;
    s.vec.push_back(nullptr);

    s.directElements["first"].directElements["second"].directElements["third"].vec.push_back(nullptr);

    Json::Value j = jsonEncode(s);

    cerr << j << endl;

    RecursiveStructure s2 = jsonDecode<RecursiveStructure>(j);

    Json::Value j2 = jsonEncode(s2);

    BOOST_CHECK_EQUAL(j, j2);
}

BOOST_AUTO_TEST_CASE( test_date_value_description )
{
    auto desc = DefaultDescription<Date>();

    Date d = Date::now().quantized(0.001);

    /* timezone is "Z" */
    string isoZ = d.printIso8601();
    BOOST_CHECK_EQUAL(jsonDecode<Date>(isoZ), d);

    /* timezone is "+00:00" */
    string iso00 = isoZ;
    iso00.resize(iso00.size() - 1);
    iso00.append("+00:00");
    BOOST_CHECK_EQUAL(jsonDecode<Date>(iso00), d);

    /* normal (2014-May-02 14:33:02) */
    Date normal = d.quantized(1);
    string normalStr = normal.print();
    BOOST_CHECK_EQUAL(jsonDecode<Date>(normalStr), normal);
}


/* ensure that struct description invoke struct validators and child
   validators */

int numParentValidations(0);
int numChildValidations(0);

BOOST_AUTO_TEST_CASE( test_date_value_description_validation )
{
    /* test structs */
    struct ParentStruct
    {
        string value;
    };
    struct ChildStruct : public ParentStruct
    {
        int otherValue;
    };

    /* value descriptions */

    struct ParentStructVD
        : public StructureDescriptionImpl<ParentStruct, ParentStructVD>
    {
        ParentStructVD()
        {
            addField("value", &ParentStruct::value, "");
            onPostValidate = [&] (ParentStruct * value,
                                  JsonParsingContext & context) {
                numParentValidations++;
            };
        }
    };
    
    struct ChildStructVD
        : public StructureDescriptionImpl<ChildStruct, ChildStructVD>
    {
        ChildStructVD()
        {
            addParent(new ParentStructVD());
            addField("other-value", &ChildStruct::otherValue, "");
            onPostValidate = [&] (ChildStruct * value,
                                  JsonParsingContext & context) {
                numChildValidations++;
            };
        }
    };

    string testJson("{ \"value\": \"a string value\","
                    "  \"other-value\": 5}");
    ChildStruct testStruct;

    ChildStructVD desc;
    StreamingJsonParsingContext context(testJson,
                                        testJson.c_str(),
                                        testJson.c_str()
                                        + testJson.size());
    desc.parseJson(&testStruct, context);

    BOOST_CHECK_EQUAL(numChildValidations, 1);
    BOOST_CHECK_EQUAL(numParentValidations, 1);
}
