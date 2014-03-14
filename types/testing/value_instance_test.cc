/** value_instance_test.cc                                 -*- C++ -*-
    Rémi Attab, 13 Mar 2014
    Copyright (c) 2014 Datacratic.  All rights reserved.

    Tests for the value instance class.

*/

#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include "soa/types/basic_value_descriptions.h"
#include "soa/types/value_instance.h"
#include "jml/arch/exception_handler.h"

#include <boost/test/unit_test.hpp>

using namespace std;
using namespace Datacratic;


/******************************************************************************/
/* FOO                                                                        */
/******************************************************************************/

struct Foo
{
    Foo(size_t value = 0) : value(value) {}

    Foo& operator=(const Foo& other)
    {
        value = other.value;
        return *this;
    }

    bool operator==(const Foo& other) const
    {
        return value == other.value;
    }

    size_t value;
};

std::ostream& operator<<(std::ostream& stream, const Foo& value)
{
    stream << "Foo<" << value.value << ">";
    return stream;
}

CREATE_STRUCTURE_DESCRIPTION(Foo);

FooDescription::
FooDescription()
{
    addField("value", &Foo::value, "Value");
}


/******************************************************************************/
/* BAR                                                                        */
/******************************************************************************/

struct Bar : public Foo
{
    Bar(size_t value = 0, size_t count = 0) :
        Foo(value), count(count)
    {}

    Bar& operator=(const Bar& other)
    {
        Foo::operator=(other);
        count = other.count;
        return *this;
    }

    bool operator==(const Bar& other)
    {
        return Foo::operator==(other) && count == other.count;
    }

    size_t count;
};


std::ostream& operator<<(std::ostream& stream, const Bar& value)
{
    stream << "Bar<" << value.count << "," << value.value <<  ">";
    return stream;
}

CREATE_STRUCTURE_DESCRIPTION(Bar);

BarDescription::
BarDescription()
{
    addParent<Foo>();
    addField("count", &Bar::count, "Count");
}


/******************************************************************************/
/* TESTS                                                                      */
/******************************************************************************/

template<typename T>
void checkBasics(T defValue, T newValue)
{

    T value = defValue;
    ValueInstance instance(&value);

    BOOST_CHECK_EQUAL(*instance.cast<T>(), defValue);

    T other = newValue;
    instance.set(ValueInstance(&other));

    BOOST_CHECK_EQUAL(*instance.cast<T>(), newValue);
}

BOOST_AUTO_TEST_CASE( test_basics )
{
    checkBasics<size_t>(0xDEADBEEF, 0x77777777);
    checkBasics<double>(0.1, 0.9);
    checkBasics<Foo>(1, 10);
}


BOOST_AUTO_TEST_CASE( test_objecs )
{
    Bar bar(1, 2);
    ValueInstance instance(&bar);

    *instance["value"].cast<size_t>() = 10;
    *instance["count"].cast<size_t>() = 20;

    BOOST_CHECK_EQUAL(bar.value, 10);
    BOOST_CHECK_EQUAL(bar.count, 20);

    instance.cast<Foo>()->value = 30;
    BOOST_CHECK_EQUAL(bar.value, 30);

    instance["value"].set(instance["count"]);
    BOOST_CHECK_EQUAL(bar.value, bar.count);
}

BOOST_AUTO_TEST_CASE( test_shared_pointers )
{
    std::shared_ptr<Foo> foo;
    ValueInstance instance(&foo);

    cerr << "[ shared to shared ]=====================================" << endl;
    auto bar = std::make_shared<Bar>(1, 2);
    instance.set(ValueInstance(&bar));
    BOOST_CHECK_EQUAL(foo->value, 1);

    cerr << "[ raw to shared ]========================================" << endl;
    Bar* raw = new Bar(10);
    instance.set(ValueInstance(&raw));
    BOOST_CHECK_EQUAL(foo->value, 10);

    cerr << "[ null to shared ]=======================================" << endl;
    Bar* nil = nullptr;
    instance.set(ValueInstance(&nil));
    BOOST_CHECK(!foo);

    {
        ML::Set_Trace_Exceptions guard(false);

        cerr << "[ uniq to shared ]=======================================" << endl;
        std::unique_ptr<Foo> uniq(new Bar);
        BOOST_CHECK_THROW(instance.set(ValueInstance(&uniq)), ML::Exception);
    }
}

BOOST_AUTO_TEST_CASE( test_unique_pointers )
{
    std::unique_ptr<Foo> foo;
    ValueInstance instance(&foo);

    cerr << "[ raw to uniq ]==========================================" << endl;
    Bar* raw = new Bar(10, 20);
    instance.set(ValueInstance(&raw));
    BOOST_CHECK_EQUAL(foo->value, 10);

    cerr << "[ null to uniq ]=========================================" << endl;
    Bar* nil = nullptr;
    instance.set(ValueInstance(&nil));
    BOOST_CHECK(!foo);

    {
        ML::Set_Trace_Exceptions guard(false);

        cerr << "[ uniq to uniq ]=========================================" << endl;
        std::unique_ptr<Bar> uniq(new Bar(1, 2));
        BOOST_CHECK_THROW(instance.set(ValueInstance(&uniq)), ML::Exception);

        cerr << "[ shared to uniq ]=======================================" << endl;
        auto shared = std::make_shared<Bar>(1, 2);
        BOOST_CHECK_THROW(instance.set(ValueInstance(&shared)), ML::Exception);
    }
}


BOOST_AUTO_TEST_CASE( test_raw_pointers )
{
    Foo* foo = nullptr;
    ValueInstance instance(&foo);

    cerr << "[ raw to raw ]===========================================" << endl;
    Bar* raw = new Bar(10, 20);
    instance.set(ValueInstance(&raw));
    BOOST_CHECK(foo);
    BOOST_CHECK_EQUAL(foo->value, 10);

    cerr << "[ null to raw ]==========================================" << endl;
    Bar* nil = nullptr;
    instance.set(ValueInstance(&nil));
    BOOST_CHECK(!foo);

    cerr << "[ shared to raw ]========================================" << endl;
    auto shared = std::make_shared<Bar>(30, 40);
    instance.set(ValueInstance(&shared));
    BOOST_CHECK(foo);
    BOOST_CHECK_EQUAL(foo->value, 30);

    cerr << "[ uniq to raw ]==========================================" << endl;
    std::unique_ptr<Bar> uniq(new Bar(50, 60));
    instance.set(ValueInstance(&uniq));
    BOOST_CHECK(foo);
    BOOST_CHECK_EQUAL(foo->value, 50);

}
