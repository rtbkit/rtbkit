/* info_test.cc
   Jeremy Barnes, 21 February 2007
   Copyright (c) 2007 Jeremy Barnes.  All rights reserved.

   Test for the info functions.
*/

#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include "jml/arch/rtti_utils.h"
#include "jml/arch/exception.h"
#include "jml/utils/vector_utils.h"
#include "jml/arch/exception_handler.h"

#include <boost/test/unit_test.hpp>
#include <boost/bind.hpp>
#include <iostream>
#include <dirent.h>
#include "jml/utils/guard.h"
#include <errno.h>
#include <sys/mman.h>



using namespace ML;
using namespace std;

using boost::unit_test::test_suite;

struct T0 {
    virtual ~T0()
    {
    }

    int i;
};

struct T1 {
    virtual ~T1()
    {
    }

    int i;
};

struct T2 : public T1 {

    int i;
};

struct T3 : public T2 {
    int i;
};

struct T4 : private T2 {
    int i;
};

struct T5 : virtual public T2 {
    int i;
};

struct T6 : virtual public T2 {
    int i;
};

struct T7 : public T5, public T6 {
    int i;
};

struct T8 : public T7 {
    int i;
};

struct T9 : virtual public T8 {
    int i;
};

struct T10 : public T9 {
    int i;
};

BOOST_AUTO_TEST_CASE( test_is_convertible )
{
    BOOST_CHECK(ML::is_convertible<int>(int()));
    BOOST_CHECK(!ML::is_convertible<float>(int()));

    BOOST_CHECK(ML::is_convertible<T0>(T0()));
    BOOST_CHECK(!ML::is_convertible<T1>(T0()));
    BOOST_CHECK(!ML::is_convertible<T0>(T1()));

    BOOST_CHECK(!ML::is_convertible<T2>(T1()));
    BOOST_CHECK(ML::is_convertible<T1>(T2()));

    BOOST_CHECK(!ML::is_convertible<T3>(T1()));
    BOOST_CHECK(!ML::is_convertible<T3>(T2()));
    BOOST_CHECK(ML::is_convertible<T1>(T3()));
    BOOST_CHECK(ML::is_convertible<T2>(T3()));

    // Private should block inheritance
    BOOST_CHECK(!ML::is_convertible<T4>(T1()));
    BOOST_CHECK(!ML::is_convertible<T4>(T2()));
    BOOST_CHECK(!ML::is_convertible<T1>(T4()));
    BOOST_CHECK(!ML::is_convertible<T2>(T4()));

    // Diamond
    BOOST_CHECK(ML::is_convertible<T5>(T7()));
    BOOST_CHECK(ML::is_convertible<T6>(T7()));
    BOOST_CHECK(ML::is_convertible<T2>(T7()));
    BOOST_CHECK(ML::is_convertible<T1>(T7()));
    BOOST_CHECK(!ML::is_convertible<T0>(T7()));

    BOOST_CHECK(ML::is_convertible<T2>(T5()));
    BOOST_CHECK(ML::is_convertible<T2>(T6()));
    BOOST_CHECK(ML::is_convertible<T1>(T5()));
    BOOST_CHECK(ML::is_convertible<T1>(T6()));

    BOOST_CHECK(ML::is_convertible<T2>(T7()));
    BOOST_CHECK(ML::is_convertible<T1>(T7()));

    BOOST_CHECK(ML::is_convertible<T8>(T8()));
    BOOST_CHECK(ML::is_convertible<T7>(T8()));
    BOOST_CHECK(ML::is_convertible<T6>(T8()));
    BOOST_CHECK(ML::is_convertible<T5>(T8()));
    BOOST_CHECK(ML::is_convertible<T2>(T8()));
    BOOST_CHECK(ML::is_convertible<T1>(T8()));

    BOOST_CHECK(ML::is_convertible<T8>(T9()));
    BOOST_CHECK(ML::is_convertible<T7>(T9()));
    BOOST_CHECK(ML::is_convertible<T6>(T9()));
    BOOST_CHECK(ML::is_convertible<T5>(T9()));
    BOOST_CHECK(ML::is_convertible<T2>(T9()));
    BOOST_CHECK(ML::is_convertible<T1>(T9()));

    {
        T2 obj;
        T1 * volatile p = &obj;

        BOOST_CHECK(ML::is_convertible<T2>(*p));
    }

    {
        T3 obj;
        T1 * volatile p = &obj;

        BOOST_CHECK_EQUAL(ML::is_convertible<T3>(*p), &obj);
        BOOST_CHECK_EQUAL(ML::is_convertible<T2>(*p), &obj);
    }

    {
        T7 obj;
        T1 * volatile p = &obj;

        cerr << "p = " << p << endl;
        cerr << "&obj = " << &obj << endl;

        BOOST_CHECK_EQUAL(ML::is_convertible<T3>(*p), dynamic_cast<T3 *>(p));
        BOOST_CHECK_EQUAL(ML::is_convertible<T5>(*p), dynamic_cast<T5 *>(p));
        BOOST_CHECK_EQUAL(ML::is_convertible<T6>(*p), dynamic_cast<T6 *>(p));
        BOOST_CHECK_EQUAL(ML::is_convertible<T7>(*p), dynamic_cast<T7 *>(p));
    }
}

