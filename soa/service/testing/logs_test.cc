/* logs_test.cc                                 -*- C++ -*-
   Rémi Attab (remi.attab@gmail.com), 02 May 2014
   FreeBSD-style copyright and disclaimer apply

   Tests for log
*/

#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include "soa/service/logs.h"

#include <boost/test/unit_test.hpp>

using namespace std;
using namespace Datacratic;

extern Logging::Category a;
static Logging::Category c("c", "a");
Logging::Category a("a");
static Logging::Category b("b", "a");

BOOST_AUTO_TEST_CASE(blah)
{
    Logging::Category d("d", "a");
    Logging::Category e("e", "d");
    Logging::Category f("f", "d", false /* enabled */);

    BOOST_CHECK(a.isEnabled());
    BOOST_CHECK(b.isEnabled());
    BOOST_CHECK(c.isEnabled());
    BOOST_CHECK(d.isEnabled());
    BOOST_CHECK(e.isEnabled());
    BOOST_CHECK(!f.isEnabled());

    BOOST_CHECK(!!a.getWriter());
    BOOST_CHECK(!!b.getWriter());
    BOOST_CHECK(!!c.getWriter());
    BOOST_CHECK(!!d.getWriter());
    BOOST_CHECK(!!e.getWriter());

    Logging::Category::root().deactivate();
    BOOST_CHECK(!a.isEnabled());
    BOOST_CHECK(!b.isEnabled());
    BOOST_CHECK(!c.isEnabled());
    BOOST_CHECK(!d.isEnabled());
    BOOST_CHECK(!e.isEnabled());
}
