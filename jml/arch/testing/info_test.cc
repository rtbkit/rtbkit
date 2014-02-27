/* info_test.cc
   Jeremy Barnes, 21 February 2007
   Copyright (c) 2007 Jeremy Barnes.  All rights reserved.

   Test for the info functions.
*/

#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include "jml/utils/info.h"
#include "jml/utils/environment.h"
#include "jml/arch/exception.h"

#include <boost/test/unit_test.hpp>
#include <iostream>

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>


using namespace ML;
using namespace std;

using boost::unit_test::test_suite;

BOOST_AUTO_TEST_CASE( test1 )
{
    BOOST_CHECK_EQUAL(userid_to_username(0), "root");
    BOOST_CHECK_EQUAL(userid_to_username(getuid()),
                      Environment::instance()["USER"]);
    BOOST_CHECK(num_cpus() > 0 && num_cpus() < 1024);
    cerr << "num_cpus = " << num_cpus() << endl;
}

BOOST_AUTO_TEST_CASE( test_num_open_files )
{
    int base = num_open_files();

    BOOST_CHECK(base > 3);

    int fd = open("/dev/null", O_RDONLY);

    BOOST_CHECK_EQUAL(num_open_files(), base + 1);

    int fd2 = open("/dev/zero", O_RDONLY);

    BOOST_CHECK_EQUAL(num_open_files(), base + 2);

    close(fd2);

    BOOST_CHECK_EQUAL(num_open_files(), base + 1);

    close(fd);

    BOOST_CHECK_EQUAL(num_open_files(), base);
}

BOOST_AUTO_TEST_CASE( test_fd_to_filename )
{
    int fd = open("/dev/null", O_RDONLY);
    BOOST_CHECK_EQUAL(fd_to_filename(fd), "/dev/null");

    BOOST_CHECK_THROW(fd_to_filename(500), ML::Exception);
    BOOST_CHECK_THROW(fd_to_filename(-1), ML::Exception);
}

BOOST_AUTO_TEST_CASE( test_fqdn_hostname )
{
    auto host = hostname();
    auto fqdn = fqdn_hostname("18142");
    std::cerr << "host: " << host << std::endl;
    std::cerr << "fqdn: " << fqdn << std::endl;
}

