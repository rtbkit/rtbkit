/* info_test.cc
   Jeremy Barnes, 21 February 2007
   Copyright (c) 2007 Jeremy Barnes.  All rights reserved.

   Test for the info functions.
*/

#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include "jml/arch/vm.h"
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

// Copied from utils/info.cc due to not being able to include utils lib
size_t num_open_files()
{
    DIR * dfd = opendir("/proc/self/fd");
    if (dfd == 0)
        throw Exception("num_open_files(): opendir(): "
                        + string(strerror(errno)));

    Call_Guard closedir_dfd(boost::bind(closedir, dfd));

    size_t result = 0;
    
    dirent entry;
    for (dirent * current = &entry;  current;  ++result) {
        int res = readdir_r(dfd, &entry, &current);
        if (res != 0)
            throw Exception("num_open_files(): readdir_r: "
                            + string(strerror(errno)));
    }

    return result;
}

void test_function()
{
    cerr << "hello" << endl;
}

BOOST_AUTO_TEST_CASE( test_page_info )
{
    BOOST_REQUIRE_EQUAL(sizeof(Pagemap_Entry), 8);

    vector<Page_Info> null_pi = page_info(0, 1);

    BOOST_CHECK_EQUAL(null_pi.size(), 1);
    BOOST_CHECK_EQUAL(null_pi[0].pfn, 0);

    BOOST_CHECK_EQUAL(null_pi, page_info((void *)1, 1));
    
    vector<Page_Info> stack_pi = page_info(&null_pi, 1);

    BOOST_CHECK_EQUAL(stack_pi.size(), 1);

    cerr << "null_pi  = " << null_pi[0] << endl;
    cerr << "stack_pi = " << stack_pi[0] << endl;

    vector<Page_Info> code_pi = page_info((void *)&test_function, 1);
    
    cerr << "code_pi  = " << code_pi[0] << endl;
}

BOOST_AUTO_TEST_CASE( test_pagemap_reader )
{
    int npages = 10;
    char * memory = (char *)mmap(0, page_size * npages, PROT_READ | PROT_WRITE,
                                 MAP_PRIVATE | MAP_ANON, -1, 0);
    size_t mresult = (size_t)memory;
    BOOST_CHECK(mresult != 0 && mresult != -1);

    cerr << "memory = " << (void *)memory << endl;

    int nfiles_before = num_open_files();
    
    {
        Pagemap_Reader reader(memory, page_size * npages);
        
        cerr << reader << endl;

        BOOST_CHECK_EQUAL(reader.num_pages(), npages);
        BOOST_CHECK_EQUAL(reader[(int)0].present, false);
        BOOST_CHECK_EQUAL(reader[9].present, false);
        
        {
            JML_TRACE_EXCEPTIONS(false);
            BOOST_CHECK_THROW(reader[10], ML::Exception);
            BOOST_CHECK_THROW(reader[memory - 1], ML::Exception);
            BOOST_CHECK_THROW(reader[memory + npages * page_size], ML::Exception);
        }
        
        BOOST_CHECK_EQUAL(reader[memory].present, false);
        BOOST_CHECK_EQUAL(reader[memory + npages * page_size - 1].present, false);
        
        memory[0] = 'a';
        
        BOOST_CHECK_EQUAL(memory[0], 'a');
        
        BOOST_CHECK_EQUAL(reader[(int)0].present, false);
        
        BOOST_CHECK_EQUAL(reader.update(), 1);

        cerr << reader << endl;

        BOOST_CHECK(reader[memory].present);
        BOOST_CHECK(reader[memory + page_size].present == false);
        BOOST_CHECK_EQUAL(reader[memory], reader[(int)0]);

        BOOST_CHECK_EQUAL(memory[page_size], 0);
        BOOST_CHECK_EQUAL(memory[page_size * 2], 0);

        BOOST_CHECK_EQUAL(reader.update(memory + page_size * 2 - 1,
                                        memory + page_size * 2), 1);
        BOOST_CHECK_EQUAL(reader.update(memory + page_size * 2,
                                        memory + page_size * 2), 0);
        BOOST_CHECK_EQUAL(reader.update(memory + page_size * 2,
                                        memory + page_size * 2 + 1), 1);

        BOOST_CHECK_EQUAL(memory[page_size * 5], 0);
        BOOST_CHECK_EQUAL(memory[page_size * 6], 0);

        BOOST_CHECK_EQUAL(reader.update(memory + page_size * 6 - 1,
                                        memory + page_size * 6 + 1), 2);

        // Mapped zeroed pages may be shared by recent versions of the kernel;
        // here we look for that
        bool shared_zeroed_pages = reader[5].pfn == reader[6].pfn;

        cerr << reader << endl;

        BOOST_CHECK(reader[1].present);

        if (shared_zeroed_pages)
            BOOST_CHECK(reader[1] == reader[2]);
        else BOOST_CHECK(reader[1] != reader[2]);

        memory[page_size * 2] = 'x';

        if (shared_zeroed_pages)
            BOOST_CHECK_EQUAL(reader.update(memory + page_size * 2, 1), 1);
        else BOOST_CHECK_EQUAL(reader.update(memory + page_size * 2, 1), 0);
        
        cerr << reader << endl;

        BOOST_CHECK(reader[2].present);
        BOOST_CHECK(reader[1] != reader[2]);
    }

    BOOST_CHECK_EQUAL(num_open_files(), nfiles_before);
}
