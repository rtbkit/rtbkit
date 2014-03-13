/* string_test.cc
   Copyright (c) 2012 Datacratic.  All rights reserved.
*/

#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include "soa/types/string.h"
#include <boost/test/unit_test.hpp>
#include <boost/regex/icu.hpp>
#include <boost/regex.hpp>
#include "soa/jsoncpp/json.h"
#include "jml/arch/format.h"

using namespace std;
using namespace ML;
using namespace Datacratic;


BOOST_AUTO_TEST_CASE( test_print_format )
{
   	std::string raw = "saint-jérôme";
   	// Test 1 - Iterate through the raw string with normal iterators we should not find 'é'
   	unsigned numAccentedChars = 0;
   	for(string::const_iterator it = raw.begin() ; it != raw.end(); ++it)
   	{
   		if (*it ==  L'é' || *it ==  L'ô')
   			numAccentedChars++;
   	}
   	BOOST_CHECK_EQUAL(numAccentedChars, 0);
   	Utf8String utf8(raw);
   	// Now iterate through the utf8 string
   	for (Utf8String::const_iterator it = utf8.begin(); it != utf8.end(); ++it)
   	{
   		if (*it ==  L'é' || *it ==  L'ô')
   			numAccentedChars++;
   	}
   	BOOST_CHECK_EQUAL(numAccentedChars, 2);
   	// Now add another string to it
  	std::string raw2 = "saint-jérôme2";
  	utf8+=raw2;
  	numAccentedChars=0;
  	// Now iterate through the utf8 string
   	for (Utf8String::const_iterator it = utf8.begin(); it != utf8.end(); ++it)
   	{
   		if (*it ==  L'é' || *it ==  L'ô')
   			numAccentedChars++;
   	}
   	BOOST_CHECK_EQUAL(numAccentedChars, 4);
   	string theString(utf8.rawData(), utf8.rawLength());
   	size_t found = raw.find(L'é') ;
   	BOOST_CHECK_EQUAL(found, string::npos);
   	// We do a normal regex first
   	boost::regex reg("é");
   	std::string raw4 = "saint-jérôme";
   	BOOST_CHECK_EQUAL( boost::regex_search(raw4, reg), true);
   	// Please see Saint-j\xC3A9r\xC3B4me for UTF-8 character table
   	boost::u32regex withHex = boost::make_u32regex("saint-j\xc3\xa9r\xc3\xb4me");
   	boost::u32regex withoutHex = boost::make_u32regex(L"[a-z]*-jérôme");
    boost::match_results<std::string::const_iterator> matches;
    BOOST_CHECK_EQUAL(boost::u32regex_search(raw, matches, withoutHex), true);
    if (boost::u32regex_search(raw, matches, withoutHex))
    {
    	for (boost::match_results< std::string::const_iterator >::const_iterator i = matches.begin(); i != matches.end(); ++i)
    	{
    	        if (i->matched) std::cout << "matches :       [" << i->str() << "]\n";
    	        else            std::cout << "doesn't match : [" << i->str() << "]\n";
    	}
    }
    else
    {
    	cerr << "did not get a match without hex" << endl;
    }
    BOOST_CHECK_EQUAL(boost::u32regex_search(raw, matches, withHex), true);
}
