/* string.h                                                          -*- C++ -*-
   Sunil Rottoo, 27 April 2012
   Copyright (c) 20102 Datacratic.  All rights reserved.

   Basic classes for dealing with string including internationalisation
*/

#pragma once

#include <string>
#include "soa/utf8cpp/source/utf8.h"
#include "jml/db/persistent_fwd.h"


namespace Datacratic {


/*****************************************************************************/
/* Utf8String                                                               */
/*****************************************************************************/

namespace JS {
struct JSValue;
} // namespace JS


class Utf8String
{
public:
	/**
	 * Take possession of a utf8-encoded string.
	 * @param input A string that contains utf8-encoded characters
	 * @param check If true we will make sure that the string contains valid
	 * utf-8 characters and will throw an exception if invalid characters are found
	 */
	explicit Utf8String(const std::string &in, bool check=true) ;

	typedef utf8::iterator<std::string::const_iterator> const_iterator;
	typedef utf8::iterator<std::string::iterator> iterator;

    const_iterator begin() const;
    const_iterator end() const ;

    Utf8String operator=(const std::string &str)
    {
    	data_ = str;
    	return *this;
    }
    Utf8String&  operator+=(const std::string& str)
	{
    	data_+=str;
    	return *this;
	}
    Utf8String &operator+=(const Utf8String &utf8str);
    /*
     * Returns access to the underlying representation - unsafe
     */
    const char * rawData() const { return data_.c_str(); }
    size_t rawLength() const { return data_.length() ; }

    void serialize(ML::DB::Store_Writer & store) const;
    void reconstitute(ML::DB::Store_Reader & store);

    std::string extractAscii();

private:
	std::string data_; // original utf8-encoded string
};

std::ostream & operator << (std::ostream & stream, const Utf8String & str);

IMPL_SERIALIZE_RECONSTITUTE(Utf8String);

#if 0
namespace JS {

void to_js(JSValue & jsval, const Utf8String & value);
Utf8String from_js(const JSValue & val, Utf8String *);

} // namespace JS
#endif
} // namespace Datacratic

