/* string.cc
   Sunil Rottoo, 27 April 2012
   Copyright (c) 2012 Datacratic.  All rights reserved.

*/

#include "string.h"
#include "soa/js/js_value.h"
#include "soa/jsoncpp/json.h"
#include <iostream>
#include "jml/arch/exception.h"
#include "jml/db/persistent.h"

using namespace std;


namespace Datacratic {


/*****************************************************************************/
/* DATE                                                                      */
/*****************************************************************************/

Utf8String::Utf8String(const string & in, bool check)
    : data_(in)
{
    if (check)
    {
        // Check if we find an invalid encoding
        string::const_iterator end_it = utf8::find_invalid(in.begin(), in.end());
        if (end_it != in.end())
        {
            throw ML::Exception("Invalid sequence within utf-8 string");
        }
    }
}

Utf8String::const_iterator
Utf8String::begin() const
{
    return Utf8String::const_iterator(data_.begin(), data_.begin(), data_.end()) ;
}

Utf8String::const_iterator
Utf8String::end() const
{
    return Utf8String::const_iterator(data_.end(), data_.begin(), data_.end()) ;
}

Utf8String &Utf8String::operator+=(const Utf8String &utf8str)
{
    data_ += utf8str.data_;
    return *this;
}

std::ostream & operator << (std::ostream & stream, const Utf8String & str)
{
    stream << string(str.rawData(), str.rawLength()) ;
    return stream;
}

void
Utf8String::
serialize(ML::DB::Store_Writer & store) const
{
    store << data_;
}

void
Utf8String::
reconstitute(ML::DB::Store_Reader & store)
{
    store >> data_;
}
    
string Utf8String::extractAscii()
{
    string s;
    for(auto it = begin(); it != end(); it++) {
        char c = *it;
        if (c >= ' ' && c < 127) {
            s += c;
        } else {
            s += '?';
        }
    }
    return s;
}

} // namespace Datacratic
