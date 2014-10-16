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
/* UTF8STRING                                                                */
/****************************************************************************/

Utf8String
Utf8String::fromLatin1(const std::string & lat1Str)
{
    size_t bufferSize = lat1Str.size();
    const char *inBuf = lat1Str.c_str();
    string utf8Str(bufferSize * 4, '.');

    auto iter = utf8Str.begin();
    auto start = iter;
    for (size_t i = 0; i < bufferSize; i++) {
        uint32_t cp(inBuf[i] & 0xff);
        iter = utf8::append(cp, iter);
    }
    utf8Str.resize(iter-start);

    return Utf8String(utf8Str);
}

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

Utf8String::Utf8String(string && in, bool check)
    : data_(std::move(in))
{
    if (check)
    {
        // Check if we find an invalid encoding
        string::const_iterator end_it = utf8::find_invalid(data_.begin(), data_.end());
        if (end_it != data_.end())
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
    
string Utf8String::extractAscii() const
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

/*****************************************************************************/
/* UTF32STRING                                                                */
/****************************************************************************/

Utf32String Utf32String::fromLatin1(const std::string &str) {
    std::u32string u32str;
    for (auto c: str) {
        u32str.push_back(static_cast<char32_t>(static_cast<uint8_t>(c)));
    }

    Utf32String ret;
    ret.data_ = u32str;
    return ret;
}

Utf32String Utf32String::fromUtf8(const Utf8String &str) {
    return Utf32String(str.rawString());
}

string Utf32String::extractAscii() const {
    string ascii;
    for (auto c: data_) {
        if ((c & 0x80) == 0)
            ascii += c;
        else
            ascii += '?';
    }

    return ascii;
}

Utf32String operator+(Utf32String lhs, const Utf32String &rhs) {
    return lhs += rhs;
}


std::ostream & operator << (std::ostream & stream, const Utf32String & str)
{
    return stream;
}

void
Utf32String::
serialize(ML::DB::Store_Writer & store) const
{
    std::string utf8Str;
    utf8::utf32to8(std::begin(data_), std::end(data_), std::back_inserter(utf8Str));
    store << utf8Str;
}

void
Utf32String::
reconstitute(ML::DB::Store_Reader & store)
{
    std::string utf8Str;
    store >> utf8Str;

    std::u32string utf32Str;
    utf8::utf8to32(std::begin(utf8Str), std::end(utf8Str), std::back_inserter(utf32Str));

    data_ = std::move(utf32Str);
}

} // namespace Datacratic
