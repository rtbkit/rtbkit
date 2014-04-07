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
    static Utf8String fromLatin1(const std::string & lat1Str);

    /** Allow default construction of an empty string. */
    Utf8String()
    {
    }

    /** Move constructor. */
    Utf8String(Utf8String && str) noexcept
        : data_(std::move(str.data_))
    {
    }

    /** Copy constructor. */
    Utf8String(const Utf8String & str)
        : data_(str.data_)
    {
    }

    /**
     * Take possession of a utf8-encoded string.
     * @param input A string that contains utf8-encoded characters
     * @param check If true we will make sure that the string contains valid
     * utf-8 characters and will throw an exception if invalid characters are found
     */
    explicit Utf8String(const std::string &in, bool check=true) ;

    explicit Utf8String(std::string &&in, bool check=true) ;

    Utf8String & operator=(Utf8String && str) noexcept
    {
        Utf8String newMe(std::move(str));
        swap(newMe);
        return *this;
    }

    Utf8String & operator=(const Utf8String & str)
    {
        Utf8String newMe(str);
        swap(newMe);
        return *this;
    }

    Utf8String & operator=(const std::string &str)
    {
    	data_ = str;
    	return *this;
    }

    Utf8String & operator=(std::string &&str)
    {
    	data_ = std::move(str);
    	return *this;
    }

    void swap(Utf8String & other)
    {
        data_.swap(other.data_);
    }

    bool empty() const
    {
        return data_.empty();
    }

    typedef utf8::iterator<std::string::const_iterator> const_iterator;
    typedef utf8::iterator<std::string::iterator> iterator;

    const_iterator begin() const;
    const_iterator end() const ;

    Utf8String&  operator+=(const std::string& str)
    {
    	data_+=str;
    	return *this;
    }
    Utf8String &operator+=(const Utf8String &utf8str);
    /*
     * Returns access to the underlying representation - unsafe
     */
    const std::string & rawString() const { return data_; }
    const std::string & utf8String() const { return data_; }
    const char * rawData() const { return data_.c_str(); }
    size_t rawLength() const { return data_.length() ; }

    void serialize(ML::DB::Store_Writer & store) const;
    void reconstitute(ML::DB::Store_Reader & store);

    std::string extractAscii() const;

    bool operator == (const Utf8String & other) const
    {
        return data_ == other.data_;
    }

    bool operator != (const Utf8String & other) const
    {
        return data_ != other.data_;
    }

    bool operator < (const Utf8String & other) const
    {
        return data_ < other.data_;
    }

private:
    std::string data_; // original utf8-encoded string
};

inline void swap(Utf8String & s1, Utf8String & s2)
{
    s1.swap(s2);
}

std::ostream & operator << (std::ostream & stream, const Utf8String & str);

IMPL_SERIALIZE_RECONSTITUTE(Utf8String);

#if 0
namespace JS {

void to_js(JSValue & jsval, const Utf8String & value);
Utf8String from_js(const JSValue & val, Utf8String *);

} // namespace JS
#endif

class Utf32String {
public:
    typedef std::u32string::iterator iterator;
    typedef std::u32string::const_iterator const_iterator;

    static Utf32String fromLatin1(const std::string &str);
    static Utf32String fromUtf8(const Utf8String &utf8Str);

    Utf32String()
    {
    }

    explicit Utf32String(const std::string &str) {
        utf8::utf8to32(std::begin(str), std::end(str), std::back_inserter(data_));
    }

    Utf32String(const Utf32String &other)
    : data_(other.data_)
    {
    }
    Utf32String(Utf32String &&other) noexcept
    {
        *this = std::move(other);
    }

    Utf32String &operator=(const Utf32String &other)
    {
        Utf32String newMe(other);
        swap(newMe);
        return *this;
    }

    Utf32String &operator=(Utf32String &&other) noexcept
    {
        data_ = std::move(other.data_);
        return *this;
    }

    Utf32String &operator=(const std::string &other) {
        utf8::utf8to32(std::begin(other), std::end(other), std::back_inserter(data_));
        return *this;
    }

    void swap(Utf32String &other)
    {
        std::swap(data_, other.data_);
    }

    bool empty() const {
        return data_.empty();
    }

    iterator begin() {
        return data_.begin();
    }

    const_iterator begin() const {
        return data_.begin();
    }

    iterator end() {
        return data_.end();
    }

    const_iterator end() const {
        return data_.end();
    }

    std::u32string rawString() const {
        return data_;
    }

    std::string utf8String() const {
        std::string utf8Str;
        utf8::utf32to8(std::begin(data_), std::end(data_), std::back_inserter(utf8Str));
        return utf8Str;
    }

    const char32_t *rawData() const {
        return data_.c_str();
    }

    const size_t rawLength() const {
        return data_.length();
    }

    void serialize(ML::DB::Store_Writer & store) const;
    void reconstitute(ML::DB::Store_Reader & store);

    Utf32String &operator+=(const std::string &other) {
        std::u32string u32other;
        utf8::utf8to32(std::begin(other), std::end(other), std::back_inserter(u32other));
        data_ += u32other;

        return *this;
    }

    Utf32String &operator+=(const Utf32String &other) {
        data_ += other.data_;

        return *this;
    }

    bool operator==(const Utf32String &other) const {
        return data_ == other.data_;
    }

    bool operator!=(const Utf32String &other) const {
        return !operator==(other);
    }

    bool operator<(const Utf32String &other) const {
        return data_ < other.data_;
    }

    std::string extractAscii() const;


private:
    std::u32string data_;
};

Utf32String operator+(Utf32String lhs, const Utf32String &rhs);

std::ostream & operator << (std::ostream & stream, const Utf32String & str);

IMPL_SERIALIZE_RECONSTITUTE(Utf32String);

typedef Utf8String UnicodeString;

} // namespace Datacratic

