/* http_header.h                                                 -*- C++ -*-
   Jeremy Barnes, 18 February 2011
   Copyright (c) 2011 Datacratic.  All rights reserved.

   http header parsing class.
*/

#pragma once

#include <string>
#include <map>
#include <iostream>
#include <vector>
#include "jml/arch/exception.h"


namespace Datacratic {

/*****************************************************************************/
/* REST PARAMS                                                               */
/*****************************************************************************/

struct RestParams
    : public std::vector<std::pair<std::string, std::string> > {
    RestParams()
    {
    }

    RestParams(std::initializer_list<std::pair<std::string, std::string> > l)
        : std::vector<std::pair<std::string, std::string> >(l.begin(), l.end())
    {
    }

    bool hasValue(const std::string & key) const;

    /** Return the value of the given key.  Throws an exception if it's not
        found.
    */
    std::string getValue(const std::string & key) const;

    std::string uriEscaped() const;

    static RestParams fromBinary(const std::string & binary);
    std::string toBinary() const;
};


/*****************************************************************************/
/* HTTP HEADER                                                               */
/*****************************************************************************/

/** Header for an HTTP request.  Just a place to dump the data. */

struct HttpHeader {
    HttpHeader()
        : contentLength(-1), isChunked(false)
    {
    }

    void swap(HttpHeader & other);

    void parse(const std::string & headerAndData, bool checkBodyLength = true);

    std::string verb;       // GET, PUT, etc
    std::string resource;   // after the get
    std::string version;    // after the get

    int responseCode() const;  // for responses; parses it out of the "version" field

    RestParams queryParams; // Query parameters pulled out of the URL

    // These headers are automatically pulled out
    std::string contentType;
    int64_t contentLength;
    bool isChunked;

    // The rest of the headers are here
    std::map<std::string, std::string> headers;

    std::string getHeader(const std::string & key) const
    {
        auto it = headers.find(key);
        if (it == headers.end())
            throw ML::Exception("couldn't find header " + key);
        return it->second;
    }

    std::string tryGetHeader(const std::string & key) const
    {
        auto it = headers.find(key);
        if (it == headers.end())
            return "";
        return it->second;
    }

    // If some portion of the data is known, it's put in here
    std::string knownData;
};

std::ostream & operator << (std::ostream & stream, const HttpHeader & header);

/** Returns the reason phrase for the given code.
    See http://www.w3.org/Protocols/rfc2616/rfc2616-sec6.html
*/
std::string getResponseReasonPhrase(int code);

} // namespace Datacratic
