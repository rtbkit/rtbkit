/* configuration.h                                                 -*- C++ -*-
   Jeremy Barnes, 3 April 2006
   Copyright (c) 2006 Jeremy Barnes.  All rights reserved.

   This file is part of "Jeremy's Machine Learning Library", copyright (c)
   1999-2006 Jeremy Barnes.
   
   This program is available under the GNU General Public License, the terms
   of which are given by the file "license.txt" in the top level directory of
   the source code distribution.  If this file is missing, you have no right
   to use the program; please contact the author.

   This program is distributed in the hope that it will be useful, but
   WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
   or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
   for more details.

   ---

   Functions to allow key-based configuration files.
*/

#ifndef __utils__configuration_h__
#define __utils__configuration_h__

#include <map>
#include <vector>
#include <string>
#include <boost/shared_ptr.hpp>
#include <boost/lexical_cast.hpp>
#include "jml/arch/exception.h"
#include <iostream>
#include "jml/utils/string_functions.h"
#include "enum_info.h"


namespace ML {

struct Parse_Context;

/*****************************************************************************/
/* CONFIGURATION                                                             */
/*****************************************************************************/

class Configuration {
public:
    enum Prefix_Op {
        PREFIX_APPEND,   ///< Append the given prefix to the current
        PREFIX_REPLACE   ///< Replace the current prefix with the given
    };

    Configuration();

    Configuration(const Configuration & other,
                  const std::string & prefix,
                  Prefix_Op op);

    struct Accessor {
        Accessor(Configuration * config, const std::string & key)
            : config(config), key(key)
        {
        }

        Configuration * config;
        std::string key;

        Accessor & operator = (const std::string & t)
        {
            config->raw_set(key, t);
            return *this;
        }

        operator std::string() const
        {
            return config->raw_get(key);
        }

        bool operator == (const std::string & other) const
        {
            return config->raw_get(key) == other;
        }

        bool operator != (const std::string & other) const
        {
            return config->raw_get(key) != other;
        }
    };

    Accessor operator [] (const std::string & key);

    std::string operator [] (const std::string & key) const;

    bool count(const std::string & key, bool search_parents = false) const;

    std::string find_key(const std::string & key,
                         bool search_parents = false) const;

    std::string prefix() const { return prefix_; }

    void load(const std::string & filename);
    void parse_string(const std::string & str, const std::string & filename);
    void parse_command_line(const std::vector<std::string> & options);
    void parse(Parse_Context & context);

    void parse_value(std::string & val, const std::string & str,
                     const std::string & full_key) const
    {
        val = str;
    }

    void exception_for_key(const std::string & full_key,
                           const std::string & message) const;

    std::vector<std::string> allKeys() const;

    void parse_value(bool & val, const std::string & str,
                     const std::string & full_key) const
    {
        if (str == "0") { val = false; return; }
        if (str == "1") { val = true; return; }
        std::string strl = lowercase(str);
        if (strl == "true") { val = true; return; }
        if (strl == "false") { val = false; return; }
        throw Exception("couldn't parse " + str + " as bool under key "
                        + full_key);
    }

    void parse_value(std::vector<std::string> & val, const std::string & str,
                     const std::string & full_key, Not_Enum_Tag) const
    {
        //bool in_quote = false, after_backslash = false;
        //..
    }

    template<class X>
    void parse_value(std::vector<X> & val, const std::string & str,
                     const std::string & full_key, Not_Enum_Tag) const
    {
        std::vector<std::string> vals = split(str, ',');
        val.resize(vals.size());
        for (unsigned i = 0;  i < vals.size();  ++i)
            parse_value(val[i], vals[i], full_key);
    }

    template<class X>
    void parse_value(X & val, const std::string & str,
                     const std::string & full_key, Not_Enum_Tag) const
    {
        try {
            val = boost::lexical_cast<X>(str);
        }
        catch (const std::exception & exc) {
            throw Exception("couldn't parse value " + str + " from key "
                            + full_key + ": " + exc.what());
        }
    }

    template<class X>
    void parse_value(X & val, const std::string & str,
                     const std::string & full_key, Enum_Tag) const
    {
        try {
            val = enum_value<X>(str);
        }
        catch (const std::exception & exc) {
            throw Exception("parsing value from key " + full_key + ": "
                            + exc.what());
        }
    }

    template<class X>
    void parse_value(X & val, const std::string & str,
                     const std::string & full_key) const
    {
        return parse_value(val, str, full_key, typename Enum_Info<X>::Tag());
    }

    template<class X>
    bool get(X & val, const std::string & key) const
    {
        std::string full_key = find_key(key, false);
        if (!raw_count(full_key)) return false;
        std::string value = raw_get(full_key);
        parse_value(val, value, full_key);
        return true;
    }

    template<class X>
    bool find(X & val, const std::string & key) const
    {
        //std::cerr << "find: key = " << key << std::endl;
        std::string full_key = find_key(key, true);
        if (!raw_count(full_key)) return false;
        std::string value = raw_get(full_key);
        parse_value(val, value, full_key);
        return true;
    }

    template<class X>
    void require(X & val, const std::string & key) const
    {
        if (!get(val, key))
            throw Exception("required key " + key + " not found");
    }

    template<class X>
    void must_find(X & val, const std::string & key) const
    {
        if (!find(val, key))
            throw Exception("required key " + key + " not found");
    }

private:
    struct Data;
    std::shared_ptr<Data> data_;
    std::string prefix_;
    bool writeable_;

    static std::string
    add_prefix(const std::string & prefix, const std::string & rest);

    void raw_set(const std::string & key, const std::string & value);
    std::string raw_get(const std::string & key) const;
    bool raw_count(const std::string & key) const;
};

std::ostream & operator << (std::ostream & stream,
                            const Configuration::Accessor & acc);

} // namespace ML


#endif /* __utils__configuration_h__ */
