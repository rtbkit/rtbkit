/* enum_info.h                                                     -*- C++ -*-
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

   Functions to allow us to know the contents of enumerations.
*/

#ifndef __utils__enum_info_h__
#define __utils__enum_info_h__

#include "jml/boosting/config.h"
#include <string>
#include <map>
#include "jml/arch/atomic_init.h"


namespace ML {

struct Enum_Tag {};
struct Not_Enum_Tag {};

template<typename Enum>
struct Enum_Opt {
    const char * name;
    Enum val;
};

template<typename Enum>
struct Enum_Info {
    enum { IS_SPECIALIZED = false };
    typedef Not_Enum_Tag Tag;
};

template<typename Enum>
std::string
enum_values()
{
    std::string result;
    for (unsigned i = 0;  i < Enum_Info<Enum>::NUM;  ++i)
        result += std::string(i > 0, ' ') + Enum_Info<Enum>::OPT[i].name;
    return result;
}

template<typename Enum>
std::string
enum_value(Enum val)
{
    typedef std::map<Enum, std::string> values_type; 
    static values_type * values;
    if (!values) {
        values_type * new_values = new values_type;

        for (unsigned i = 0;  i < Enum_Info<Enum>::NUM;  ++i) {
            const Enum_Opt<Enum> & opt = Enum_Info<Enum>::OPT[i];
            if (!new_values->count(opt.val))
                (*new_values)[opt.val] = opt.name;
        }

        atomic_init(values, new_values);
    }
    typename values_type::const_iterator found = values->find(val);
    if (found == values->end())
        return "";
    return found->second;
}

template<typename Enum>
Enum
enum_value(const std::string & name)
{
    typedef std::map<std::string, Enum> values_type; 
    static values_type * values;
    if (!values) {
        values_type * new_values = new values_type;

        for (unsigned i = 0;  i < Enum_Info<Enum>::NUM;  ++i) {
            const Enum_Opt<Enum> & opt = Enum_Info<Enum>::OPT[i];
            if (!new_values->count(opt.name))
                (*new_values)[opt.name] = opt.val;
        }

        atomic_init(values, new_values);
    }
    typename values_type::const_iterator found = values->find(name);
    if (found == values->end())
        throw Exception("couldn't parse '" + name + "' as "
                        + Enum_Info<Enum>::NAME + " (possibilities are "
                        + enum_values<Enum>());
    return found->second;
}

#define DECLARE_ENUM_INFO(type, num_values) \
namespace ML { \
template<> \
struct Enum_Info<type> { \
    enum { NUM = num_values, IS_SPECIALIZED = true }; \
    typedef Enum_Tag Tag; \
    static const Enum_Opt<type> OPT[num_values]; \
    static const char * NAME; \
}; \
\
} // namespace ML

} // namespace ML

#endif /* __utils__enum_info_h__ */
