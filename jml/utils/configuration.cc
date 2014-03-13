/* configuration.cc
   Jeremy Barnesm, 18 February 2007
   Copyright (c) 2007 Jeremy Barnes.  All rights reserved.

   Configuration file parser.
*/

#include "configuration.h"
#include "parse_context.h"
#include "jml/utils/string_functions.h"


using namespace std;


namespace ML {

/*****************************************************************************/
/* CONFIGURATION::DATA                                                       */
/*****************************************************************************/

struct Configuration::Data {
    typedef std::map<std::string, std::string> Entries;
    Entries entries;

    void copy_entries(const std::string & from,
                      const std::string & to)
    {
        //cerr << "** Copying entries from " << from << " to " << to << endl;

        if (to.find(from) == 0)
            throw Exception("Attempt to copy entries to within their tree");

        string from2 = from + ".";
        
        for (Entries::const_iterator it = entries.lower_bound(from);
             it != entries.end() && it->first.find(from2) == 0; ++it) {
            string rest(it->first, from2.size());
            string key = add_prefix(to, rest);
            entries[key] = it->second;

            //cerr << "copying " << it->first << "=" << it->second << " to "
            //     << key << endl;
        }
    }
};


/*****************************************************************************/
/* CONFIGURATION                                                             */
/*****************************************************************************/

Configuration::
Configuration()
    : data_(new Data()), prefix_(""), writeable_(true)
{
}

Configuration::
Configuration(const Configuration & other,
              const std::string & prefix,
              Prefix_Op op)
    : data_(other.data_), writeable_(false)
{
    if (op == PREFIX_REPLACE)
        prefix_ = prefix;
    else prefix_ = add_prefix(other.prefix(), prefix);

    //cerr << "created new config with prefix=" << prefix_ << " from "
    //     << other.prefix_ << endl;
}

Configuration::Accessor
Configuration::
operator [] (const std::string & key)
{
    string real_key = find_key(key, false /* find_parents */);
    return Accessor(this, real_key);
}

std::string
Configuration::
operator [] (const std::string & key) const
{
    string real_key = find_key(key, false /* find_parents */);
    return data_->entries[real_key];
}

bool
Configuration::
count(const std::string & key, bool search_parents) const
{
    string real_key = find_key(key, search_parents);
    return data_->entries.count(real_key);
}

namespace {

bool shorter_prefix(string & prefix)
{
    size_t pos = prefix.rfind('.');
    if (pos == string::npos) {
        if (prefix == "") return false;
        prefix = "";
        return true;
    }
    prefix = string(prefix, 0, pos);
    return true;
}

void lengthen(string & prefix, const std::string & name)
{
    if (prefix == "") prefix = name;
    else prefix += "." + name;
}

} // file scope

std::string
Configuration::
find_key(const std::string & key,
         bool search_parents) const
{
    //cerr << "find_key(" << key << ", " << search_parents << ")" << endl;

    /* If we search parents, then we look at succeedingly shorter prefixes
       until we find it. */

    string prefix = prefix_;

    do {
        string full = add_prefix(prefix, key);
        //cerr << " prefix = " << prefix << " full = " << full << endl;
        if (data_->entries.count(full)) {
            //cerr << "  --> found under " << full << endl;
            return full;
        }
    } while (search_parents && shorter_prefix(prefix));

    //cerr << "  --> not found" << endl;
    return add_prefix(prefix_, key);  // not found; return first location
}

void
Configuration::
load(const std::string & filename)
{
    Parse_Context context(filename);
    parse(context);
}

void
Configuration::
parse_string(const std::string & str, const std::string & filename)
{
    Parse_Context context(filename, str.c_str(), str.c_str() + str.size());
    parse(context);
}

namespace {

struct Found_End_Name {
    bool operator () (char c) const
    {
        if (isalnum(c) || c == '_' || c == '.') return false;
        return true;
    }
};

std::string expect_name(Parse_Context & context)
{
    std::string result;
    if (!context.match_text(result, Found_End_Name())
        || result.empty())
        context.exception("expected name of option");
    return result;
}

struct Found_End_Value {
    bool operator () (char c) const
    {
        return c == '\n' || c == ';' || c == ' ' || c == '\t';
    }
};

std::string expect_value(Parse_Context & context)
{
    std::string result;
    if (!context.match_text(result, Found_End_Value())
        || result.empty())
        context.exception("expected name of option");
    return result;
}

} // file scope

void
Configuration::
parse(Parse_Context & context)
{
    string scope;

    while (context) {
        //cerr << "scope = " << scope << endl;
        context.skip_whitespace();
        if (context.match_eol()) continue;  // blank line
        else if (context.match_literal('#')) {
            // comment
            context.skip_line();
            continue;
        }
        else if (context.match_literal('}')) {
            // end of scope
            //cerr << "end of scope = " << scope << endl;
            if (!shorter_prefix(scope))
                context.exception("configuration tried to close unopened "
                                  "scope");
            continue;
        }

        string name = expect_name(context);

        context.skip_whitespace();
        
        if (context.match_literal('=')) {
            context.skip_whitespace();
            string value = expect_value(context);

            context.match_literal(';');
            context.skip_whitespace();
            if (context.match_literal('#'))
                context.expect_text('\n', true, "expected comment");
            
            context.expect_eol();

            data_->entries[add_prefix(scope, name)] = value;
        }
        else if (context.match_literal(':')) {
            // start a new scope copying from another
            context.skip_whitespace();
            string copy_from = expect_name(context);
            context.skip_whitespace();
            context.expect_literal('{');
            lengthen(scope, name);

            data_->copy_entries(copy_from, scope);
        }
        else if (context.match_literal('{')) {
            // start a new scope
            lengthen(scope, name);
        }
        else context.exception("expected :, = or { after name");
    }

    if (!scope.empty()) {
        context.exception("configuration ended with scope " + scope
                          + " open");
    }
}

void
Configuration::
parse_command_line(const std::vector<std::string> & options)
{
    /* Each option should be in the form key=value.  We parse these simply. */
    if (!writeable_)
        throw Exception("Configuration::parse_command_line(): "
                        "object not writeable");

    for (unsigned i = 0;  i < options.size();  ++i) {
        parse_string(options[i],
                     format("command line:%d:\"%s\"", i, options[i].c_str()));
    }
}

std::string
Configuration::
add_prefix(const std::string & prefix, const std::string & rest)
{
    string result = prefix;
    lengthen(result, rest);
    return result;
}

void
Configuration::
raw_set(const std::string & key, const std::string & value)
{
    if (!writeable_)
        throw Exception("Configuration::operator []: "
                        "object is not writeable");

    data_->entries[key] = value;
}

std::string
Configuration::
raw_get(const std::string & key) const
{
    return data_->entries[key];
}

bool
Configuration::
raw_count(const std::string & key) const
{
    return data_->entries.count(key);
}

std::ostream &
operator << (std::ostream & stream, const Configuration::Accessor & acc)
{
    return stream << acc.operator std::string();
}

std::vector<std::string>
Configuration::
allKeys() const
{
    vector<string> result;

    for (Data::Entries::const_iterator
             it = data_->entries.begin(),
             end = data_->entries.end();
         it != end;  ++it)
        result.push_back(it->first);

    return result;
}

} // namespace ML
