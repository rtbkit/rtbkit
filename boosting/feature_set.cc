/* feature_set.cc
   Jeremy Barnes, 10 June 2003
   Copyright (c) 2003 Jeremy Barnes.  All rights reserved.
   $Source$

   Implementation of the feature set.
*/

#include "config_impl.h"
#include "feature_set.h"
#include "training_data.h"
#include "jml/utils/parse_context.h"
#include "jml/arch/demangle.h"
#include "registry.h"
#include <typeinfo>
#include <string>


using namespace std;
using namespace DB;



namespace ML {


/*****************************************************************************/
/* FEATURE_SET                                                               */
/*****************************************************************************/

std::ostream & operator << (std::ostream & stream, const Feature & feature)
{
    return stream << feature.print();
}


/*****************************************************************************/
/* MUTABLE_FEATURE_SET                                                       */
/*****************************************************************************/

/** Structure used to implement the sort() routine. */
struct Mutable_Feature_Set::compare_feature {
    bool operator () (const std::pair<Feature, float> & p1,
                      const std::pair<Feature, float> & p2) const
    {
        safe_less<float> value_cmp;
        if (p1.first < p2.first) return true;
        else if (p2.first < p1.first) return false;
        return value_cmp(p1.second, p2.second);
    }
};

void Mutable_Feature_Set::sort()
{
    Mutable_Feature_Set::do_sort();
}

void Mutable_Feature_Set::do_sort() const
{
    if (is_sorted) return;
    if (locked) throw Exception("mutating locked feature set");
    std::sort(features.begin(), features.end(), compare_feature());
    is_sorted = true;
}

void Mutable_Feature_Set::add(const Feature & feat, float val)
{
    if (locked) throw Exception("mutating locked feature set");
    features.push_back(std::make_pair(feat, val));
    if (features.size() == 1) is_sorted = true;
    else if (is_sorted && compare_feature()(features[features.size() - 2],
                                            features.back()))
        is_sorted = true;
    else is_sorted = false;
}

void Mutable_Feature_Set::
replace(const Feature & feat, float val)
{
    if (locked) throw Exception("mutating locked feature set");
    if (is_sorted) {
        /* Can do this more sensibly, and keep it sorted... */
        features_type::iterator it
            = std::lower_bound(features.begin(), features.end(),
                               make_pair(feat, -INFINITY));
        
        if (it == end() || it->first != feat)
            add(feat, val);
        else {
            it->second = val;
            ++it;
            int start = it - features.begin();
            int pos = 0;

            /* Shift those to delete back to maintain sorted property. */
            int num_to_delete = 0;
            while (pos < features.size() && features[pos].first == feat) {
                ++num_to_delete;
                ++pos;
            }
                
            for (unsigned i = 0;  i < num_to_delete;  ++i)
                features[start + i] = features[start + i + num_to_delete];

            for (unsigned i = 0;  i < num_to_delete;  ++i)
                features.pop_back();
        }
    }
    else {
        /* Scan through the entire thing, replacing each one we find. */
        bool done = false;
        for (unsigned i = 0;  i < features.size(); /* no inc */) {
            if (features[i].first == feat) {
                if (done) {
                    /* Remove a duplicate instance */
                    std::swap(features[i], features.back());
                    features.pop_back();
                    continue;  // without incrementing i
                }
                else {
                    /* Replace this instance */
                    features[i].second = val;
                    done = true;
                }
            }

            ++i;
        }

        if (!done) add(feat, val);
    }
}

Mutable_Feature_Set * Mutable_Feature_Set::make_copy() const
{
    return new Mutable_Feature_Set(*this);
}


/*****************************************************************************/
/* MISCELLANEOUS                                                             */
/*****************************************************************************/

std::string escape_feature_name(const std::string & feature)
{
    int quotes = 0, backslashes = 0, escaped = 0, unescaped = 0;
    for (unsigned i = 0;  i < feature.length();  ++i) {
        switch (feature[i]) {
        case '"':  quotes += 1;       break;
        case '\\': backslashes += 1;  break;
        case ':':
        case '|':
        case ',':
        case ' ':  escaped += 1;      break;
        default:
            unescaped += 1;
        }
    }

    /* Decide how to go about it (which gives the shortest length?) */
    if (quotes == 0 && backslashes == 0 && escaped == 0) return feature;
    int quoted_length = unescaped + escaped + 2 * (backslashes + quotes + 1); 
    int escaped_length = unescaped + 2 * (backslashes + quotes + escaped);

    /* Give a one character bonus to the quoted version since it's easier to
       read. */
    if (quoted_length <= escaped_length + 1) {
        /* Use the quoted version. */
        string result;
        result.reserve(quoted_length);
        result += '"';

        for (unsigned i = 0;  i < feature.length();  ++i) {
            switch (feature[i]) {
            case '"':  result += "\\\"";  break;
            case '\\': result += "\\\\";  break;
            default:
                result += feature[i];
            }
        }

        result += '"';
        
        return result;
    }
    else {
        /* Use the escaped version. */
        string result;
        result.reserve(escaped_length);
        
        for (unsigned i = 0;  i < feature.length();  ++i) {
            switch (feature[i]) {
            case '"':
            case '\\':
            case ':':
            case '|':
            case ' ':
            case ',':
            result += '\\';
            default:
                result += feature[i];
            }
        }

        return result;
    }
}

std::string expect_feature_name(Parse_Context & c)
{
    std::string result;
    if (c.match_literal('"')) {
        /* We have a quoted name. */

        bool after_backslash = false;
        int len = 0;

        Parse_Context::Revert_Token tok(c);

        while (c && (*c != '"' || after_backslash)) {
            if ((!after_backslash && *c == '"') || *c == '\n') break;
            if (*c == '\\') after_backslash = true;
            else { ++len;  after_backslash = false; }
            ++c;
        }

        c.expect_literal('"', "expected closing quotation mark ('\"')");
        if (after_backslash) c.exception("Invalid backslash escaping");

        result.reserve(len);
        tok.apply();  // back to the start...

        after_backslash = false;
        while (c && (*c != '"' || after_backslash)) {
            if ((!after_backslash && *c == '"') || *c == '\n') break;
            if (*c == '\\') after_backslash = true;
            else { result += *c;  after_backslash = false; }
            ++c;
        }
        c.expect_literal('"', "expected closing quotation mark ('\"')");
    }
    else {
        /* We have a backslash escaped name. */
        bool after_backslash = false;
        
        Parse_Context::Revert_Token tok(c);

        int len = 0;
        while (c && *c != '\n') {
            if (!after_backslash && (isspace(*c) || *c == ':' || *c == ','))
                break;
            if (*c == '\\') after_backslash = true;
            else { ++len;  after_backslash = false; }
            ++c;
        }

        result.reserve(len);
        tok.apply();

        if (after_backslash) c.exception("Invalid backslash escaping");
        after_backslash = false;
        while (c && *c != '\n') {
            if (!after_backslash && (isspace(*c) || *c == ':'
                                     || *c == ',')) break;
            if (*c == '\\') after_backslash = true;
            else { result += *c;  after_backslash = false; }
            ++c;
        }
    }

    if (result.empty())
        c.exception("expect_feature_name(): no feature name found");
    
    return result;
}

} // namespace ML



