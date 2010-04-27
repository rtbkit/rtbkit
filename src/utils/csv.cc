/* csv.cc
   Jeremy Barnes, 5 April 2010
   Copyright (c) 2010 Jeremy Barnes.  All rights reserved.

   Code to parse a CSV file.
*/

#include "csv.h"
#include "parse_context.h"


using namespace std;


namespace ML {

std::string expect_csv_field(Parse_Context & context, bool & another)
{
    bool quoted = false;
    std::string result;
    another = false;
    
    while (context) {
        if (context.get_line() == 9723 && false)
            cerr << "*context = " << *context << " quoted = " << quoted
                 << " result = " << result << endl;
        
        if (quoted) {
            if (context.match_literal("\"\"")) {
                result += "\"";
                continue;
            }
            if (context.match_literal('\"')) {
                if (!*context == ',')
                    another = true;
                if (!context || context.match_literal(',')
                    || *context == '\n' || *context == '\r')
                    return result;
                cerr << "(bool)context = " << (bool)context << endl;
                cerr << "*context = " << *context << endl;
                cerr << "result = " << result << endl;

                for (unsigned i = 0; i < 20;  ++i)
                    cerr << *context++;

                context.exception_fmt("invalid end of line: %d %c",
                                      (int)*context, *context);
            }
        }
        else {
            if (context.match_literal('\"')) {
                if (result == "") {
                    quoted = true;
                    continue;
                }
                else context.exception("non-quoted string with embedded quote");
            }
            else if (context.match_literal(',')) {
                another = true;
                return result;
            }
            else if (*context == '\n' || *context == '\r')
                return result;
            
        }
        result += *context++;
    }

    if (quoted)
        throw Exception("file finished inside quote");

    return result;
}

std::vector<std::string>
expect_csv_row(Parse_Context & context, int length)
{
    context.skip_whitespace();

    vector<string> result;
    if (length != -1)
        result.reserve(length);

    bool another = false;
    while (another || (context && !context.match_eol())) {
        result.push_back(expect_csv_field(context, another));
        //cerr << "read " << result.back() << endl;
    }

    if (length != -1 && result.size() != length)
        throw Exception("Wrong CSV length: expected %d, got %zd", length,
                        result.size());
    
    return result;
}

std::string csv_escape(const std::string & s)
{
    int quote_pos = s.find('"');
    int nl_pos = s.find('\n');
    int comma_pos = s.find(',');

    if (quote_pos == string::npos && nl_pos == string::npos
        && comma_pos == string::npos)
        return s;

    string result = "\"";
    result.reserve(s.size() + 4);

    for (unsigned i = 0;  i < s.size();  ++i) {
        if (s[i] == '\"') result += "\"\"";
        else result += s[i];
    }

    result += "\"";

    return result;
}

} // namespace ML
