/* csv.h                                                           -*- C++ -*-
   Jeremy Barnes, 5 April 2010
   Copyright (c) 2010 Jeremy Barnes.  All rights reserved.

   Comma Separated Value parsing code.
*/

#ifndef __utils__csv_h__
#define __utils__csv_h__

#include <string>
#include <vector>

namespace ML {

struct Parse_Context;

/** Expect a CSV field from the given parse context.  Another will be set
    to true if there is still another field in the CSV row. */
std::string expect_csv_field(Parse_Context & context, bool & another,
                             char separator = ',');


/** Expect a row of CSV from the given parse context.  If length is not -1,
    then the extact number of fields required is given in that parameter. */
std::vector<std::string>
expect_csv_row(Parse_Context & context, int length = -1, char separator = ',');

/** Convert the string to a CSV representation, escaping everything that
    needs to be escaped. */
std::string csv_escape(const std::string & s);

} // namespace ML


#endif /* __utils__csv_h__ */
