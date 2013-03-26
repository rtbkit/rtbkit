/* json_printing.cc
   Jeremy Barnes, 8 March 2013
   Copyright (c) 2013 Datacratic Inc.  All rights reserved.

   Functionality to print JSON values.
*/

#include "json_printing.h"

namespace Datacratic {


void
StreamJsonPrintingContext::
writeStringUtf8(const Utf8String & s)
{
    stream << '\"';

    for (auto it = s.begin(), end = s.end();  it != end;  ++it) {
        int c = *it;
        if (c >= ' ' && c < 127 && c != '\"' && c != '\\')
            stream << c;
        else {
            switch (c) {
            case '\t': stream << "\\t";  break;
            case '\n': stream << "\\n";  break;
            case '\r': stream << "\\r";  break;
            case '\b': stream << "\\b";  break;
            case '\f': stream << "\\f";  break;
            case '/':
            case '\\':
            case '\"': stream << '\\' << c;  break;
            default:
                ExcAssert(c >= 0 && c < 65536);
                stream << "\\u" << ML::format("%04x", c);
            }
        }
    }
    
    stream << '\"';
}


} // namespace Datacratic
