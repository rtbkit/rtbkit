/* json_printing.cc
   Jeremy Barnes, 8 March 2013
   Copyright (c) 2013 Datacratic Inc.  All rights reserved.

   Functionality to print JSON values.
*/

#include "jml/utils/exc_assert.h"

#include "json_printing.h"


using namespace std;


namespace Datacratic {


void
StreamJsonPrintingContext::
writeStringUtf8(const Utf8String & s)
{
    stream << '\"';

    for (auto it = s.begin(), end = s.end();  it != end;  ++it) {
        int c = *it;
        if (c >= ' ' && c < 127 && c != '\"' && c != '\\')
            stream << (char)c;
        else {
            switch (c) {
            case '\t': stream << "\\t";  break;
            case '\n': stream << "\\n";  break;
            case '\r': stream << "\\r";  break;
            case '\b': stream << "\\b";  break;
            case '\f': stream << "\\f";  break;
            case '/':
            case '\\':
            case '\"': stream << '\\' << (char)c;  break;
            default:
                if (writeUtf8) {
                    char buf[4];
                    char * p = utf8::unchecked::append(c, buf);
                    stream.write(buf, p - buf);
                }
                else {
                    ExcAssert(c >= 0 && c < 65536);
                    stream << ML::format("\\u%04x", (unsigned)c);
                }
            }
        }
    }
    
    stream << '\"';
}


} // namespace Datacratic
