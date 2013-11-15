/* log_message_splitter.h                                          -*- C++ -*-
   Jeremy Barnes, 6 February 2012
   Copyright (c) 2012 Datacratic.  All rights reserved.

*/
#ifndef __logger__log_message_splitter_h__
#define __logger__log_message_splitter_h__

#include <cstring>
#include <string>

#include "jml/arch/exception.h"

namespace Datacratic {


/*****************************************************************************/
/* FIELD                                                                     */
/*****************************************************************************/

struct Field {
    const char * start;
    const char * end;

    size_t length() const { return end - start; }

    /* compare */
    int compare(const char * data) const
    { return std::strncmp(start, data, length()); }

    int compare(const std::string & str) const
    { return compare(str.c_str()); }

    /* operator == */
    bool operator == (const char * data) const
    { return (compare(data) == 0); }

    bool operator == (const std::string & str) const
    { return (compare(str) == 0); }

    /* operator != */
    bool operator != (const std::string & str) const
    { return !(operator ==(str)); }
    
    bool operator != (const char * data) const
    { return !(operator ==(data)); }

    /* string converter */
    operator std::string() const { return std::string(start, end); };
};

inline std::ostream & operator << (std::ostream & stream, const Field & field)
{
    return stream << field.operator std::string();
}


/*****************************************************************************/
/* LOG MESSAGE SPLITTER                                                      */
/*****************************************************************************/

template<int maxFields>
struct LogMessageSplitter {

    LogMessageSplitter(const std::string & str, char split = '\t')
        : str(str), data(this->str.c_str()), numFields(0)
    {
        numFields = 0;
        for (unsigned i = 0;  i <= maxFields;  ++i)
            offsets[i] = -1;

        const char * start = data;
        const char * end = start + str.size();

        while (start <= end && numFields < maxFields) {
            offsets[numFields++] = start - data;
            while (start < end && *start != split)
                ++start;
            if (start == end) break;
            ++start;  // skip delimiter
        }

        if (start == end)
            offsets[numFields] = end - data + 1;
        else offsets[numFields] = (start - data);
    }

    Field operator [] (int index) const
    {
        if (!(index >= 0) || index >= numFields || index >= maxFields)
            throw ML::Exception("invalid index");
        Field result;
        result.start = data + offsets[index];
        result.end = data + offsets[index + 1] - 1;
        return result;
    }

    size_t size() const { return numFields; }

    std::string str;
    const char * data;
    int numFields;
    int offsets[maxFields + 1];

};

} // namespace Datacratic

#endif /* __logger__log_message_splitter_h__ */


