/** periodic_utils_value_descriptions.cc
    Jeremy Barnes, 26 August 2013
    Copyright (c) 2013 Datacratic Inc.  All rights reserved.

*/

#include "periodic_utils.h"
#include "value_description.h"

namespace Datacratic {

struct TimePeriodDescription: public ValueDescriptionT<TimePeriod> {
    virtual void parseJsonTyped(TimePeriod * val,
                                JsonParsingContext & context) const
    {
        val->parse(context.expectStringAscii());
    }

    virtual void printJsonTyped(const TimePeriod * val,
                                JsonPrintingContext & context) const
    {
        context.writeString(val->toString());
    }
};

ValueDescriptionT<TimePeriod> * getDefaultDescription(TimePeriod *)
{
    return new TimePeriodDescription();
}

} // namespace Datacratic
