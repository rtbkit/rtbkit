/* stat_aggregator.cc
   Jeremy Banres, 3 August 2011
   Copyright (c) 2011 Datacratic.  All rights reserved.

*/

#include "soa/service/stat_aggregator.h"
#include "ace/INET_Addr.h"
#include "jml/arch/exception.h"
#include "jml/arch/format.h"
#include <iostream>
#include "jml/arch/cmp_xchg.h"
#include "jml/arch/atomic_ops.h"
#include "jml/utils/floating_point.h"
#include "jml/utils/smart_ptr_utils.h"
#include "jml/utils/exc_check.h"
#include <boost/tuple/tuple.hpp>
#include <algorithm>


using namespace std;
using namespace ML;

namespace Datacratic {


/*****************************************************************************/
/* COUNTER AGGREGATOR                                                        */
/*****************************************************************************/

CounterAggregator::
CounterAggregator()
    : start(Date::now()), total(0.0),
      totalsBuffer() // Keep 10sec of data.
{
}

CounterAggregator::
~CounterAggregator()
{
}

void
CounterAggregator::
record(float value)
{
    double oldval = total;

    while (!ML::cmp_xchg(total, oldval, oldval + value));
}

std::pair<double, Date>
CounterAggregator::
reset()
{
    double oldval = total;

    while (!ML::cmp_xchg(total, oldval, 0.0));

    Date oldStart = start;
    start = Date::now();

    return make_pair(oldval, oldStart);
}

std::vector<StatReading>
CounterAggregator::
read(const std::string & prefix)
{
    double current;
    Date oldStart;

    boost::tie(current, oldStart) = reset();

    if (totalsBuffer.size() >= 10)
        totalsBuffer.pop_front();
    totalsBuffer.push_back(current);

    // Grab the average of the last x seconds to make sure that sparse values
    // show up in at least one of the x second carbon window. (x = 10 for now).
    double value = accumulate(totalsBuffer.begin(), totalsBuffer.end(), 0.0);
    value /= totalsBuffer.size();

    return vector<StatReading>(1, StatReading(prefix, value, start));
}


/*****************************************************************************/
/* GAUGE AGGREGATOR                                                          */
/*****************************************************************************/

GaugeAggregator::
GaugeAggregator(Verbosity verbosity, std::vector<int> extra)
    : verbosity(verbosity), values(new ML::distribution<float>())
    , extra(std::move(extra))
{
    if (verbosity == Outcome)
        ExcCheck(this->extra.size() > 0, "Can not construct with empty percentiles");

    values->reserve(100);
}

GaugeAggregator::
~GaugeAggregator()
{
    delete values;
}

void
GaugeAggregator::
record(float value)
{
    ML::distribution<float> * current = values;
    while ((current = values) == 0 || !cmp_xchg(values, current,
                                     (ML::distribution<float>*)0));
    
    current->push_back(value);

    memory_barrier();

    values = current;
}

std::pair<ML::distribution<float> *, Date>
GaugeAggregator::
reset()
{
    ML::distribution<float> * current = values;
    ML::distribution<float> * new_current = new ML::distribution<float>();

    // TODO: reserve memory for new_current

    while ((current = values) == 0 || !cmp_xchg(values, current, new_current));

    // Date oldStart = start;
    start = Date::now();

    return make_pair(current, start);
}

std::vector<StatReading>
GaugeAggregator::
read(const std::string & prefix)
{
    ML::distribution<float> * values;
    Date oldStart;

    boost::tie(values, oldStart) = reset();

    std::auto_ptr<ML::distribution<float> > vptr(values);

    if (values->empty())
        return vector<StatReading>();
    
    vector<StatReading> result;

    auto addMetric = [&] (const char * name, double value)
        {
            result.push_back(StatReading(prefix + "." + name,
                                         value, start));
        };
    
    auto percentile = [&] (float outOf100) -> double
        {
            int element
                = std::max(0,
                           std::min<int>(values->size() - 1,
                                         outOf100 / 100.0 * values->size()));
            return (*values)[element];
        };
    
    std::sort(values->begin(), values->end(),
              ML::safe_less<float>());

    if (verbosity == StableLevel)
        result.push_back(StatReading(prefix, values->mean(), start));
    
    else {
        addMetric("mean", values->mean());
        addMetric("upper", values->back());
        addMetric("lower", values->front());

        if (verbosity == Outcome) {
            addMetric("count", values->size());
            for (int pct: extra) {
                addMetric(ML::format("upper_%d", pct).c_str(), percentile(pct));
            }
        }
    }

    return result;
}

} // namespace Datacratic
