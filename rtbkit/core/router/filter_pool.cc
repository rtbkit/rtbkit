/** filter_pool.cc                                 -*- C++ -*-
    Rémi Attab, 24 Jul 2013
    Copyright (c) 2013 Datacratic.  All rights reserved.

    Implementation of the filter pool.

*/

#include "filter_pool.h"
#include "rtbkit/common/bid_request.h"
#include "rtbkit/common/exchange_connector.h"
#include "rtbkit/core/agent_configuration/agent_config.h"
#include "soa/service/service_base.h"
#include "jml/utils/exc_check.h"
#include "jml/arch/tick_counter.h"


using namespace std;
using namespace ML;


namespace RTBKIT {


/******************************************************************************/
/* FILTER POOL                                                                */
/******************************************************************************/

FilterPool::
FilterPool() : data(new Data()), events(nullptr) {}


void
FilterPool::
init(EventRecorder* events)
{
    this->events = events;
}


bool
FilterPool::
setData(Data*& oldData, unique_ptr<Data>& newData)
{
    if (!data.compare_exchange_strong(oldData, newData.get()))
        return false;

    newData.release();
    gc.defer([=] { delete oldData; });

    return true;
}


FilterPool::
~FilterPool()
{
    {
        GcLockBase::SharedGuard guard(gc);

        unique_ptr<Data> nil;
        Data* current = data.load();
        while (!setData(current, nil));
    }

    gc.deferBarrier();
}


void
FilterPool::
recordDiff(const Data* data, const FilterBase* filter, const ConfigSet& diff)
{
    for (size_t cfg = diff.next(); cfg < diff.size(); cfg = diff.next(cfg+1)) {
        const AgentConfig& config = *data->configs[cfg].config;

        events->recordHit("accounts.%s.filter.static.%s",
                config.account.toString('.'), filter->name());
    }
}

void
FilterPool::
recordReason(const Data* data, const FilterBase* f, FilterState & state){

    FilterState::FilterReasons& reasons = state.getFilterReasons();
    for ( auto it = reasons.begin() ; it != reasons.end(); ++it) {

        for (std::size_t idx = it->second.next();
             idx < it->second.size();
             idx = it->second.next(idx + 1))
        {
            const AgentConfig& config = *data->configs[idx].config;
            events->recordHit("accounts.%s.filter.static.reasons.%s.%s",
                               config.account.toString('.'),
                               f->name(),
                               it->first);
        }

    }

}

uint64_t
FilterPool::
recordTime(uint64_t start, const FilterBase* filter)
{
    uint64_t now = ticks();
    double us = ((now - start) / ticks_per_second) * 1000000.0;

    events->recordLevel(us, "filters.timingUs.%s", filter->name());

    return now;
}


FilterPool::ConfigList
FilterPool::
filter(const BidRequest& br, const ExchangeConnector* conn, const ConfigSet& mask)
{
    GcLockBase::SharedGuard guard(gc, GcLockBase::RD_NO);

    const Data* current = data.load();
    ExcCheck(!current->filters.empty(), "No filters registered");

    FilterState state(br, conn, current->activeConfigs);
    state.narrowConfigs(mask);

    ConfigSet configs = state.configs();

    bool sampleStats = events && (random() % 10 == 0);
    uint64_t ticksStart = sampleStats ? ticks() : 0;

    for (FilterBase* filter : current->filters) {
        filter->filter(state);

        const ConfigSet& filtered = state.configs();

        if (sampleStats) {
            ticksStart = recordTime(ticksStart, filter);
            recordDiff(current, filter, configs ^ filtered);
            if (!state.getFilterReasons().empty()) {
                recordReason(current, filter, state);
            }
            configs = filtered;
        }
        state.resetFilterReasons();

        if (filtered.empty()) {
            if (sampleStats) 
                events->recordHit("filters.breakLoop.%s", filter->name());
            break;
        }
    }

    auto biddableSpots = state.biddableSpots();
    configs = state.configs();

    ConfigList result;
    for (size_t i = configs.next(); i < configs.size(); i = configs.next(i + 1)) {
        ConfigEntry entry = current->configs[i];
        entry.biddableSpots = std::move(biddableSpots[i]);
        result.emplace_back(std::move(entry));
    }

    return result;
}


void
FilterPool::
addFilter(const string& name)
{
    GcLockBase::SharedGuard guard(gc);

    Data* oldData = data.load();
    unique_ptr<Data> newData;

    do {
        newData.reset(new Data(*oldData));
        newData->addFilter(PluginInterface<FilterBase>::getPlugin(name)());
    } while (!setData(oldData, newData));

    if (events) events->recordHit("filters.addFilter.%s", name);
}


void
FilterPool::
removeFilter(const string& name)
{
    GcLockBase::SharedGuard guard(gc);

    unique_ptr<Data> newData;
    Data* oldData = data.load();

    do {
        newData.reset(new Data(*oldData));
        newData->removeFilter(name);
    } while (!setData(oldData, newData));

    if (events) events->recordHit("filters.removeFilter.%s", name);
}

void
FilterPool::
initWithDefaultFilters()
{
    GcLockBase::SharedGuard guard(gc);

    Data* oldData = data.load();
    unique_ptr<Data> newData;

    do {
        newData.reset(new Data);
        for (const auto& ele: PluginInterface<FilterBase>::getNames()) {
            newData->addFilter(PluginInterface<FilterBase>::getPlugin(ele)());
            if (events) events->recordHit("filters.addFilter.%s", ele);
        }

    } while (!setData(oldData, newData));

}

void
FilterPool::
initWithFiltersFromJson(const Json::Value & json)
{

    GcLockBase::SharedGuard guard(gc);

    Data* oldData = data.load();
    unique_ptr<Data> newData;

    if (!json.isArray())
        throw Exception("filter list must be an array");

    do {
        newData.reset(new Data);

        for (unsigned i = 0;  i < json.size();  ++i) {
            const Json::Value & val = json[i];
            newData->addFilter(PluginInterface<FilterBase>::getPlugin(val.asString())());
            if (events) events->recordHit("filters.addFilter.%s", val.asString());
        }
    } while (!setData(oldData, newData));
}


unsigned
FilterPool::
addConfig(const string& name, const AgentInfo& info)
{
    GcLockBase::SharedGuard guard(gc);

    unique_ptr<Data> newData;
    Data* oldData = data.load();
    unsigned index;

    do {
        newData.reset(new Data(*oldData));
        index = newData->addConfig(name, info);
    } while (!setData(oldData, newData));

    if (events) events->recordHit("filters.addConfig");

    return index;
}


void
FilterPool::
removeConfig(const string& name)
{
    GcLockBase::SharedGuard guard(gc);

    unique_ptr<Data> newData;
    Data* oldData = data.load();

    do {
        newData.reset(new Data(*oldData));
        newData->removeConfig(name);
    } while (!setData(oldData, newData));

    if (events) events->recordHit("filters.removeConfig");
}

std::vector<string>
FilterPool::
getFilterNames() const
{
    GcLockBase::SharedGuard guard(gc, GcLockBase::RD_NO);

    const Data* current = data.load();
    std::vector<string> filter_names;
    filter_names.reserve(current->filters.size());

    for (FilterBase* filter : current->filters) {
        filter_names.push_back(filter->name());
    }

    return filter_names;
}


/******************************************************************************/
/* FILTER POOL - DATA                                                         */
/******************************************************************************/

FilterPool::Data::
Data(const Data& other) :
    configs(other.configs),
    activeConfigs(other.activeConfigs)
{
    filters.reserve(other.filters.size());
    for (FilterBase* filter : other.filters)
        filters.push_back(filter->clone());
}


FilterPool::Data::
~Data()
{
    for (FilterBase* filter : filters) delete filter;
}

ssize_t
FilterPool::Data::
findConfig(const string& name) const
{
    for (size_t i = 0; i < configs.size(); ++i) {
        if (configs[i].name == name) return i;
    }
    return -1;
}

unsigned
FilterPool::Data::
addConfig(const string& name, const AgentInfo& info)
{
    // If our config already exists, we have to deregister it with the filters
    // before we can add the new config.
    removeConfig(name);

    ssize_t index = findConfig("");
    if (index >= 0)
        configs[index] = ConfigEntry(name, info);
    else {
        index = configs.size();
        configs.emplace_back(name, info);
    }

    activeConfigs.setConfig(index, info.config->creatives.size());

    for (FilterBase* filter : filters)
        filter->addConfig(index, info.config);

    return index;
}

void
FilterPool::Data::
removeConfig(const string& name)
{
    ssize_t index = findConfig(name);
    if (index < 0) return;

    activeConfigs.resetConfig(index);

    for (FilterBase* filter : filters)
        filter->removeConfig(index, configs[index].config);

    configs[index].reset();
}


ssize_t
FilterPool::Data::
findFilter(const string& name) const
{
    for (size_t i = 0; i < filters.size(); ++i) {
        if (filters[i]->name() == name) return i;
    }
    return -1;
}

void
FilterPool::Data::
addFilter(FilterBase* filter)
{
    ConfigSet active = activeConfigs.aggregate();
    for (size_t cfgId = active.next();
         cfgId < active.size();
         cfgId += active.next(cfgId + 1))
    {
        filter->addConfig(cfgId, configs[cfgId].config);
    }

    filters.push_back(filter);
    sort(filters.begin(), filters.end(), [] (FilterBase* lhs, FilterBase* rhs) {
                return lhs->priority() < rhs->priority();
            });
}

void
FilterPool::Data::
removeFilter(const string& name)
{
    ssize_t index = findFilter(name);
    if (index < 0) return;

    delete filters[index];

    for (size_t i = index; i < filters.size() - 1; ++i)
        filters[i] = filters[i+1];

    filters.pop_back();
}

} // namepsace RTBKit
