/* service_base.h                                                  -*- C++ -*-
   Jeremy Barnes, 29 May 2012
   Copyright (c) 2012 Datacratic.  All rights reserved.

*/

#ifndef __service__service_base_h__
#define __service__service_base_h__

#include "port_range_service.h"
#include "soa/service/stats_events.h"
#include "stdarg.h"
#include "jml/compiler/compiler.h"
#include <string>
#include <boost/shared_ptr.hpp>
#include "jml/arch/exception.h"
#include "jml/arch/format.h"
#include "jml/arch/spinlock.h"
#include <map>
#include <mutex>
#include "soa/jsoncpp/json.h"
#include <unordered_map>
#include <mutex>
#include <thread>
#include <initializer_list>
#include "jml/utils/exc_assert.h"
#include "jml/utils/unnamed_bool.h"


namespace zmq {
struct context_t;
} // namespace zmq

namespace Datacratic {


class MultiAggregator;
class CarbonConnector;

/*****************************************************************************/
/* EVENT SERVICE                                                             */
/*****************************************************************************/

struct EventService {
    virtual ~EventService()
    {
    }
    
    virtual void onEvent(const std::string & name,
                         const char * event,
                         StatEventType type,
                         float value,
                         std::initializer_list<int> extra = DefaultOutcomePercentiles) = 0;

    virtual void dump(std::ostream & stream) const
    {
    }

    /** Dump the content
    */
    std::map<std::string, double> get(std::ostream & output) const;
};

/*****************************************************************************/
/* NULL EVENT SERVICE                                                        */
/*****************************************************************************/

struct NullEventService : public EventService {

    NullEventService();
    ~NullEventService();
    
    virtual void onEvent(const std::string & name,
                         const char * event,
                         StatEventType type,
                         float value,
                         std::initializer_list<int> extra = DefaultOutcomePercentiles);

    virtual void dump(std::ostream & stream) const;

    std::unique_ptr<MultiAggregator> stats;
};


/*****************************************************************************/
/* CARBON EVENT SERVICE                                                      */
/*****************************************************************************/

struct CarbonEventService : public EventService {

    CarbonEventService(std::shared_ptr<CarbonConnector> conn);
    CarbonEventService(const std::string & connection,
                       const std::string & prefix = "",
                       double dumpInterval = 1.0);
    CarbonEventService(const std::vector<std::string> & connections,
                       const std::string & prefix = "",
                       double dumpInterval = 1.0);

    virtual void onEvent(const std::string & name,
                         const char * event,
                         StatEventType type,
                         float value,
                         std::initializer_list<int> extra = std::initializer_list<int>());

    std::shared_ptr<CarbonConnector> connector;
};


/*****************************************************************************/
/* CONFIGURATION SERVICE                                                     */
/*****************************************************************************/

struct ConfigurationService {

    virtual ~ConfigurationService()
    {
    }

    enum ChangeType {
        VALUE_CHANGED,   ///< Contents of value have changed
        DELETED,         ///< Value has been deleted outright
        CREATED,         ///< Entry was created
        NEW_CHILD        ///< Entry has new children
    };
    
    /** Callback that will be called if a given entry changes. */
    typedef std::function<void (std::string path,
                                ChangeType changeType)> OnChange;

    /** Type used to hold a watch on a node. */
    struct Watch {
        struct Data {
            Data(const OnChange & onChange)
                : onChange(onChange),
                  watchReferences(1)
            {
            }

            OnChange onChange;
            int watchReferences;  ///< How many watches reference this callback?
        };
        
        Watch()
            : data(0)
        {
        }

        /** Set up a callback for a watch.  The path and the change type will
            be passed in.
        */
        Watch(const OnChange & onChange)
            : data(new Data(onChange))
        {
        }

        Watch(const Watch & other)
            : data(other.data)
        {
            if (data)
                ++data->watchReferences;
        }

        Watch(Watch && other)
            : data(other.data)
        {
            other.data = 0;
        }

        void swap(Watch & other)
        {
            using std::swap;
            swap(data, other.data);
        }

        Watch & operator = (const Watch & other)
        {
            Watch newMe(other);
            swap(newMe);
            return *this;
        }

        Watch & operator = (Watch && other)
        {
            Watch newMe(other);
            swap(newMe);
            return *this;
        }

        void init(const OnChange & onChange)
        {
            if (data)
                throw ML::Exception("double initializing watch");
            data.reset(new Data(onChange));
        }

        void trigger(const std::string & path, ChangeType change)
        {
            if (!data)
                throw ML::Exception("triggered unused watch");
            if (!data->watchReferences <= 0)
                return;
            ExcAssert(data);
            data->onChange(path, change);
        }

        void disable()
        {
            if (!data)
                return;
            data->watchReferences = 0;
        }

        std::shared_ptr<Data> * get()
        {
            if (!data)
                return nullptr;
            return new std::shared_ptr<Data>(data);
        }

        ~Watch()
        {
            if (data) {
                __sync_add_and_fetch(&data->watchReferences, -1);
            }
            // TODO: we need to unregister ourselves here so that we don't get a
            // callback to something that doesn't exist
        }

        JML_IMPLEMENT_OPERATOR_BOOL(data.get());

    private:
        std::shared_ptr<Data> data;
    };

    /** Set the JSON value of a node.  If the linkedKey is set, then the
        lifecycle of the value is linked to the lifecycle of the linked
        key, and if the linked key is deleted the given value will be
        deleted also.
    */
    virtual void set(const std::string & key,
                     const Json::Value & value) = 0;

    /** Set the value of a node, ensuring that it's created by adding a
        suffix if necessary.  Returns the actual place under which it was
        added.
    */
    virtual std::string setUnique(const std::string & key,
                                  const Json::Value & value) = 0;

    /** Get the JSON value of a node in a synchronous manner. */
    virtual Json::Value
    getJson(const std::string & key, Watch watch = Watch()) = 0;
    
    /** Callback called for each entry.  Return code is whether we should
        continue iterating or not.
    */
    typedef std::function<bool (std::string key, Json::Value value)>
    OnEntry;

    /** Return the set of children for the given node.  If the node does
        not exist, the watch will be set for the creation of the node
        itself.
    */
    virtual std::vector<std::string>
    getChildren(const std::string & key,
                Watch watch = Watch()) = 0;

    /** Iterate over all entries in the configuration database recursively,
        calling the given callback.  Note that no watches can be set using
        this function.
    */
    virtual bool forEachEntry(const OnEntry & onEntry,
                              const std::string & startPrefix = "") const = 0;

    /** Recursively remove everything below this path. */
    virtual void removePath(const std::string & path) = 0;

    /** Dump the contents to the given stream.  Mostly useful for
        debugging purposes.
    */
    void dump(std::ostream & stream) const;

    /** Dump the contents to a json object.
    */
    Json::Value jsonDump() const;

    static std::pair<std::string, std::string>
    splitPath(const std::string & path);

    /** Store the current installation and location
     */

    std::string currentInstallation;
    std::string currentLocation;
};


/*****************************************************************************/
/* INTERNAL CONFIGURATION SERVICE                                            */
/*****************************************************************************/

/** In-process configuration service with synchronization via locks. */

struct InternalConfigurationService : public ConfigurationService {

    virtual ~InternalConfigurationService()
    {
    }

    /** Get the JSON value of a node.  Will return a null value if the
        node doesn't exist.
    */
    virtual Json::Value getJson(const std::string & value,
                                Watch watch = Watch());
    
    /** Set the value of a node. */
    virtual void set(const std::string & key,
                     const Json::Value & value);

    /** Set the value of a node, ensuring that it's created by adding a
        suffix if necessary.  Returns the actual place under which it was
        added.
    */
    virtual std::string setUnique(const std::string & key,
                                  const Json::Value & value);

    virtual std::vector<std::string>
    getChildren(const std::string & key,
                Watch watch = Watch());

    virtual bool forEachEntry(const OnEntry & onEntry,
                              const std::string & startPrefix = "") const;

    /** Recursively remove everything below this path. */
    virtual void removePath(const std::string & path);

private:
    struct Entry {
        Entry()
            : hasValue(false)
        {
        }

        bool hasValue;
        Json::Value value;
        Watch watch;
        std::unordered_map<std::string, std::shared_ptr<Entry> > children;
    };

    Entry & createNode(Entry & node, const std::string & key);
    const Entry * getNode(const Entry & node, const std::string & key) const;
    Entry * getNode(Entry & node, const std::string & key);

    /** Base entry for the entire service. */
    Entry root;
    
    typedef std::recursive_mutex Lock;
    typedef std::unique_lock<Lock> Guard;
    mutable Lock lock;
};

/*****************************************************************************/
/* NULL CONFIGURATION SERVICE                                                */
/*****************************************************************************/

/** Configuration Service that does nothing and does not keep any information */

struct NullConfigurationService : public ConfigurationService {
    virtual ~NullConfigurationService()
    {
    }

    virtual Json::Value getJson(const std::string & value,
                                Watch watch = Watch());

    virtual void set(const std::string & key,
                     const Json::Value & value);

    virtual std::string setUnique(const std::string & key,
                                  const Json::Value & value);

    virtual std::vector<std::string>
    getChildren(const std::string & key,
                Watch watch = Watch());

    virtual bool forEachEntry(const OnEntry & onEntry,
                              const std::string & startPrefix = "") const;

    virtual void removePath(const std::string & path);
};


/*****************************************************************************/
/* SERVICE PROXIES                                                           */
/*****************************************************************************/

struct ServiceProxies {
    ServiceProxies();

    std::shared_ptr<EventService> events;
    std::shared_ptr<ConfigurationService> config;
    std::shared_ptr<PortRangeService> ports;
    Json::Value params;

    std::string bankerUri;

    /** Zeromq context for communication. */
    std::shared_ptr<zmq::context_t> zmqContext;

    template<typename Configuration> 
    JML_ALWAYS_INLINE
    std::shared_ptr<Configuration> configAs() 
    {
        return std::static_pointer_cast<Configuration>(config);
    }

    void logToCarbon(std::shared_ptr<CarbonConnector> conn);
    void logToCarbon(const std::string & carbonConnection,
                     const std::string & prefix = "",
                     double dumpInterval = 1.0);
    void logToCarbon(const std::vector<std::string> & carbonConnections,
                     const std::string & prefix = "",
                     double dumpInterval = 1.0);

    void useZookeeper(std::string url = "localhost:2181",
                      std::string prefix = "CWD",
                      std::string location = "global");

    void usePortRanges(const std::string& path);
    void usePortRanges(const Json::Value& config);

    std::vector<std::string>
    getServiceClassInstances(std::string const & name,
                             std::string const & protocol = "http");

    std::vector<std::string>
    getEndpointInstances(std::string const & name,
                         std::string const & protocol = "http");

    // Bootstrap the proxies services using a json configuration.
    void bootstrap(const std::string& path);
    void bootstrap(const Json::Value& config);
};


/*****************************************************************************/
/* EVENT RECORDER                                                            */
/*****************************************************************************/

/** Bridge class to an event recorder. */

struct EventRecorder {

    EventRecorder(const std::string & eventPrefix,
                  const std::shared_ptr<EventService> & events);

    EventRecorder(const std::string & eventPrefix,
                  const std::shared_ptr<ServiceProxies> & services);


    /*************************************************************************/
    /* EVENT RECORDING                                                       */
    /*************************************************************************/

    /** Notify that an event has happened.  Fields are:
        eventNum:  an ID for the event;
        eventName: the name of the event;
        eventType: the type of the event.  Default is ET_COUNT;
        value:     the value of the event (quantity being measured).
                   Default is 1.0;
        units:     the units of the event (eg, ms).  Default is unitless.
    */
    void recordEvent(const char * eventName,
                     StatEventType type = ET_COUNT,
                     float value = 1.0,
                     std::initializer_list<int> extra = DefaultOutcomePercentiles) const
    {
        EventService * es = 0;
        if (events_)
            es = events_.get();
        if (!es && services_)
            es = services_->events.get();
        if (!es)
        {
            std::cerr << "no services configured!!!!" << std::endl;
            return;
        }
        es->onEvent(eventPrefix_, eventName, type, value, extra);
    }

    void recordEventFmt(StatEventType type,
                        float value,
                        std::initializer_list<int> extra,
                        const char * fmt, ...) const JML_FORMAT_STRING(5, 6);

    template<typename... Args>
    void recordHit(const std::string & event, Args... args) const
    {
        return recordEventFmt(ET_HIT, 1.0, {}, event.c_str(),
                              ML::forwardForPrintf(args)...);
    }

    template<typename... Args>
    JML_ALWAYS_INLINE
    void recordHit(const char * event, Args... args) const
    {
        return recordEventFmt(ET_HIT, 1.0, {}, event,
                              ML::forwardForPrintf(args)...);
    }

    void recordHit(const char * event) const
    {
        recordEvent(event, ET_HIT);
    }

    void recordHit(const std::string & event) const
    {
        recordEvent(event.c_str(), ET_HIT);
    }

    template<typename... Args>
    void recordCount(float count, const std::string & event, Args... args) const
    {
        return recordEventFmt(ET_COUNT, count, {}, event.c_str(),
                              ML::forwardForPrintf(args)...);
    }
    
    template<typename... Args>
    JML_ALWAYS_INLINE
    void recordCount(float count, const char * event, Args... args) const
    {
        return recordEventFmt(ET_COUNT, count, {}, event,
                              ML::forwardForPrintf(args)...);
    }

    void recordCount(float count, const char * event) const
    {
        recordEvent(event, ET_COUNT, count);
    }

    void recordCount(float count, const std::string & event) const
    {
        recordEvent(event.c_str(), ET_COUNT, count);
    }

    template<typename... Args>
    void recordOutcome(float outcome,
                       const std::string & event, Args... args) const
    {
        return recordEventFmt(ET_OUTCOME, outcome, DefaultOutcomePercentiles, event.c_str(),
                             ML::forwardForPrintf(args)...);
    }
    
    template<typename... Args>
    void recordOutcome(float outcome, const char * event, Args... args) const
    {
        return recordEventFmt(ET_OUTCOME, outcome, DefaultOutcomePercentiles, event,
                              ML::forwardForPrintf(args)...);
    }

    void recordOutcome(float outcome, const char * event) const
    {
        recordEvent(event, ET_OUTCOME, outcome, DefaultOutcomePercentiles);
    }

    void recordOutcome(float outcome, const std::string & event) const
    {
        recordEvent(event.c_str(), ET_OUTCOME, outcome, DefaultOutcomePercentiles);
    }

    template<typename... Args>
    void recordOutcomeCustom(float outcome, std::initializer_list<int> percentiles,
                               const std::string& event, Args... args) const
    {
        return recordEventFmt(ET_OUTCOME, outcome, percentiles, event.c_str(),
                             ML::forwardForPrintf(args)...);
    }

    template<typename... Args>
    void recordOutcomeCustom(float outcome, std::initializer_list<int> percentiles,
                               const char * event, Args... args) const
    {
        return recordEventFmt(ET_OUTCOME, outcome, percentiles, event,
                              ML::forwardForPrintf(args)...);
    }

    void recordOutcomeCustom(float outcome, std::initializer_list<int> percentiles,
                               const char * event) const
    {
        recordEvent(event, ET_OUTCOME, outcome, percentiles);
    }

    void recordOutcomeCustom(float outcome, std::initializer_list<int> percentiles,
                               const std::string & event) const
    {
        recordEvent(event.c_str(), ET_OUTCOME, outcome, percentiles);
    }
    
    template<typename... Args>
    void recordLevel(float level, const std::string & event, Args... args) const
    {
        return recordEventmt(ET_LEVEL, level, {}, event.c_str(),
                             ML::forwardForPrintf(args)...);
    }
    
    template<typename... Args>
    void recordLevel(float level, const char * event, Args... args) const
    {
        return recordEventFmt(ET_LEVEL, level, {}, event,
                              ML::forwardForPrintf(args)...);
    }

    void recordLevel(float level, const char * event) const
    {
        recordEvent(event, ET_LEVEL, level);
    }

    void recordLevel(float level, const std::string & event) const
    {
        recordEvent(event.c_str(), ET_LEVEL, level);
    }

    template<typename... Args>
    void recordStableLevel(float level, const std::string & event, Args... args) const
    {
        return recordEventmt(ET_STABLE_LEVEL, level, event.c_str(),
                             ML::forwardForPrintf(args)...);
    }

    template<typename... Args>
    void recordStableLevel(float level, const char * event, Args... args) const
    {
        return recordEventFmt(ET_STABLE_LEVEL, level, event,
                              ML::forwardForPrintf(args)...);
    }

    void recordStableLevel(float level, const char * event) const
    {
        recordEvent(event, ET_STABLE_LEVEL, level);
    }

    void recordStableLevel(float level, const std::string & event) const
    {
        recordEvent(event.c_str(), ET_STABLE_LEVEL, level);
    }

protected:
    std::string eventPrefix_;
    std::shared_ptr<EventService> events_;
    std::shared_ptr<ServiceProxies> services_;
};


/*****************************************************************************/
/* SERVICE BASE                                                              */
/*****************************************************************************/

struct ServiceBase: public EventRecorder {
    /** Construct as a top level parent. */
    ServiceBase(const std::string & serviceName,
                std::shared_ptr<ServiceProxies> proxies = std::shared_ptr<ServiceProxies>());

    /** Construct as a child of an existing parent. */
    ServiceBase(const std::string & subServiceName,
                ServiceBase & parent);

    virtual ~ServiceBase();

    void setServices(std::shared_ptr<ServiceProxies> services)
    {
        services_ = services;
    }

    std::shared_ptr<ServiceProxies> getServices() const
    {
        return services_;
    }

    std::string serviceName() const
    {
        return serviceName_;
    }

    /*************************************************************************/
    /* REGISTRATION                                                          */
    /*************************************************************************/

    /** Register this as a service, that provides the given service classes.

        This will create sufficient entries in the configuration that something
        needing to find providers of any of the given services will be able
        to find this service.
    */
    
    void registerServiceProvider(const std::string & name,
                                 const std::vector<std::string> & serviceClasses);

    void registerShardedServiceProvider(const std::string & name,
                                        const std::vector<std::string> & serviceClasses,
                                        size_t shardIndex);

    /** Unregister service from configuration service. */

    void unregisterServiceProvider(const std::string & name,
                                   const std::vector<std::string> & serviceClasses);


    /*************************************************************************/
    /* ZEROMQ CONTEXT                                                        */
    /*************************************************************************/

    std::shared_ptr<zmq::context_t> getZmqContext() const
    {
        return services_->zmqContext;
    }

    /*************************************************************************/
    /* CONFIGURATION SERVICE                                                 */
    /*************************************************************************/

    std::shared_ptr<ConfigurationService> getConfigurationService() const
    {
        return services_->config;
    }

    /*************************************************************************/
    /* EXCEPTION LOGGING                                                     */
    /*************************************************************************/

    void logException(std::exception_ptr exc,
                      const std::string & context)
    {
        std::cerr << "error: exception in context " << context << std::endl;
        std::rethrow_exception(exc);
    }


    /*************************************************************************/
    /* STATUS                                                                */
    /*************************************************************************/
    
    /** Function to be called by something that wants to know the current
        status of this service.  Returns a JSON object that could be
        inspected by a human or consumed by a service.
    */
    virtual Json::Value getServiceStatus() const;

    /** Function that iterates over all children and adds their service
        status to the given JSON object with keys as their names.

        Mutates the result argument in-place.
    */
    virtual void addChildServiceStatus(Json::Value & result) const;

protected:
    std::shared_ptr<ServiceProxies> services_;
    std::string serviceName_;
    ServiceBase * parent_;
    std::vector<ServiceBase *> children_;
};


/*****************************************************************************/
/* SUB SERVICE BASE                                                          */
/*****************************************************************************/

struct SubServiceBase : public ServiceBase {
};

} // namespace Datacratic


#endif /* __service__service_base_h__ */
   
