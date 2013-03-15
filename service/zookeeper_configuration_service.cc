/** zookeeper_configuration_service.cc
    Jeremy Barnes, 26 September 2012
    Copyright (c) 2012 Datacratic Inc.  All rights reserved.

    Configuration service using Zookeeper.
*/

#include "zookeeper_configuration_service.h"
#include "soa/service/zookeeper.h"
#include "jml/utils/exc_assert.h"
#include <boost/algorithm/string.hpp>


using namespace std;
using namespace ML;

namespace Datacratic {

std::string printZookeeperEventType(int type)
{

    if (type == ZOO_CREATED_EVENT)
        return "CREATED";
    if (type == ZOO_DELETED_EVENT)
        return "DELETED";
    if (type == ZOO_CHANGED_EVENT)
        return "CHANGED";
    if (type == ZOO_CHILD_EVENT)
        return "CHILD";
    if (type == ZOO_SESSION_EVENT)
        return "SESSION";
    if (type == ZOO_NOTWATCHING_EVENT)
        return "NOTWATCHING";
    return ML::format("UNKNOWN(%d)", type);
}

std::string printZookeeperState(int state)
{
    if (state == ZOO_EXPIRED_SESSION_STATE)
        return "ZOO_EXPIRED_SESSION_STATE";
    if (state == ZOO_AUTH_FAILED_STATE)
        return "ZOO_AUTH_FAILED_STATE";
    if (state == ZOO_CONNECTING_STATE)
        return "ZOO_CONNECTING_STATE";
    if (state == ZOO_ASSOCIATING_STATE)
        return "ZOO_ASSOCIATING_STATE";
    if (state == ZOO_CONNECTED_STATE)
        return "ZOO_CONNECTED_STATE";
    return ML::format("ZOO_UNKNOWN_STATE(%d)", state);
}

void
watcherFn(int type, std::string const & path, void * watcherCtx)
{
    typedef std::shared_ptr<ConfigurationService::Watch::Data> SharedPtr;
    std::unique_ptr<SharedPtr> data(reinterpret_cast<SharedPtr *>(watcherCtx));
#if 0
    cerr << "type = " << printZookeeperEventType(type)
         << " state = " << printZookeeperState(state)
         << " path = " << path << " context "
         << watcherCtx << " data " << data->get() << endl;
#endif

    ConfigurationService::ChangeType change;
    if (type == ZOO_CREATED_EVENT)
        change = ConfigurationService::CREATED;
    if (type == ZOO_DELETED_EVENT)
        change = ConfigurationService::DELETED;
    if (type == ZOO_CHANGED_EVENT)
        change = ConfigurationService::VALUE_CHANGED;
    if (type == ZOO_CHILD_EVENT)
        change = ConfigurationService::NEW_CHILD;

    auto & item = *data;
    if (item->watchReferences > 0) {
        item->onChange(path, change);
    }
}

ZookeeperConnection::Callback::Type
getWatcherFn(const ConfigurationService::Watch & watch)
{
    if (!watch)
        return nullptr;
    return watcherFn;
}



/*****************************************************************************/
/* ZOOKEEPER CONFIGURATION SERVICE                                           */
/*****************************************************************************/

ZookeeperConfigurationService::
ZookeeperConfigurationService()
{
}

ZookeeperConfigurationService::
ZookeeperConfigurationService(const std::string & host,
                              const std::string & prefix,
                              int timeout)
{
    init(host, prefix, timeout);
}
    
ZookeeperConfigurationService::
~ZookeeperConfigurationService()
{
}

void
ZookeeperConfigurationService::
init(const std::string & host,
     const std::string & prefix,
     int timeout)
{
    zoo.reset(new ZookeeperConnection());
    zoo->connect(host, timeout);
    this->prefix = prefix;

    if (!this->prefix.empty()
        && this->prefix[this->prefix.size() - 1] != '/')
        this->prefix = this->prefix + "/";

    if (!this->prefix.empty()
        && this->prefix[0] != '/')
        this->prefix = "/" + this->prefix;
    
    zoo->createPath(this->prefix);

#if 0
    for (unsigned i = 1;  i < prefix.size();  ++i) {
        if (prefix[i] == '/') {
            zoo->createNode(string(prefix, 0, i),
                            "",
                            false,
                            false,
                            false /* must succeed */);
        }
    }
#endif
}

Json::Value
ZookeeperConfigurationService::
getJson(const std::string & key, Watch watch)
{
    ExcAssert(zoo);
    auto val = zoo->readNode(prefix + key, getWatcherFn(watch),
                             watch.get());
    try {
        if (val == "")
            return Json::Value();
        return Json::parse(val);
    } catch (...) {
        cerr << "error parsing JSON entry '" << val << "'" << endl;
        throw;
    }
}
    
void
ZookeeperConfigurationService::
set(const std::string & key,
    const Json::Value & value)
{
    //cerr << "setting " << key << " to " << value << endl;
    // TODO: race condition
    if (!zoo->createNode(prefix + key, boost::trim_copy(value.toString()),
                         false, false,
                         false /* must succeed */,
                         true /* create path */).second)
        zoo->writeNode(prefix + key, boost::trim_copy(value.toString()));
    ExcAssert(zoo);
}

std::string
ZookeeperConfigurationService::
setUnique(const std::string & key,
          const Json::Value & value)
{
    //cerr << "setting unique " << key << " to " << value << endl;
    ExcAssert(zoo);
    return zoo->createNode(prefix + key, boost::trim_copy(value.toString()),
                           true /* ephemeral */,
                           false /* sequential */,
                           true /* mustSucceed */,
                           true /* create path */)
        .first;
}

std::vector<std::string>
ZookeeperConfigurationService::
getChildren(const std::string & key,
            Watch watch)
{
    //cerr << "getChildren " << key << " watch " << watch << endl;
    return zoo->getChildren(prefix + key,
                            false /* fail if not there */,
                            getWatcherFn(watch),
                            watch.get());
}

bool
ZookeeperConfigurationService::
forEachEntry(const OnEntry & onEntry,
                      const std::string & startPrefix) const
{
    //cerr << "forEachEntry: startPrefix = " << startPrefix << endl;

    ExcAssert(zoo);

    std::function<bool (const std::string &)> doNode
        = [&] (const std::string & currentPrefix)
        {
            //cerr << "doNode " << currentPrefix << endl;

            string r = zoo->readNode(prefix + currentPrefix);

            //cerr << "r = " << r << endl;
            
            if (r != "") {
                if (!onEntry(currentPrefix, Json::parse(r)))
                    return false;
            }

            vector<string> children = zoo->getChildren(prefix + currentPrefix,
                                                       false);
            
            for (auto child: children) {
                //cerr << "child = " << child << endl;
                string newPrefix = currentPrefix + "/" + child;
                if (currentPrefix.empty())
                    newPrefix = child;
                
                if (!doNode(newPrefix))
                    return false;
            }
            
            return true;
        };

    if (!zoo->nodeExists(prefix + startPrefix)) {
        

        return true;
    }
    
    return doNode(startPrefix);
}

void
ZookeeperConfigurationService::
removePath(const std::string & path)
{
    ExcAssert(zoo);
    zoo->removePath(prefix + path);
}


} // namespace Datacratic
